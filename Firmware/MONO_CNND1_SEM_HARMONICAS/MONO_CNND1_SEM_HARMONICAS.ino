#include <Arduino.h>
#include <WiFi.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_task_wdt.h"

#include <Wire.h>
#include <INA226.h>   // Rob Tillaart v0.6.4

// ===== Config =====
#define MODEL_NAME  "CNND1_MONO_SEM_HARMONICAS_INT8"
#define ARENA_SIZE  98304           // ~96 KB (sugerido 89.6 KB + folga)
#define WARMUP      5
#define RUNS        60
#define PIN_PROBE   -1

// INA226 (pinos do seu setup)
#define I2C_SDA     8
#define I2C_SCL     9
#define INA226_ADDR 0x40
#define RSHUNT_OHM  0.1f

#include "model_data.h"  // deve expor: alignas(16) const unsigned char g_model[]; const int g_model_len;

// ===== TFLM =====
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#ifndef TFLITE_SCHEMA_VERSION
#  define TFLITE_SCHEMA_VERSION 3
#endif

// ===== Utils =====
static inline uint64_t now_us(){ return (uint64_t) esp_timer_get_time(); }
static const char* tname(TfLiteType t){
  switch(t){
    case kTfLiteFloat32: return "float32";
    case kTfLiteInt8:    return "int8";
    case kTfLiteUInt8:   return "uint8";
    case kTfLiteInt16:   return "int16";
    default:             return "?";
  }
}
static void print_shape(const TfLiteTensor* t){
  if (!t || !t->dims) { Serial.print("[]"); return; }
  Serial.print("[");
  for (int i=0;i<t->dims->size;i++){
    Serial.print(t->dims->data[i]);
    if (i+1<t->dims->size) Serial.print(", ");
  }
  Serial.print("]");
}
static void disable_radios(){ WiFi.persistent(false); WiFi.disconnect(true,true); WiFi.mode(WIFI_OFF); }

// ===== Estado do benchmark =====
enum Phase { PH_INIT, PH_WARMUP, PH_BASELINE, PH_RUN, PH_DONE };
static Phase phase = PH_INIT;
static int iter = 0;
static uint64_t best_us = UINT64_MAX, worst_us = 0, sum_us = 0;

static uint8_t* tensor_arena = nullptr;
static tflite::MicroInterpreter* interp = nullptr;

// ===== INA226 =====
INA226 ina(INA226_ADDR, &Wire);
static float baseline_mW = 0.0f;

// ===== Métricas de memória =====
static size_t free_int_before=0, free_psram_before=0;
static size_t free_int_after_arena=0, free_psram_after_arena=0;
static size_t free_int_after_alloc=0, free_psram_after_alloc=0;
static size_t bigblk_int_after_alloc=0, bigblk_psram_after_alloc=0;
static size_t arena_psram_bytes=0;
static size_t arena_used_bytes_est=0, arena_headroom_bytes=0;

// ---- Energia / Potência ----
static inline float read_power_mW_calc() {
  const float vbus_V    = ina.getBusVoltage();
  const float vshunt_mV = ina.getShuntVoltage_mV();
  const float i_mA      = fabsf(vshunt_mV) / RSHUNT_OHM;
  return vbus_V * i_mA;  // mW
}
static float measure_baseline_mW(uint16_t samples = 200, uint32_t us_between = 200) {
  float sum = 0;
  for (uint16_t i=0;i<samples;i++){
    sum += read_power_mW_calc();
    delayMicroseconds(us_between);
  }
  return sum / samples;
}

// Dump de ops (diagnóstico)
static void dump_ops(const tflite::Model* model){
  auto subgraphs = model->subgraphs();
  if (!subgraphs || subgraphs->size() == 0) { Serial.println("Sem subgraphs."); return; }
  auto sub = subgraphs->Get(0);
  auto ops = sub->operators();
  auto codes = model->operator_codes();

  Serial.println("OPS do modelo (ordem no grafo):");
  for (uint32_t i=0; i<ops->size(); i++){
    auto op = ops->Get(i);
    int oc_idx = op->opcode_index();
    auto oc = codes->Get(oc_idx);
    auto bc = oc->builtin_code();
    const char* name = tflite::EnumNameBuiltinOperator((tflite::BuiltinOperator)bc);
    Serial.printf("  %2u: %s (%d)\n", (unsigned)i, name ? name : "UNKNOWN", (int)bc);
  }
}

// Snapshot simples de memória
static void snapshotMem(const char* tag){
  size_t fi = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  size_t fp = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t bi = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
  size_t bp = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  Serial.printf("MEM[%s] free_int=%u B | free_psram=%u B | bigblk_int=%u B | bigblk_psram=%u B\n",
                tag, (unsigned)fi, (unsigned)fp, (unsigned)bi, (unsigned)bp);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  if (PIN_PROBE >= 0) { pinMode(PIN_PROBE, OUTPUT); digitalWrite(PIN_PROBE, LOW); }

  disable_radios();
  esp_task_wdt_deinit();
  disableLoopWDT();
  disableCore0WDT();
  disableCore1WDT();

  // INA226
  Wire.begin(I2C_SDA, I2C_SCL, 400000);
  if (!ina.begin()) { Serial.println("INA226 nao encontrado. Confira I2C/endereco."); while (1) delay(1000); }
  ina.reset();
  (void)ina.setMaxCurrentShunt(1.5f, RSHUNT_OHM, true);
  ina.setAverage(INA226_1_SAMPLE);
  ina.setBusVoltageConversionTime(INA226_140_us);
  ina.setShuntVoltageConversionTime(INA226_140_us);
  ina.setModeShuntBusContinuous();

  // ===== Memória: antes da arena =====
  free_int_before   = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  free_psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  snapshotMem("before_arena");

  // Arena em PSRAM + watermark
  tensor_arena = (uint8_t*) heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) { Serial.printf("Falha ao alocar %u bytes na PSRAM.\n", (unsigned)ARENA_SIZE); while (1) delay(1000); }
  memset(tensor_arena, 0xA5, ARENA_SIZE);

  // ===== Memória: após arena =====
  free_int_after_arena   = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  free_psram_after_arena = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  arena_psram_bytes      = (free_psram_before > free_psram_after_arena) ? (free_psram_before - free_psram_after_arena) : 0;
  snapshotMem("after_arena");
  Serial.printf("ARENA requested=%u B | PSRAM delta (arena_psram_bytes)=%u B\n",
                (unsigned)ARENA_SIZE, (unsigned)arena_psram_bytes);

  // Modelo
  const tflite::Model* model = tflite::GetModel(g_model);
  Serial.printf("model_len(header)=%d bytes\n", (int)g_model_len);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Schema mismatch: model %d vs TFLite %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    while (1) delay(1000);
  }

  dump_ops(model);

  // ===== Resolver para CNND1 INT8 (superset, inclui SHAPE etc.) =====
  static tflite::MicroMutableOpResolver<29> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddAveragePool2D();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddPad();
  resolver.AddUnpack();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddTanh();
  resolver.AddLogistic();
  resolver.AddFullyConnected();
  resolver.AddQuantize();
  resolver.AddDequantize();          // segurança
  resolver.AddSoftmax();             // caso a cabeça use

  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, ARENA_SIZE);
  interp = &interpreter;

  if (kTfLiteOk != interp->AllocateTensors()) {
    Serial.println("AllocateTensors FAILED. Tente ARENA_SIZE = 106496 (+8 KB) ou 114688 (+16 KB).");
    while (1) delay(1000);
  }

  TfLiteTensor* input  = interp->input(0);
  TfLiteTensor* output = interp->output(0);

  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    Serial.println("ERRO: esperado I/O INT8 para este modelo.");
    while (1) delay(1000);
  }

  // ===== Memória: após AllocateTensors() =====
  free_int_after_alloc     = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  free_psram_after_alloc   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  bigblk_int_after_alloc   = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
  bigblk_psram_after_alloc = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);

  // Uso real da arena (watermark)
  arena_used_bytes_est = 0;
  for (int i = (int)ARENA_SIZE - 1; i >= 0; --i) {
    if (tensor_arena[i] != 0xA5) { arena_used_bytes_est = (size_t)(i + 1); break; }
  }
  arena_headroom_bytes = (ARENA_SIZE > arena_used_bytes_est) ? (ARENA_SIZE - arena_used_bytes_est) : 0;

  snapshotMem("after_AllocateTensors");
  Serial.printf("TFLM arena_used_bytes(est)=%u B | arena_headroom=%u B\n",
                (unsigned)arena_used_bytes_est, (unsigned)arena_headroom_bytes);

  Serial.println();
  Serial.printf("=== %s ===\n", MODEL_NAME);
  Serial.printf("ARENA=%u B | RUNS=%d | WARMUP=%d\n", (unsigned)ARENA_SIZE, RUNS, WARMUP);
  Serial.print("Input:  shape="); print_shape(input);
  Serial.printf(" | type=%s | bytes=%d | q=(scale=%.6g, zp=%d)\n",
                tname(input->type), input->bytes, input->params.scale, input->params.zero_point);
  Serial.print("Output: shape="); print_shape(output);
  Serial.printf(" | type=%s | bytes=%d | q=(scale=%.6g, zp=%d)\n",
                tname(output->type), output->bytes, output->params.scale, output->params.zero_point);

  // Preenche entrada com zero lógico (zp ≈ -93 do diagnóstico)
  memset(input->data.raw, input->params.zero_point, input->bytes);

  phase = PH_WARMUP; iter = 0;
  Serial.println("Aquecendo...");
}

void loop() {
  if (!interp) { delay(10); return; }

  switch (phase) {
    case PH_WARMUP:
      if (iter < WARMUP) { (void)interp->Invoke(); iter++; delay(0); }
      else {
        baseline_mW = measure_baseline_mW(200, 200);
        Serial.printf("Baseline potencia = %.1f mW\n", baseline_mW);
        phase = PH_RUN; iter = 0;
        Serial.println("CSV:run,dt_us,energy_uJ,energy_uJ_net,avgP_mW,avgP_mW_net");
      }
      break;

    case PH_RUN:
      if (iter < RUNS) {
        if (PIN_PROBE >= 0) digitalWrite(PIN_PROBE, HIGH);

        float p0_mW = read_power_mW_calc();
        uint64_t t0 = now_us();

        TfLiteStatus st = interp->Invoke();

        uint64_t t1 = now_us();
        float p1_mW = read_power_mW_calc();

        if (PIN_PROBE >= 0) digitalWrite(PIN_PROBE, LOW);

        if (st != kTfLiteOk) { Serial.printf("Invoke falhou em r=%d\n", iter); phase = PH_DONE; break; }

        uint64_t dt_us = t1 - t0;
        if (dt_us < best_us) best_us = dt_us;
        if (dt_us > worst_us) worst_us = dt_us;
        sum_us += dt_us;

        // Energia por trapézio (mW·us → µJ)
        float e_uJ = 0.5f * (p0_mW + p1_mW) * (dt_us * 1e-3f);
        float avgP_mW = e_uJ / (dt_us * 1e-3f);

        // Remoção do baseline
        float e_idle_uJ   = baseline_mW * (dt_us * 1e-3f);
        float e_net_uJ    = e_uJ - e_idle_uJ;
        float avgP_net_mW = avgP_mW - baseline_mW;

        Serial.print("CSV:");
        Serial.print(iter);                 Serial.print(',');
        Serial.print((unsigned long)dt_us); Serial.print(',');
        Serial.print(e_uJ, 1);              Serial.print(',');
        Serial.print(e_net_uJ, 1);          Serial.print(',');
        Serial.print(avgP_mW, 1);           Serial.print(',');
        Serial.println(avgP_net_mW, 1);

        iter++;
        delay(0);
      } else {
        float mean_us = (float)sum_us / (float)RUNS;
        Serial.printf("RESULT: mean=%.2f us | best=%llu us | worst=%llu us\n",
                      mean_us, (unsigned long long)best_us, (unsigned long long)worst_us);

        // Desempenho (linha-resumo)
        Serial.println("CSV:model_name,model_bytes,arena_bytes,runs,mean_us,best_us,worst_us,baseline_mW");
        Serial.printf("CSV:%s,%d,%u,%d,%.2f,%llu,%llu,%.1f\n",
                      MODEL_NAME, g_model_len, (unsigned)ARENA_SIZE, RUNS,
                      mean_us, (unsigned long long)best_us, (unsigned long long)worst_us, baseline_mW);

        // Memória (linha-resumo para planilha)
        Serial.println("CSV:mem,model_name,model_bytes,arena_bytes,"
                       "free_int_before,free_psram_before,"
                       "free_int_after_arena,free_psram_after_arena,"
                       "free_int_after_alloc,free_psram_after_alloc,"
                       "bigblk_int_after_alloc,bigblk_psram_after_alloc,"
                       "arena_psram_bytes,arena_used_bytes_est,arena_headroom_bytes");
        Serial.printf("CSV:mem,%s,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                      MODEL_NAME, g_model_len, (unsigned)ARENA_SIZE,
                      (unsigned)free_int_before, (unsigned)free_psram_before,
                      (unsigned)free_int_after_arena, (unsigned)free_psram_after_arena,
                      (unsigned)free_int_after_alloc, (unsigned)free_psram_after_alloc,
                      (unsigned)bigblk_int_after_alloc, (unsigned)bigblk_psram_after_alloc,
                      (unsigned)arena_psram_bytes, (unsigned)arena_used_bytes_est, (unsigned)arena_headroom_bytes);

        phase = PH_DONE;
      }
      break;

    default:
      delay(1000);
      break;
  }
}
