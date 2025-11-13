#include <Arduino.h>
#include <WiFi.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include <math.h>

#include <Wire.h>
#include <INA226.h>   // Rob Tillaart v0.6.4

// ===== Modelo (árvore de decisão em C) =====
// Gere este header a partir do seu modelo DecisionTree + PCA.
// Deve expor:  #define DT_INPUT_DIM  <n>
//              static inline float dt_predict(float f[]);
#include "model_data.h"  // dt_predict(), DT_INPUT_DIM

// ===== Config =====
#define MODEL_NAME   "DT_PCA_MONO_FP32"   // ajuste se quiser
#define MODEL_BYTES  26144                // (~25.55 KB estimados) ajuste se medir outro valor
#define ARENA_SIZE   0                    // sem TFLM
#define WARMUP       5
#define RUNS         60
#define PIN_PROBE    -1

// INA226 (seus pinos)
#define I2C_SDA      8
#define I2C_SCL      9
#define INA226_ADDR  0x40
#define RSHUNT_OHM   0.1f

// ===== INA226 =====
INA226 ina(INA226_ADDR, &Wire);
static float baseline_mW = 0.0f;

// ===== Estado do benchmark =====
enum Phase { PH_INIT, PH_WARMUP, PH_BASELINE, PH_RUN, PH_DONE };
static Phase phase = PH_INIT;
static int iter = 0;
static uint64_t best_us = UINT64_MAX, worst_us = 0, sum_us = 0;

// ===== Métricas de memória =====
static size_t free_int_before=0, free_psram_before=0;
static size_t free_int_after_alloc=0, free_psram_after_alloc=0;
static size_t bigblk_int_after_alloc=0, bigblk_psram_after_alloc=0;

// ===== Utils =====
static inline uint64_t now_us(){ return (uint64_t) esp_timer_get_time(); }
static void disable_radios(){ WiFi.persistent(false); WiFi.disconnect(true,true); WiFi.mode(WIFI_OFF); }

static void snapshotMem(const char* tag){
  size_t fi = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  size_t fp = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t bi = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
  size_t bp = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  Serial.printf("MEM[%s] free_int=%u B | free_psram=%u B | bigblk_int=%u B | bigblk_psram=%u B\n",
                tag, (unsigned)fi, (unsigned)fp, (unsigned)bi, (unsigned)bp);
}

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

// ===== Entrada para o modelo =====
// Substitua este vetor por suas features PCA reais se desejar medir com dados reais.
// Para benchmark de tempo/energia, um vetor constante é suficiente.
static float g_features[DT_INPUT_DIM];

// Para evitar que o compilador otimize a chamada fora, acumule os resultados aqui.
static volatile float g_sink = 0.0f;

// Marcar como noinline ajuda a evitar inlining/otimizações agressivas.
__attribute__((noinline)) static float one_infer(const float* f) {
  // dt_predict aceita float f[]; precisamos de ponteiro não-const
  // pois a assinatura gerada pode não ser const-correct.
  float* p = const_cast<float*>(f);
  return dt_predict(p);
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

  // Memória antes
  free_int_before   = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  free_psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  snapshotMem("before_alloc");

  // "Alocação": nada a alocar; apenas inicialize o vetor de entrada
  for (int i=0;i<DT_INPUT_DIM;i++) g_features[i] = 0.0f; // use 0.0 ou seu vetor-medio PCA

  // Memória depois
  free_int_after_alloc     = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  free_psram_after_alloc   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  bigblk_int_after_alloc   = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
  bigblk_psram_after_alloc = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  snapshotMem("after_alloc");

  // Cabeçalho informativo (mantém padrão dos logs)
  Serial.println();
  Serial.printf("=== %s ===\n", MODEL_NAME);
  Serial.printf("ARENA=%u B | RUNS=%d | WARMUP=%d\n", (unsigned)ARENA_SIZE, RUNS, WARMUP);
  Serial.printf("Input:  shape=[1,%d] | type=float32\n", (int)DT_INPUT_DIM);
  Serial.printf("Output: shape=[1]    | type=float32\n");

  phase = PH_WARMUP; iter = 0;
  Serial.println("Aquecendo...");
}

void loop() {
  switch (phase) {
    case PH_WARMUP: {
      if (iter < WARMUP) {
        (void)one_infer(g_features);
        iter++; delay(0);
      } else {
        baseline_mW = measure_baseline_mW(200, 200);
        Serial.printf("Baseline potencia = %.1f mW\n", baseline_mW);
        phase = PH_RUN; iter = 0;
        Serial.println("CSV:run,dt_us,energy_uJ,energy_uJ_net,avgP_mW,avgP_mW_net");
      }
    } break;

    case PH_RUN: {
      if (iter < RUNS) {
        if (PIN_PROBE >= 0) digitalWrite(PIN_PROBE, HIGH);

        float p0_mW = read_power_mW_calc();
        uint64_t t0 = now_us();

        float y = one_infer(g_features);

        uint64_t t1 = now_us();
        float p1_mW = read_power_mW_calc();

        if (PIN_PROBE >= 0) digitalWrite(PIN_PROBE, LOW);

        // consome saída para evitar otimização
        g_sink += y * 1e-6f;

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

        // Desempenho
        Serial.println("CSV:model_name,model_bytes,arena_bytes,runs,mean_us,best_us,worst_us,baseline_mW");
        Serial.printf("CSV:%s,%d,%u,%d,%.2f,%llu,%llu,%.1f\n",
                      MODEL_NAME, (int)MODEL_BYTES, (unsigned)ARENA_SIZE, RUNS,
                      mean_us, (unsigned long long)best_us, (unsigned long long)worst_us, baseline_mW);

        // Memória (mantém o formato)
        Serial.println("CSV:mem,model_name,model_bytes,arena_bytes,"
                       "free_int_before,free_psram_before,"
                       "free_int_after_arena,free_psram_after_arena,"
                       "free_int_after_alloc,free_psram_after_alloc,"
                       "bigblk_int_after_alloc,bigblk_psram_after_alloc,"
                       "arena_psram_bytes,arena_used_bytes_est,arena_headroom_bytes");
        Serial.printf("CSV:mem,%s,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                      MODEL_NAME, (int)MODEL_BYTES, (unsigned)ARENA_SIZE,
                      (unsigned)free_int_before, (unsigned)free_psram_before,
                      (unsigned)free_int_before, (unsigned)free_psram_before,   // sem arena: iguais ao "before"
                      (unsigned)free_int_after_alloc, (unsigned)free_psram_after_alloc,
                      (unsigned)bigblk_int_after_alloc, (unsigned)bigblk_psram_after_alloc,
                      0u, 0u, 0u);

        phase = PH_DONE;
      }
    } break;

    default:
      delay(1000);
      break;
  }
}
