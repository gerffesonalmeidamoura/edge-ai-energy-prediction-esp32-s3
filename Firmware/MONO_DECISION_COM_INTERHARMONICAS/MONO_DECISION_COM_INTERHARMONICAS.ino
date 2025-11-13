// ==== DT_PCA_MONO_COM_INTERHARMONICAS_FP32.ino ====
// Benchmark de latência e energia para Decision Tree (FP32) gerado em C (header-only).
// Entradas: PCA com 50 componentes (float32). Sem TFLM / sem arena.
//
// Requisitos: 
//  - Biblioteca INA226 (Rob Tillaart v0.6.4)
//  - ESP32-S3 (pinos I2C ajustáveis abaixo)
//  - Arquivo "model_data.h" com: 
//      #define DT_INPUT_DIM 50
//      static inline float dt_predict(float f[]);

#include <Arduino.h>
#include <WiFi.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_task_wdt.h"

#include <Wire.h>
#include <INA226.h>   // Rob Tillaart v0.6.4

// ===== Config =====
#define MODEL_NAME   "DT_PCA_MONO_COM_INTERHARMONICAS_FP32"
#define MODEL_BYTES_EST (size_t)(15.05 * 1024)  // estimativa do analisador (~15.05 KB)
#define ARENA_SIZE   0            // sem TFLM
#define WARMUP       5
#define RUNS         60
#define PIN_PROBE    -1

// INA226 (ajuste se necessário)
#define I2C_SDA      8
#define I2C_SCL      9
#define INA226_ADDR  0x40
#define RSHUNT_OHM   0.1f

#include "model_data.h"  // deve expor: DT_INPUT_DIM e dt_predict(float f[])

// ===== Utils =====
static inline uint64_t now_us(){ return (uint64_t) esp_timer_get_time(); }

static void disable_radios(){
  WiFi.persistent(false);
  WiFi.disconnect(true,true);
  WiFi.mode(WIFI_OFF);
}

// ===== INA226 =====
INA226 ina(INA226_ADDR, &Wire);
static float baseline_mW = 0.0f;

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

// ===== Snapshot simples de memória =====
static void snapshotMem(const char* tag,
                        size_t& free_int, size_t& free_psram,
                        size_t& bigblk_int, size_t& bigblk_psram){
  free_int     = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  free_psram   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  bigblk_int   = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
  bigblk_psram = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  Serial.printf("MEM[%s] free_int=%u B | free_psram=%u B | bigblk_int=%u B | bigblk_psram=%u B\n",
                tag, (unsigned)free_int, (unsigned)free_psram,
                (unsigned)bigblk_int, (unsigned)bigblk_psram);
}

// ===== Estado do benchmark =====
enum Phase { PH_INIT, PH_WARMUP, PH_BASELINE, PH_RUN, PH_DONE };
static Phase phase = PH_INIT;
static int iter = 0;
static uint64_t best_us = UINT64_MAX, worst_us = 0, sum_us = 0;

// ===== Métricas de memória =====
static size_t free_int_before=0, free_psram_before=0;
static size_t free_int_after_arena=0, free_psram_after_arena=0;
static size_t free_int_after_alloc=0, free_psram_after_alloc=0;
static size_t bigblk_int_after_alloc=0, bigblk_psram_after_alloc=0;
static size_t arena_psram_bytes=0;
static size_t arena_used_bytes_est=0, arena_headroom_bytes=0;

static float features[DT_INPUT_DIM]; // buffer da entrada (PCA 50)

// ================== SETUP ==================
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

  // ===== Mem: antes de "arena" (não há arena, mas mantemos consistência) =====
  size_t bi, bp;
  snapshotMem("before_arena", free_int_before, free_psram_before, bi, bp);

  // Sem arena/TFLM → mantemos deltas zerados
  free_int_after_arena   = free_int_before;
  free_psram_after_arena = free_psram_before;
  arena_psram_bytes      = 0;
  snapshotMem("after_arena", free_int_after_arena, free_psram_after_arena, bi, bp);
  Serial.printf("ARENA requested=%u B | PSRAM delta (arena_psram_bytes)=%u B\n",
                (unsigned)ARENA_SIZE, (unsigned)arena_psram_bytes);

  // "AllocateTensors()" não se aplica. Consideramos que nada mudou.
  free_int_after_alloc     = free_int_after_arena;
  free_psram_after_alloc   = free_psram_after_arena;
  bigblk_int_after_alloc   = bi;
  bigblk_psram_after_alloc = bp;
  arena_used_bytes_est     = 0;
  arena_headroom_bytes     = 0;

  snapshotMem("after_AllocateTensors", free_int_after_alloc, free_psram_after_alloc,
              bigblk_int_after_alloc, bigblk_psram_after_alloc);
  Serial.printf("TFLM arena_used_bytes(est)=%u B | arena_headroom=%u B\n",
                (unsigned)arena_used_bytes_est, (unsigned)arena_headroom_bytes);

  // Info do "modelo"
  Serial.println();
  Serial.printf("=== %s ===\n", MODEL_NAME);
  Serial.printf("ARENA=%u B | RUNS=%d | WARMUP=%d\n", (unsigned)ARENA_SIZE, RUNS, WARMUP);
  Serial.printf("Input:  shape=[1,%d] | type=float32 | bytes=%d\n", DT_INPUT_DIM, (int)(DT_INPUT_DIM * sizeof(float)));
  Serial.printf("Output: shape=[1]    | type=float32 | bytes=%d\n", (int)sizeof(float));
  Serial.printf("model_len(header-estimado)=%u bytes\n", (unsigned)MODEL_BYTES_EST);

  // Preenche entrada com zeros (determinístico e suficiente p/ benchmark)
  for (int i=0;i<DT_INPUT_DIM;i++) features[i] = 0.0f;

  phase = PH_WARMUP; iter = 0;
  Serial.println("Aquecendo...");
}

// ================== LOOP ==================
void loop() {
  switch (phase) {
    case PH_WARMUP:
      if (iter < WARMUP) {
        (void)dt_predict(features);
        iter++; delay(0);
      } else {
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

        volatile float y = dt_predict(features);

        uint64_t t1 = now_us();
        float p1_mW = read_power_mW_calc();

        if (PIN_PROBE >= 0) digitalWrite(PIN_PROBE, LOW);

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

        (void)y; // evita otimização agressiva
        iter++;
        delay(0);
      } else {
        float mean_us = (float)sum_us / (float)RUNS;
        Serial.printf("RESULT: mean=%.2f us | best=%llu us | worst=%llu us\n",
                      mean_us, (unsigned long long)best_us, (unsigned long long)worst_us);

        // Desempenho
        Serial.println("CSV:model_name,model_bytes,arena_bytes,runs,mean_us,best_us,worst_us,baseline_mW");
        Serial.printf("CSV:%s,%u,%u,%d,%.2f,%llu,%llu,%.1f\n",
                      MODEL_NAME, (unsigned)MODEL_BYTES_EST, (unsigned)ARENA_SIZE, RUNS,
                      mean_us, (unsigned long long)best_us, (unsigned long long)worst_us, baseline_mW);

        // Memória
        Serial.println("CSV:mem,model_name,model_bytes,arena_bytes,"
                       "free_int_before,free_psram_before,"
                       "free_int_after_arena,free_psram_after_arena,"
                       "free_int_after_alloc,free_psram_after_alloc,"
                       "bigblk_int_after_alloc,bigblk_psram_after_alloc,"
                       "arena_psram_bytes,arena_used_bytes_est,arena_headroom_bytes");
        Serial.printf("CSV:mem,%s,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                      MODEL_NAME, (unsigned)MODEL_BYTES_EST, (unsigned)ARENA_SIZE,
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
