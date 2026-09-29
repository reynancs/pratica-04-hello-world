/*
 * Prática 4 (extra) — Classificador de orientação com MPU6050 + TFLite Micro
 * UC: IA Embarcada e Modelos Compactos — UniSENAI
 *
 * Aplicação: monitor de postura de um equipamento (ex.: cilindro de gás,
 * carrinho/AGV, gabinete). O MPU6050 fica preso ao equipamento e um modelo
 * int8 classifica a leitura do acelerômetro em:
 *   REPOUSO   (theta < 30 graus)   -> posição normal de operação
 *   INCLINADO (30 a 120 graus)     -> alerta: base cedendo, carga desbalanceada
 *   INVERTIDO (theta > 120 graus)  -> alarme: equipamento tombou/capotou
 *
 * Fluxo: app_main -> init sensor -> init classificador -> self-test com
 * vetores fixos (comparar com `python eval.py`) -> loop leitura/inferência.
 *
 * Wokwi: clique no MPU6050 durante a simulação e mova os sliders de
 * aceleração (accelX/Y/Z) para ver a classe mudar.
 */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "sdkconfig.h"

#include "orientation_classifier.h"
#include "sensor_mpu6050.h"

namespace {
const char *TAG = "orientacao";

// Mesmos vetores de GOLDEN_VECTORS em training/eval.py
struct GoldenVector {
  const char *description;
  float accel_g[kNumFeatures];
};
const GoldenVector kGoldenVectors[] = {
    {"plana, face p/ cima (0 graus)", {0.0f, 0.0f, 1.0f}},
    {"inclinada 45 graus", {0.707f, 0.0f, 0.707f}},
    {"de lado (90 graus)", {1.0f, 0.0f, 0.0f}},
    {"de cabeca p/ baixo (180 graus)", {0.0f, 0.0f, -1.0f}},
};

// Self-test: prova que o modelo embarcado responde igual ao .tflite avaliado
// no PC. Os valores in_q/out_q devem bater com a saída do eval.py.
void run_self_test(void) {
  ESP_LOGI(TAG, "=== Self-test (compare com python eval.py) ===");
  for (const GoldenVector &g : kGoldenVectors) {
    ClassifierResult r = {};
    if (classifier_predict(g.accel_g, &r) != ESP_OK) {
      ESP_LOGW(TAG, "Self-test falhou em '%s'", g.description);
      continue;
    }
    ESP_LOGI(TAG, "%-32s in_q=[%d, %d, %d] out_q=[%d, %d, %d] -> %s",
             g.description, r.input_q[0], r.input_q[1], r.input_q[2],
             r.output_q[0], r.output_q[1], r.output_q[2],
             kLabels[r.class_index]);
  }
}
}  // namespace

// extern "C": o ESP-IDF chama app_main a partir de código C
extern "C" void app_main(void) {
  // Setup: sem sensor ou sem modelo não há o que fazer -> aborta (reinicia)
  ESP_ERROR_CHECK(sensor_mpu6050_init());
  ESP_ERROR_CHECK(classifier_init());
  run_self_test();

  ESP_LOGI(TAG, "Classificando a cada %d ms. Mova os sliders do MPU6050.",
           CONFIG_INFERENCE_PERIOD_MS);

  int last_class = -1;
  while (true) {
    float accel_g[kNumFeatures] = {0};
    ClassifierResult r = {};

    // No loop, falha vira aviso: uma leitura ruim não derruba o dispositivo
    if (sensor_mpu6050_read_accel(accel_g) != ESP_OK) {
      ESP_LOGW(TAG, "Falha ao ler o acelerômetro");
    } else if (classifier_predict(accel_g, &r) != ESP_OK) {
      ESP_LOGW(TAG, "Falha na inferência");
    } else {
      ESP_LOGI(TAG, "ACC[g] x=%6.2f y=%6.2f z=%6.2f -> %-9s (%.0f%%)",
               static_cast<double>(accel_g[0]), static_cast<double>(accel_g[1]),
               static_cast<double>(accel_g[2]), kLabels[r.class_index],
               static_cast<double>(r.confidence * 100.0f));

      // Evento só na TRANSIÇÃO de estado: é o que uma aplicação real
      // publicaria (MQTT, buzzer, LED) em vez de repetir a cada leitura.
      if (r.class_index != last_class) {
        if (r.class_index == 0) {
          ESP_LOGI(TAG, ">>> Estado: REPOUSO (operação normal)");
        } else if (r.class_index == 1) {
          ESP_LOGW(TAG, ">>> ALERTA: equipamento INCLINADO");
        } else {
          ESP_LOGE(TAG, ">>> ALARME: equipamento INVERTIDO/tombado");
        }
        last_class = r.class_index;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(CONFIG_INFERENCE_PERIOD_MS));
  }
}
