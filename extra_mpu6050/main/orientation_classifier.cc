/*
 * Mesmo esqueleto do main_functions.cc do hello_world (GetModel -> op resolver
 * -> MicroInterpreter -> AllocateTensors -> Invoke), trocando:
 *   - 1 entrada escalar  -> 3 features (ax, ay, az)
 *   - 1 saída de regressão -> 3 probabilidades (softmax) + argmax
 *   - quantização por truncamento -> round + clamp (ver RELATORIO.md, 7.2)
 */
#include "orientation_classifier.h"

#include <cmath>

#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

const char *const kLabels[kNumClasses] = {"REPOUSO", "INCLINADO", "INVERTIDO"};

namespace {
const char *TAG = "classifier";

// O modelo tem ~130 parâmetros; as ativações cabem em < 1 KB. 4 KB dão folga.
// Ajuste pelo valor de classifier_arena_used_bytes() logado no boot.
constexpr int kTensorArenaSize = 4 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

int8_t quantize(float value, float scale, int zero_point) {
  // round (e não truncamento) + clamp na faixa do int8
  long q = std::lround(value / scale) + zero_point;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}
}  // namespace

esp_err_t classifier_init(void) {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Schema do modelo %lu != suportado %d",
             static_cast<unsigned long>(model->version()), TFLITE_SCHEMA_VERSION);
    return ESP_FAIL;
  }

  // Só as operações que o modelo usa (lista impressa pelo train.py em
  // models/train_summary.json -> "int8_ops"). Cada op a mais custa flash.
  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    ESP_LOGE(TAG, "Falha ao registrar operações");
    return ESP_FAIL;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() falhou: arena de %d bytes é pequena?",
             kTensorArenaSize);
    return ESP_FAIL;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Confere o contrato Python <-> firmware: int8 e dimensões certas
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      input->dims->data[input->dims->size - 1] != kNumFeatures ||
      output->dims->data[output->dims->size - 1] != kNumClasses) {
    ESP_LOGE(TAG, "Modelo incompatível: esperado int8 [1,%d] -> [1,%d]",
             kNumFeatures, kNumClasses);
    return ESP_FAIL;
  }

  ESP_LOGI(TAG, "Modelo: %d bytes | arena usada: %u de %d bytes", g_model_len,
           static_cast<unsigned>(interpreter->arena_used_bytes()),
           kTensorArenaSize);
  ESP_LOGI(TAG, "input : scale=%.8f zero_point=%ld",
           static_cast<double>(input->params.scale),
           static_cast<long>(input->params.zero_point));
  ESP_LOGI(TAG, "output: scale=%.8f zero_point=%ld",
           static_cast<double>(output->params.scale),
           static_cast<long>(output->params.zero_point));
  return ESP_OK;
}

esp_err_t classifier_predict(const float accel_g[kNumFeatures],
                             ClassifierResult *result) {
  if (interpreter == nullptr || result == nullptr) return ESP_ERR_INVALID_STATE;

  for (int i = 0; i < kNumFeatures; i++) {
    result->input_q[i] = quantize(accel_g[i], input->params.scale,
                                  input->params.zero_point);
    input->data.int8[i] = result->input_q[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGW(TAG, "Invoke() falhou");
    return ESP_FAIL;
  }

  // argmax direto no int8: a desquantização é crescente, não muda o vencedor
  int best = 0;
  for (int i = 0; i < kNumClasses; i++) {
    result->output_q[i] = output->data.int8[i];
    if (result->output_q[i] > result->output_q[best]) best = i;
  }
  result->class_index = best;
  result->confidence =
      (result->output_q[best] - output->params.zero_point) * output->params.scale;
  return ESP_OK;
}

size_t classifier_arena_used_bytes(void) {
  return interpreter ? interpreter->arena_used_bytes() : 0;
}
