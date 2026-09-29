/*
 * Classificador de orientação com TensorFlow Lite Micro.
 *
 * Esconde do main.cc os detalhes do TFLM (modelo, op resolver, arena,
 * quantização). Entrada: aceleração em g; saída: classe + confiança.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

constexpr int kNumFeatures = 3;  // ax, ay, az
constexpr int kNumClasses = 3;

// Nomes na MESMA ordem de CLASSES em training/train.py
extern const char *const kLabels[kNumClasses];

struct ClassifierResult {
  int class_index;             // argmax da saída
  float confidence;            // probabilidade (softmax desquantizado) da classe
  int8_t input_q[kNumFeatures];  // entrada quantizada (para conferir com eval.py)
  int8_t output_q[kNumClasses];  // saída bruta int8 (idem)
};

esp_err_t classifier_init(void);
esp_err_t classifier_predict(const float accel_g[kNumFeatures],
                             ClassifierResult *result);

// Bytes da tensor arena realmente usados (medido após AllocateTensors)
size_t classifier_arena_used_bytes(void);
