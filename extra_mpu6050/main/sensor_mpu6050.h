/*
 * Camada de hardware: leitura do acelerômetro do MPU6050 via I2C.
 *
 * Escrita em C (como na prática 2) e exposta ao main.cc (C++) com
 * `extern "C"`: sem isso o compilador C++ "decora" os nomes das funções e o
 * linker não as encontra.
 */
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Inicializa o barramento I2C, acorda o MPU6050 e confere o WHO_AM_I. */
esp_err_t sensor_mpu6050_init(void);

/* Lê a aceleração em g. accel_g[0..2] = ax, ay, az. */
esp_err_t sensor_mpu6050_read_accel(float accel_g[3]);

#ifdef __cplusplus
}
#endif
