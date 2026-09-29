/*
 * Leitura do MPU6050 com o componente espressif/mpu6050 (driver I2C legado).
 * Mesmo fluxo da prática 2: I2C mestre -> create -> config -> wake_up -> WHO_AM_I.
 *
 * Ligação (ver diagram.json e Kconfig.projbuild):
 *   MPU6050 VCC -> 3V3 | GND -> GND | SDA -> GPIO8 | SCL -> GPIO9
 */
#include "sensor_mpu6050.h"

#include <stddef.h>

#include "driver/i2c.h"
#include "esp_check.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include "mpu6050.h"

static const char *TAG = "sensor_mpu6050";

#define I2C_MASTER_SDA_IO   CONFIG_I2C_MASTER_SDA_IO
#define I2C_MASTER_SCL_IO   CONFIG_I2C_MASTER_SCL_IO
#define I2C_MASTER_FREQ_HZ  CONFIG_I2C_MASTER_FREQ_HZ
#define I2C_MASTER_NUM      I2C_NUM_0

static mpu6050_handle_t s_mpu = NULL;

static esp_err_t i2c_bus_init(void)
{
    /* I2C é dreno aberto: sem pull-up a linha nunca volta para nível alto. */
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
        .clk_flags = I2C_SCLK_SRC_FLAG_FOR_NOMAL,
    };
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_MASTER_NUM, &conf), TAG,
                        "i2c_param_config falhou");
    return i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
}

esp_err_t sensor_mpu6050_init(void)
{
    ESP_RETURN_ON_ERROR(i2c_bus_init(), TAG, "falha ao iniciar o I2C");
    ESP_LOGI(TAG, "I2C pronto (SDA=GPIO%d, SCL=GPIO%d, %d Hz)",
             I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO, I2C_MASTER_FREQ_HZ);

    /* 0x68 = AD0 em nível baixo (padrão do Wokwi) */
    s_mpu = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);
    ESP_RETURN_ON_FALSE(s_mpu != NULL, ESP_FAIL, TAG, "mpu6050_create falhou");

    /* +/-4 g: MESMA faixa assumida no dataset sintético do train.py.
     * Parado, o sensor mede ~1 g; 4 g dá folga para trancos sem saturar. */
    ESP_RETURN_ON_ERROR(mpu6050_config(s_mpu, ACCE_FS_4G, GYRO_FS_500DPS), TAG,
                        "mpu6050_config falhou");
    /* O MPU6050 liga em sleep: sem wake_up as leituras vêm zeradas. */
    ESP_RETURN_ON_ERROR(mpu6050_wake_up(s_mpu), TAG, "mpu6050_wake_up falhou");

    uint8_t who_am_i = 0;
    ESP_RETURN_ON_ERROR(mpu6050_get_deviceid(s_mpu, &who_am_i), TAG,
                        "falha ao ler WHO_AM_I");
    ESP_LOGI(TAG, "MPU6050 WHO_AM_I: 0x%02x (esperado: 0x68)", who_am_i);
    return ESP_OK;
}

esp_err_t sensor_mpu6050_read_accel(float accel_g[3])
{
    ESP_RETURN_ON_FALSE(s_mpu != NULL, ESP_ERR_INVALID_STATE, TAG,
                        "sensor não inicializado");
    mpu6050_acce_value_t acce = {0};
    ESP_RETURN_ON_ERROR(mpu6050_get_acce(s_mpu, &acce), TAG, "leitura falhou");
    accel_g[0] = acce.acce_x;
    accel_g[1] = acce.acce_y;
    accel_g[2] = acce.acce_z;
    return ESP_OK;
}
