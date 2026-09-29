#
# Componente principal: leitura do MPU6050 (C) + classificador TFLM (C++).
#

idf_component_register(
    SRCS main.cc sensor_mpu6050.c orientation_classifier.cc model.cc
    PRIV_REQUIRES driver
    INCLUDE_DIRS "")

# O esp-tflite-micro compila com -DESP_NN (instruções SIMD do ESP32-S3), que o
# Wokwi não emula. O -U vem depois do -D e desliga a macro: o TFLM usa os
# kernels de referência em C++. Numa placa física, remova estas duas linhas.
idf_component_get_property(tflite_micro_lib espressif__esp-tflite-micro COMPONENT_LIB)
target_compile_options(${tflite_micro_lib} PRIVATE -UESP_NN)

# Armadilha conhecida (ESP-IDF >= 5.3): o espressif/mpu6050 inclui o driver I2C
# legado (driver/i2c.h), que ficou no componente `driver`, mas não o declara no
# REQUIRES. Em vez de editar managed_components/ (não versionada), a
# dependência é ligada aqui.
idf_component_get_property(mpu6050_lib espressif__mpu6050 COMPONENT_LIB)
target_link_libraries(${mpu6050_lib} PUBLIC idf::driver)
