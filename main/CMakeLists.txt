
#
# Componente principal do exemplo TensorFlow Lite Micro 'hello_world'.
#

idf_component_register(
    SRCS main.cc main_functions.cc constants.cc output_handler.cc model.cc
    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS "")

# O esp-tflite-micro compila com -DESP_NN, que usa as instruções SIMD do
# ESP32-S3; o Wokwi não as emula e a inferência trava/sai errada. O -U vem
# depois do -D na linha de compilação e desliga a macro, fazendo o TFLM usar
# os kernels de referência em C++. Fica versionado aqui, sem editar
# managed_components/. Numa placa física, remova estas duas linhas.
idf_component_get_property(tflite_micro_lib espressif__esp-tflite-micro COMPONENT_LIB)
target_compile_options(${tflite_micro_lib} PRIVATE -UESP_NN)
