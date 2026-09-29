/*
 * Modelo .tflite embarcado como array C. O model.cc é GERADO pelo
 * training/train.py (substitui o `xxd -i` do hello_world).
 */
#pragma once

extern const unsigned char g_model[];
extern const int g_model_len;
