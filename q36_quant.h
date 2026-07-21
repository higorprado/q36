#ifndef Q36_QUANT_H
#define Q36_QUANT_H

#include <stdbool.h>
#include <stdint.h>

float q36_quant_f16_to_f32(uint16_t h);
uint16_t q36_quant_f32_to_f16(float f);
void q36_quant_q8_0(const float *x, void *out, int64_t n);
void q36_quant_q4_0(const float *x, void *out, int64_t n);
void q36_quant_q8_k(const float *x, void *out, int64_t n);
float q36_quant_dot_q8_0(const void *a, const void *b, int n);
bool q36_quant_dequantize(uint32_t type, const void *src, float *out, uint32_t n);

#endif
