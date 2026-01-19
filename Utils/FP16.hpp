#pragma once

#include <array>
#include <cinttypes>

float unmap_fp16_exp(uint16_t e);
float fp16_unmap(int val, float min, float max);

template<size_t N>
std::array<float, N> unpackFP16s(uint16_t int_exp, uint16_t* vals)
{
    auto exp = unmap_fp16_exp(int_exp);
    std::array<float, N> out;
    for (auto i=0; i < N; ++i)
        out[i] = fp16_unmap(vals[i], -exp, exp);
    return out;
}

template<size_t N>
std::array<float, N> unpackFP16s(uint16_t* vals)
{
    return unpackFP16s<N>(vals[0], &vals[1]);
}
