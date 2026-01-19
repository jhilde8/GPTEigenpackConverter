#include <cmath>

#include "FP16.hpp"

static const uint16_t USHORT_MAX = 0xFFFF;
static const double FP16_BASE = 1.4142135623730950488;


float unmap_fp16_exp(uint16_t e)
{
    float de = (float)((int)e - USHORT_MAX / 2);
    return ::pow( FP16_BASE, de );
    //auto exponent = static_cast<float>(static_cast<int>e - USHORT_MAX / 2);
    //return std::pow( FP16_BASE, exponent );
}

float fp16_unmap(int val, float min, float max)
{
    return min + (float)(val + 0.5) * (max - min)  / (float)( USHORT_MAX + 1 );
    //return min + static_cast<float>(val + 0.5) * (max - min)  / static_cast<float>( N + 1 );
}
