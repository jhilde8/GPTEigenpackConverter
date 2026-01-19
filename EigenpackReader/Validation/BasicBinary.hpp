#pragma once

#include <array>
#include <fstream>
#include <string>
#include <vector>

#include <Grid/Grid.h>
#include <Hadrons/EigenPack.hpp>

#include "../Metadata/MetadataStructs.hpp"

class BasicBinary
{
private:
    const std::string filepath;
    std::ofstream bytestream;
public:
    BasicBinary(const std::string& filepath);
    ~BasicBinary();
    template<typename T>
    void write(const T& evec, const Dimensions& dims);
    template<typename T>
    void writeCoarse(const T& evec, const Dimensions& dims, Count_t n_vectors);
    template<typename T>
    void writeEvals(const T& evals);
};

template<typename T>
void BasicBinary::write(const T& evec, const Dimensions& dims)
{
    using namespace Grid;

    iSpinColourVector<ComplexF> fermion_buffer;
    auto count = dims.volume() / 2;
    char pad = 0;
    this->bytestream.write(reinterpret_cast<char*>(&count), sizeof(count));
    auto align_bytes = (0x10 - (this->bytestream.tellp() % 0x10)) % 0x10;
    for (auto i=0; i < align_bytes; ++i)
        this->bytestream.write(&pad, 1);

    for (int Ls=0; Ls < dims.Ls; ++Ls)
    for (int  x=0;  x < dims.x; ++x)
    for (int  y=0;  y < dims.y; ++y)
    for (int  z=0;  z < dims.z; ++z)
    for (int  t=0;  t < dims.t; ++t)
    {
        if (!((x + y + z + t) % 2))
            continue;
        Coordinate site({Ls, x, y, z, t});
        peekLocalSite(fermion_buffer, evec, site);
        for (int spin=0; spin < 4; ++spin)
        for (int colour=0; colour < 3; ++colour)
        {
            auto data = fermion_buffer()(spin)(colour);
            float rdata = data.real();
            float idata = data.imag();
            this->bytestream.write(reinterpret_cast<char*>(&rdata), sizeof(rdata));
            this->bytestream.write(reinterpret_cast<char*>(&idata), sizeof(idata));
        }
    }
}

template<typename T>
void BasicBinary::writeCoarse(const T& evec, const Dimensions& dims, Count_t n_vectors)
{
    using namespace Grid;

    std::vector<float> coef_buffer;
    coef_buffer.resize(n_vectors*2);

    auto count = dims.volume();
    char pad = 0;
    this->bytestream.write(reinterpret_cast<char*>(&count), sizeof(count));
    auto align_bytes = (0x10 - (this->bytestream.tellp() % 0x10)) % 0x10;
    for (auto i=0; i < align_bytes; ++i)
        this->bytestream.write(&pad, 1);

    for (int  x=0;  x < dims.x; ++x)
    for (int  y=0;  y < dims.y; ++y)
    for (int  z=0;  z < dims.z; ++z)
    for (int  t=0;  t < dims.t; ++t)
    {
        Coordinate site({x, y, z, t});
        iVector<ComplexF, 30> coef_buffer;
        peekLocalSite(coef_buffer, evec, site);
        for (const auto& bv : coef_buffer)
        {
            auto rdata = bv.real();
            auto idata = bv.imag();
            this->bytestream.write(reinterpret_cast<char*>(&rdata), sizeof(rdata));
            this->bytestream.write(reinterpret_cast<char*>(&idata), sizeof(idata));
        }
    }
}

template<typename T>
void BasicBinary::writeEvals(const T& evals)
{
    using namespace Grid;

    auto count = evals.size();
    char pad = 0;
    this->bytestream.write(reinterpret_cast<char*>(&count), sizeof(count));
    auto align_bytes = (0x10 - (this->bytestream.tellp() % 0x10)) % 0x10;
    for (auto i=0; i < align_bytes; ++i)
        this->bytestream.write(&pad, 1);

    for (const auto& eval : evals)
    {
        this->bytestream.write(reinterpret_cast<char*>(&eval), sizeof(eval));
    }
}
