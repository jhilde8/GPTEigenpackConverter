#pragma once

#include <ostream>
#include "../../Utils/Typedefs.hpp"

struct Dimensions
{
    typedef uint64_t VolumeSize_t;

    const Count_t Ls=0;
    const Count_t  x=0;
    const Count_t  y=0;
    const Count_t  z=0;
    const Count_t  t=0;

    const VolumeSize_t volume( ) const noexcept { return this->Ls*this->x*this->y*this->z*this->t; }
    const VolumeSize_t volume4() const noexcept { return this->x*this->y*this->z*this->t; }
    const Count_t      sum()     const noexcept { return this->Ls+this->x+this->y+this->z+this->t; }
    const Count_t      sum4()    const noexcept { return this->x+this->y+this->z+this->t; }
};

Dimensions operator+(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator-(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator*(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator/(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator%(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator&&(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator||(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator>(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator>=(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator<(const Dimensions& dims1, const Dimensions& dims2);
Dimensions operator<=(const Dimensions& dims1, const Dimensions& dims2);
std::ostream& operator<<(std::ostream& os, const Dimensions& dims);
bool operator==(const Dimensions& dims1, const Dimensions& dims2);

Count_t flattenDimensionsLsSlow(const Dimensions& position, const Dimensions& boundary);
Count_t flattenDimensionsLsFast(const Dimensions& position, const Dimensions& boundary);

struct EigenvectorInfo
{
    const Count_t n_eigen=0;
    const Count_t n_basis_total=0;
    const Count_t n_basis_fp16=0;
    const Count_t n_basis_fp32=0;
    const Count_t total_blocks_per_file=0; // Product of blocks_per_file
    const Count_t n_coeffs_per_exponent=0;
};

struct FileLocationInfo
{
    Count_t directory=0;
    Count_t file=0;
};

std::ostream& operator<<(std::ostream& os, const FileLocationInfo& loc);

struct BasisDataLocationInfo
{
    Count_t block_idx;
    Count_t basis_vector_idx;
    Count_t block_site_idx;
};

std::ostream& operator<<(std::ostream& os, const BasisDataLocationInfo& loc);

struct CoarseVectorDataLocationInfo
{
    Count_t block_idx;
    Count_t eigenvector_idx;
};

std::ostream& operator<<(std::ostream& os, const CoarseVectorDataLocationInfo& loc);

struct BasisSizes
{
    Offset_t basis_vector_size;
    Offset_t block_size;
};

struct CoarseSizes
{
    Offset_t fp32_coefficients_size;
    Offset_t fp16_coefficients_size;
    Offset_t coarse_eigenvector_size;
    Offset_t blocks_for_eigenvector_size;
};

struct SectionSizes
{
    Offset_t fp32_section_size;
    Offset_t fp16_section_size;
    Offset_t coarse_section_size;
    Offset_t total_size;
};
