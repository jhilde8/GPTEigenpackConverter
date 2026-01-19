#pragma once

#include <cinttypes>

#include "MetadataStructs.hpp"
#include "../../Utils/Typedefs.hpp"

struct SiteBlock
{
  const Dimensions fine_origin; // The first site in the block in the fine lattice
  const Dimensions coarse_site; // The coordinate of the site the block corresponds to on the coarse lattice
  const bool is_even_origin;    // Whether fine_origin is even or odd
  const std::vector<Dimensions>& sites_in_block;
  const Offset_t offset_basis_fp32; // The offset in the file corresponding to the FP32 part of the block basis
  const Offset_t offset_basis_fp16; // The offset in the file corresponding to the FP16 part of the block basis
  const Offset_t offset_coefficients; // The offset in the file corresponding to the basis coefficients
};
