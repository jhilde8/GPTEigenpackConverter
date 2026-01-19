#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "MetadataStructs.hpp"
#include "SiteBlock.hpp"
#include "../../Utils/Typedefs.hpp"

class Metadata
{
private:
    using TOffsetGetter = const Offset_t(Metadata::*)(const Dimensions&, Count_t) const;
    
    // Constructor helpers
    // Size counts
    const size_t getFP16CoeffChunkSize() const noexcept;
    const size_t getFP16CoeffsSize() const noexcept;
    // Block and block size generators
    std::vector<SiteBlock> generateBlocks();
    template<bool make_even>
    std::vector<Dimensions> generateSitesInBlock();
public:
    typedef std::unordered_map<std::string, Count_t> MetadataLookup;

    Metadata(const Dimensions&, const MetadataLookup&);

    // ############ //
    // DATA MEMBERS //
    // ############ //
    // Remember: Initialisation order is extremely important. Do not reorder these!
    // C++ initialises variables in the order they are defined in the class definition!
    // Static variables
    static const size_t complex_size = 2*sizeof(float);
    static const Count_t n_fermion_elements = 4*3; // Spin * Colour
    static const size_t fp32_fermion_size = Metadata::n_fermion_elements*Metadata::complex_size;
    static const size_t fp16_fermion_size = sizeof(uint16_t)*(2*Metadata::n_fermion_elements + 1); // Additional element for the exponent
    static const size_t fp32_coefficient_size = Metadata::complex_size;
    static const size_t fp16_coefficient_size = 2*sizeof(uint16_t);
    static const Count_t num_directories = 32;

    // Instance-specific variables - contained in file
    const Dimensions grid_size;
    const Dimensions sites_per_file;
    const Dimensions sites_per_block;
    const Dimensions blocks_per_file;
    const EigenvectorInfo eigenvector_info;

    // Instance-specific variables - auxillary variables
    const Count_t total_files;
    const Count_t files_per_directory;
    const Count_t stored_sites_per_block;
    const BasisSizes fp32_sizes;
    const BasisSizes fp16_sizes;
    const CoarseSizes coarse_sizes;
    const SectionSizes section_sizes;
    // const CoarseSizes coarse_sizes;

    const std::vector<Dimensions> sites_in_even_origin_block;    // Would prefer to do a lazy iteration than a big allocation
    const std::vector<Dimensions> sites_in_odd_origin_block;     // Would prefer to do a lazy iteration than a big allocation
    const std::vector<SiteBlock> blocks;

    // ######### //
    // FUNCTIONS //
    // ######### //
    // Important for parsing a file
    const static bool isSiteOdd(const Dimensions& site) noexcept;
    // These calculate all necessary offsets, counts, and dimensions to pin down a specific basis vector
    // Basis:
    const Count_t               getBlockIdxOfFileSite          (const Dimensions& file_site)                             const noexcept; // Util for getDataLocationOfFileSite
    const Count_t               getBlockSiteIdxOfFileSite      (const Dimensions& file_site)                             const noexcept; // Util for getDataLocationOfFileSite
    const BasisDataLocationInfo getDataLocationOfFileSite      (const Dimensions& file_site, Count_t which_basis_vector) const noexcept;
    const Offset_t              getFP32BlockBasisVectorOffset  (Count_t block_idx, Count_t which_basis_vector)           const noexcept;
    const Offset_t              getFP32FileOffsetOfDataLocation(const BasisDataLocationInfo& data_loc)                   const noexcept;
    const Offset_t              getFP16BlockBasisVectorOffset  (Count_t block_idx, Count_t which_basis_vector)           const noexcept;
    const Offset_t              getFP16FileOffsetOfDataLocation(const BasisDataLocationInfo& data_loc)                   const noexcept;
    // Coarse:
    const Count_t                      getBlockIdxOfFileBlock                      (const Dimensions& file_block)                            const noexcept; // Util for getDataLocationOfFileBlock
    const CoarseVectorDataLocationInfo getDataLocationOfFileBlock                  (const Dimensions& file_site, Count_t which_basis_vector) const noexcept;
    const Offset_t                     getCoarseEigenvectorFileOffsetOfDataLocation(const CoarseVectorDataLocationInfo& data_loc)            const noexcept;
    const Offset_t                     getCoarseEigenvectorOffset                  (Count_t which_eigenvector, Count_t block_idx)            const noexcept;
    // File + directory information - Generic
    const Count_t getDirectoryFromFile(Count_t file_idx) const noexcept;
    const std::vector<FileLocationInfo> getDirectoryStructure() const noexcept;
    // File + directory information - Fine basis
    const Count_t getDirectoryOfLatticeSite(const Dimensions& site) const;
    const Count_t getFileOfLatticeSite     (const Dimensions& site) const;
    const Dimensions getFirstLatticeSiteOfFile(Count_t file_id) const noexcept;
    // File + directory information - Coarse eigenvectors
    const Count_t getDirectoryOfCoarseSite(const Dimensions& site) const noexcept;
    const Count_t getFileOfCoarseSite(const Dimensions& site) const noexcept;
    const Dimensions getFirstCoarseSiteOfFile(Count_t file_id) const noexcept;
};

Metadata readMetadataFile(const std::string& filepath, const Dimensions&);
