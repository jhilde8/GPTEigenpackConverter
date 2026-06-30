#pragma once

#include <fstream>
#include <string>

#include "../Metadata/Metadata.hpp"

class FileReader
{
protected:
    // Static variables
    static const Count_t n_fermion_elements = 24;

    // Typedefs
    typedef std::array<float, n_fermion_elements> Fermion_t;

    // Non-const members
    std::fstream bytestream;
    const size_t fp32_buffer_size;
    const size_t fp16_buffer_size;
    const size_t coef_buffer_size;
public:
    // Const variables
    const Metadata& metadata;
    const Count_t file_idx;
    const std::string filepath;
    const Offset_t base_offset;
    const Dimensions fine_origin;
    const Dimensions block_origin;

    // C/Dtors
    FileReader(Count_t file_idx, const std::string& filepath, const Metadata& metadata);
    FileReader(Count_t file_idx, const std::string& filepath, Offset_t base_offset, const Metadata& metadata);
    ~FileReader();
    FileReader(const FileReader& fr);

    // File Reading Functions
    void validateSiteIndex(const Dimensions& file_site, const BasisDataLocationInfo& loc);

    std::vector<Fermion_t> readNextBasisVectorsFP32(Count_t MAX_CONCURRENT_BASIS_VECTORS, std::vector<char>& buf);
    std::vector<Fermion_t> readNextBasisBlockFP32(std::vector<char>& buf);
    Fermion_t readNextLatticeFermionFP32();
    Fermion_t readLatticeFermionFP32At(Offset_t offset);
    Fermion_t readLatticeFermionFP32AtFileSite(const Dimensions& file_site, Count_t which_basis_vector);
    Fermion_t readLatticeFermionFP32AtLatticeSite(const Dimensions& lattice_site, Count_t which_basis_vector);
    void      seekToFP32BlockBasisVector(Count_t block_idx, Count_t which_basis_vector);

    std::vector<Fermion_t> readNextBasisVectorsFP16(Count_t MAX_CONCURRENT_BASIS_VECTORS, std::vector<char>& buf);
    std::vector<Fermion_t> readNextBasisBlockFP16(std::vector<char>& buf);
    Fermion_t readNextLatticeFermionFP16();
    Fermion_t readLatticeFermionFP16At(Offset_t offset);
    Fermion_t readLatticeFermionFP16AtFileSite(const Dimensions& file_site, Count_t which_basis_vector);
    Fermion_t readLatticeFermionFP16AtLatticeSite(const Dimensions& lattice_site, Count_t which_basis_vector);
    void      seekToFP16BlockBasisVector(Count_t block_idx, Count_t which_basis_vector);

    std::vector<float> readNextCoarseEigenvector();
    std::vector<float> readCoarseEigenvectorAt(Offset_t offset);
    std::vector<float> readCoarseEigenvectorAtFileBlock(const Dimensions& file_block, Count_t which_eigenvector);
    std::vector<float> readCoarseEigenvectorAtCoarseSite(const Dimensions& coarse_site, Count_t which_eigenvector);
    void               seekToCoarseEigenvector(Count_t which_eigenvector);

    // Utils
    Dimensions latticeSiteToFileSite(const Dimensions&);
    Dimensions fileSiteToLatticeSite(const Dimensions&);
    void boundsCheckLatticeSite(const Dimensions&);
    void boundsCheckFileSite(const Dimensions&);
    Dimensions coarseSiteToFileBlock(const Dimensions& coarse_site);
    Dimensions fileBlockToCoarseSite(const Dimensions& file_block);
    void boundsCheckCoarseSite(const Dimensions& coarse_site);
    void boundsCheckFileBlock(const Dimensions& file_block);
    auto tell() { return this->bytestream.tellg(); }
};
