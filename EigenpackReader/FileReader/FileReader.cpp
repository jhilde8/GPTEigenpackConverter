#include <array>
#include <cstring>
#include <sstream>

#include "FileReader.hpp"
#include "../../Utils/FP16.hpp"

// RAII Constructor / Destructor
// fstream is RAII anyway?
FileReader::FileReader(Count_t file_idx, const std::string& filepath, const Metadata& metadata)
    : metadata    { metadata }
    , file_idx    { file_idx }
    , filepath    { filepath }
    , fine_origin { metadata.getFirstLatticeSiteOfFile(file_idx) }
    , block_origin{ metadata.getFirstCoarseSiteOfFile(file_idx) }
    , fp32_buffer_size { metadata.fp32_sizes.basis_vector_size }
    , fp16_buffer_size { metadata.fp16_sizes.basis_vector_size }
    , coef_buffer_size { metadata.coarse_sizes.coarse_eigenvector_size }
{
    this->bytestream.open(filepath, std::ios_base::in | std::ios_base::binary);
}

FileReader::~FileReader()
{
    this->bytestream.close();
}

FileReader::FileReader(const FileReader& fr)
    : metadata     { fr.metadata }
    , file_idx     { fr.file_idx }
    , filepath     { fr.filepath }
    , fine_origin  { fr.fine_origin }
    , block_origin { fr.block_origin }
    , fp32_buffer_size { fr.fp32_buffer_size }
    , fp16_buffer_size { fr.fp16_buffer_size }
    , coef_buffer_size { fr.coef_buffer_size }
{
    this->bytestream.open(filepath, std::ios_base::in | std::ios_base::binary);
}

// ################### //
// FP32 READ FUNCTIONS //
// ################### //

void FileReader::validateSiteIndex(const Dimensions& file_site, const BasisDataLocationInfo& loc)
{
    auto& block = this->metadata.blocks[loc.block_idx];
    const std::vector<Dimensions>* site_lookup;
    if (block.is_even_origin)
        site_lookup = &this->metadata.sites_in_even_origin_block;
    else
        site_lookup = &this->metadata.sites_in_odd_origin_block;
    const auto& check_site = (*site_lookup)[loc.block_site_idx] + block.fine_origin;
    if (!(file_site == check_site))
    {
        std::stringstream ss;
        ss << "CRITICAL ERROR: SITE INTERNALLY MISCALCULATED. RECEIVED: ";
        ss << file_site << ", MATCHED WITH: ";
        ss << check_site << "." << std::endl;
        throw std::runtime_error(ss.str().c_str());
    }
}

// readNextBasisVectorsFP32
std::vector<FileReader::Fermion_t> FileReader::readNextBasisVectorsFP32(Count_t MAX_CONCURRENT_BASIS_VECTORS, std::vector<char>& buf)
{
    std::vector<Fermion_t> out;
    out.resize(MAX_CONCURRENT_BASIS_VECTORS*this->metadata.stored_sites_per_block);

    this->bytestream.read(&buf[0], buf.size());
    std::memcpy(&out[0], &buf[0], buf.size()); // memcpy: reinterpret_cast here can lead to UB

    return out;
}

// readNextLatticeFermionFP32
std::vector<FileReader::Fermion_t> FileReader::readNextBasisBlockFP32(std::vector<char>& buf)
{
    return this->readNextBasisVectorsFP32(this->metadata.eigenvector_info.n_basis_fp32, buf);
}

// readNextLatticeFermionFP32
FileReader::Fermion_t FileReader::readNextLatticeFermionFP32()
{
    FileReader::Fermion_t fermion;
    std::array<char, Metadata::fp32_fermion_size> buf;

    this->bytestream.read(&buf[0], sizeof(buf));
    std::memcpy(&fermion, &buf, sizeof(buf)); // memcpy: reinterpret_cast here can lead to UB

    return fermion;
}

// readLatticeFermionFP32At
FileReader::Fermion_t FileReader::readLatticeFermionFP32At(Offset_t offset)
{
    this->bytestream.seekg(offset);
    return this->readNextLatticeFermionFP32();
}

// readLatticeFermionFP32AtFileSite
FileReader::Fermion_t FileReader::readLatticeFermionFP32AtFileSite(const Dimensions& file_site, Count_t which_basis_vector)
{
    this->boundsCheckFileSite(file_site);
    auto loc = this->metadata.getDataLocationOfFileSite(file_site, which_basis_vector);
    this->validateSiteIndex(file_site, loc);
    auto offset = this->metadata.getFP32FileOffsetOfDataLocation(loc);
    return this->readLatticeFermionFP32At(offset);
}

// readLatticeFermionFP32AtLatticeSite
FileReader::Fermion_t FileReader::readLatticeFermionFP32AtLatticeSite(const Dimensions& lattice_site, Count_t which_basis_vector)
{
    auto file_site = this->latticeSiteToFileSite(lattice_site);
    return this->readLatticeFermionFP32AtFileSite(file_site, which_basis_vector);
}

void FileReader::seekToFP32BlockBasisVector(Count_t block_idx, Count_t which_basis_vector)
{
    auto offset = this->metadata.getFP32BlockBasisVectorOffset(block_idx, which_basis_vector);
    this->bytestream.seekg(offset);
}

// ################### //
// FP16 READ FUNCTIONS //
// ################### //

// readNextBasisVectorsFP16
std::vector<FileReader::Fermion_t> FileReader::readNextBasisVectorsFP16(Count_t MAX_CONCURRENT_BASIS_VECTORS, std::vector<char>& buf)
{
    std::vector<Fermion_t> out;
    Count_t n_fermions_to_read = MAX_CONCURRENT_BASIS_VECTORS*this->metadata.stored_sites_per_block;
    out.resize(n_fermions_to_read);

    this->bytestream.read(&buf[0], buf.size());
    static const auto n_elems = FileReader::n_fermion_elements + 1;
    std::array<uint16_t, n_elems> temp_buf;
    for (int idx=0; idx < n_fermions_to_read; ++idx)
    {
        // Convert the next 50 bytes of the buffer to uint16s and decompress to FP32
        std::memcpy(&temp_buf[0], &buf[sizeof(uint16_t)*idx*n_elems], sizeof(temp_buf)); // memcpy: reinterpret_cast here can lead to UB
        out[idx] = unpackFP16s<FileReader::n_fermion_elements>(&temp_buf[0]);
    }

    return out;
}

// readNextBasisBlockFP16
std::vector<FileReader::Fermion_t> FileReader::readNextBasisBlockFP16(std::vector<char>& buf)
{
    return this->readNextBasisVectorsFP16(this->metadata.eigenvector_info.n_basis_fp16, buf);
}

// readNextLatticeFermionFP16
FileReader::Fermion_t FileReader::readNextLatticeFermionFP16()
{
    std::array<char, Metadata::fp16_fermion_size> buf;

    this->bytestream.read(&buf[0], sizeof(buf));

    std::array<uint16_t, FileReader::n_fermion_elements+1> values;
    std::memcpy(&values[0], &buf[0], sizeof(values)); // memcpy: reinterpret_cast here can lead to UB
    
    return unpackFP16s<FileReader::n_fermion_elements>(&values[0]);
}

// readLatticeFermionFP16At
FileReader::Fermion_t FileReader::readLatticeFermionFP16At(Offset_t offset)
{
    this->bytestream.seekg(offset);
    return this->readNextLatticeFermionFP16();
}

// readLatticeFermionFP32AtFileSite
FileReader::Fermion_t FileReader::readLatticeFermionFP16AtFileSite(const Dimensions& file_site, Count_t which_basis_vector)
{
    this->boundsCheckFileSite(file_site);
    auto loc = this->metadata.getDataLocationOfFileSite(file_site, which_basis_vector);
    this->validateSiteIndex(file_site, loc);
    auto offset = this->metadata.getFP16FileOffsetOfDataLocation(loc);
    return this->readLatticeFermionFP16At(offset);
}

// readLatticeFermionFP32AtLatticeSite
FileReader::Fermion_t FileReader::readLatticeFermionFP16AtLatticeSite(const Dimensions& lattice_site, Count_t which_basis_vector)
{
    auto file_site = this->latticeSiteToFileSite(lattice_site);
    return this->readLatticeFermionFP16AtFileSite(file_site, which_basis_vector);
}

void FileReader::seekToFP16BlockBasisVector(Count_t block_idx, Count_t which_basis_vector)
{
    auto offset = this->metadata.getFP16BlockBasisVectorOffset(block_idx, which_basis_vector);
    this->bytestream.seekg(offset);
}

// ################### //
// COEF READ FUNCTIONS //
// ################### //

std::vector<float> FileReader::readNextCoarseEigenvector()
{
    const auto& n_fp32 = 2*this->metadata.eigenvector_info.n_basis_fp32;
    const auto& n_fp16 = 2*this->metadata.eigenvector_info.n_basis_fp16;
    std::vector<float> out;
    out.resize(n_fp32 + n_fp16);

    // Read FP32
    std::vector<char> fp32_buf;
    const auto& fp32_buffer_size = this->metadata.coarse_sizes.fp32_coefficients_size;
    fp32_buf.resize(fp32_buffer_size);
    this->bytestream.read(&fp32_buf[0], fp32_buffer_size);
    std::memcpy(&out[0], &fp32_buf[0], fp32_buffer_size); // memcpy: reinterpret_cast here can lead to UB

    // Read FP16
    std::vector<char> fp16_buf;
    const auto& fp16_buffer_size = this->metadata.coarse_sizes.fp16_coefficients_size;
    fp16_buf.resize(fp16_buffer_size);
    this->bytestream.read(&fp16_buf[0], fp16_buffer_size);

    std::vector<uint16_t> values;
    values.resize(fp16_buffer_size/2);
    std::memcpy(&values[0], &fp16_buf[0], fp16_buffer_size); // memcpy: reinterpret_cast here can lead to UB

    const auto n_coefs = 10;
    for (int i=0; i < n_fp16 / n_coefs; ++i)
    {
        auto floats = unpackFP16s<n_coefs>(&values[i*(n_coefs+1)]);
        for (int j=0; j < n_coefs; ++j)
            out[n_fp32 + i*n_coefs + j] = floats[j];
    }

    return out;
}

std::vector<float> FileReader::readCoarseEigenvectorAt(Offset_t offset)
{
    this->bytestream.seekg(offset);
    return this->readNextCoarseEigenvector();
}

std::vector<float> FileReader::readCoarseEigenvectorAtFileBlock(const Dimensions& file_block, Count_t which_eigenvector)
{
    this->boundsCheckFileBlock(file_block);
    auto loc = this->metadata.getDataLocationOfFileBlock(file_block, which_eigenvector);
    const auto& check_block = this->metadata.blocks[loc.block_idx].coarse_site;
    if (!(file_block == check_block))
    {
        std::stringstream ss;
        ss << "CRITICAL ERROR: BLOCK INTERNALLY MISCALCULATED. RECEIVED: ";
        ss << file_block << ", MATCHED WITH: ";
        ss << check_block << "." << std::endl;
        throw std::runtime_error(ss.str().c_str());
    }
    auto offset = this->metadata.getCoarseEigenvectorFileOffsetOfDataLocation(loc);
    return this->readCoarseEigenvectorAt(offset);
}

std::vector<float> FileReader::readCoarseEigenvectorAtCoarseSite(const Dimensions& coarse_site, Count_t which_eigenvector)
{
    auto file_block = this->coarseSiteToFileBlock(coarse_site);
    return this->readCoarseEigenvectorAtFileBlock(file_block, which_eigenvector);
}

void FileReader::seekToCoarseEigenvector(Count_t which_eigenvector)
{
    auto offset = this->metadata.getCoarseEigenvectorOffset(which_eigenvector, 0);
    this->bytestream.seekg(offset);
}

// ##### //
// Utils //
// ##### //

// latticeSiteToFileSite
// Converts a lattice site to be relative to the origin of this file.
Dimensions FileReader::latticeSiteToFileSite(const Dimensions& lattice_site)
{
    return lattice_site - this->fine_origin;
}

// fileSiteToLatticeSite
// Converts a site in this file to be relative to the origin of the lattice.
Dimensions FileReader::fileSiteToLatticeSite(const Dimensions& file_site)
{
    return file_site + this->fine_origin;
}

// boundCheck
void FileReader::boundsCheckLatticeSite(const Dimensions& lattice_site)
{
    this->boundsCheckFileSite(this->latticeSiteToFileSite(lattice_site));
}

void FileReader::boundsCheckFileSite(const Dimensions& file_site)
{
    // Just >= should work because negative numbers will overflow due to the unsigned Count_t
    // But include < also just incase we change the data type to signed later
    auto is_outside_bounds = (file_site >= this->metadata.sites_per_file) || (file_site < Dimensions{0, 0, 0, 0, 0});
    if (is_outside_bounds.Ls || is_outside_bounds.x || is_outside_bounds.y || is_outside_bounds.z || is_outside_bounds.t)
        throw std::runtime_error("File site is out-of-bounds.");
}

// coarseSiteToFileBlock
Dimensions FileReader::coarseSiteToFileBlock(const Dimensions& coarse_site)
{
    return coarse_site - this->block_origin;
}

// fileBlockToCoarseSite
Dimensions FileReader::fileBlockToCoarseSite(const Dimensions& file_block)
{
    return file_block + this->block_origin;
}

// boundCheck
void FileReader::boundsCheckCoarseSite(const Dimensions& coarse_site)
{
    this->boundsCheckFileBlock(this->coarseSiteToFileBlock(coarse_site));
}

void FileReader::boundsCheckFileBlock(const Dimensions& file_block)
{
    // Just >= should work because negative numbers will overflow due to the unsigned Count_t
    // But include < also just incase we change the data type to signed later
    auto is_outside_bounds = (file_block >= this->metadata.blocks_per_file) || (file_block < Dimensions{0, 0, 0, 0, 0});
    if (is_outside_bounds.Ls || is_outside_bounds.x || is_outside_bounds.y || is_outside_bounds.z || is_outside_bounds.t)
        throw std::runtime_error("File block is out-of-bounds.");
}
