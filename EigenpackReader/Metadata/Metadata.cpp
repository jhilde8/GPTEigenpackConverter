#include "Metadata.hpp"
#include "MetadataReader.hpp"

// ############ //
// CONSTRUCTORS //
// ############ //

Metadata::Metadata(const Dimensions& grid_size, const MetadataLookup& readin)
    : grid_size{ grid_size }
    , sites_per_file { .Ls=readin.at( "s[4]"), .x=readin.at( "s[0]"), .y=readin.at( "s[1]"), .z=readin.at( "s[2]"), .t=readin.at( "s[3]") }
    , sites_per_block{ .Ls=readin.at( "b[4]"), .x=readin.at( "b[0]"), .y=readin.at( "b[1]"), .z=readin.at( "b[2]"), .t=readin.at( "b[3]") }
    , blocks_per_file{ .Ls=readin.at("nb[4]"), .x=readin.at("nb[0]"), .y=readin.at("nb[1]"), .z=readin.at("nb[2]"), .t=readin.at("nb[3]") }
    , eigenvector_info
    {
        .n_eigen=readin.at("neig"),
        .n_basis_total=readin.at("nkeep"),
        .n_basis_fp16=readin.at("nkeep") - readin.at("nkeep_single"),
        .n_basis_fp32=readin.at("nkeep_single"),
        .total_blocks_per_file=readin.at("blocks"),
        .n_coeffs_per_exponent = readin.at("FP16_COEF_EXP_SHARE_FLOATS")
    }
    // Derive various offsets and sizes: try not to re-calculate any quantities in order to strongly couple all 
    // variables together. Don't want to update something in more than one place.
    // I think putting it all in an initialiser list is a bit neater than having lots of functions to do this,
    // and the initialisation order is explicit assuming the list follows the variable order from the header.
    , total_files{ static_cast<Count_t>((grid_size / this->sites_per_file).volume()) }
    , files_per_directory{ total_files / Metadata::num_directories }
    , stored_sites_per_block { static_cast<Count_t>(this->sites_per_block.volume() / 2) } // Red-Black preconditioning: only odd sites stored
    , fp32_sizes
    {
        .basis_vector_size = this->stored_sites_per_block        * Metadata::fp32_fermion_size,
        .block_size =        this->eigenvector_info.n_basis_fp32 * this->fp32_sizes.basis_vector_size,
    }
    , fp16_sizes
    {
        .basis_vector_size = this->stored_sites_per_block        * Metadata::fp16_fermion_size,
        .block_size =        this->eigenvector_info.n_basis_fp16 * this->fp16_sizes.basis_vector_size,
    }
    , coarse_sizes
    {
        .fp32_coefficients_size = this->eigenvector_info.n_basis_fp32*Metadata::fp32_coefficient_size,
        .fp16_coefficients_size = this->getFP16CoeffsSize(),
        .coarse_eigenvector_size = this->coarse_sizes.fp32_coefficients_size + this->coarse_sizes.fp16_coefficients_size,
        .blocks_for_eigenvector_size = this->coarse_sizes.coarse_eigenvector_size * this->eigenvector_info.total_blocks_per_file
    }
    , section_sizes
    {
        .fp32_section_size   = this->eigenvector_info.total_blocks_per_file * this->fp32_sizes.block_size,
        .fp16_section_size   = this->eigenvector_info.total_blocks_per_file * this->fp16_sizes.block_size,
        .coarse_section_size = this->eigenvector_info.n_eigen * this->coarse_sizes.blocks_for_eigenvector_size,
        .total_size = this->section_sizes.fp32_section_size + this->section_sizes.fp16_section_size + this->section_sizes.coarse_section_size
    }
    , sites_in_even_origin_block { this->generateSitesInBlock<false>() }
    , sites_in_odd_origin_block{ this->generateSitesInBlock<true>() }
    , blocks{ this->generateBlocks() }
{};

// ################### //
// CONSTRUCTOR HELPERS //
// ################### //

// Total size counts
const size_t Metadata::getFP16CoeffChunkSize() const noexcept
{
    return Metadata::fp16_coefficient_size*(this->eigenvector_info.n_coeffs_per_exponent + 1);
}

const size_t Metadata::getFP16CoeffsSize() const noexcept
{
    // n_basis_16 must always be a multiple of n_coeffs_per_exponent for a file to be valid
    // Maybe this function should throw if that's not the case?
    auto num_chunks = this->eigenvector_info.n_basis_fp16/this->eigenvector_info.n_coeffs_per_exponent;
    return num_chunks*this->getFP16CoeffChunkSize();
}

// generateBlocks
std::vector<SiteBlock> Metadata::generateBlocks()
{
    // Useless references, but reminds us that these are members of this object
    auto& sites_per_block = this->sites_per_block;

    // Pre-calculate offsets that we'll use to calculate offsets for blocks later
    Offset_t base_fp32_offset = 0;
    Offset_t base_fp16_offset = this->section_sizes.fp32_section_size;
    Offset_t base_coef_offset = this->section_sizes.fp32_section_size + this->section_sizes.fp16_section_size;

    Offset_t block_size_fp32 = this->fp32_sizes.block_size;
    Offset_t block_size_fp16 = this->fp16_sizes.block_size;
    Offset_t block_size_coef = this->coarse_sizes.coarse_eigenvector_size;

    // Set up return value generation
    auto out = std::vector<SiteBlock>();
    out.reserve(blocks_per_file.volume());
    Count_t block_idx = 0;
    for (Count_t Ls_block=0; Ls_block < blocks_per_file.Ls; ++Ls_block)
    for (Count_t  t_block=0;  t_block <  blocks_per_file.t;  ++t_block)
    for (Count_t  z_block=0;  z_block <  blocks_per_file.z;  ++z_block)
    for (Count_t  y_block=0;  y_block <  blocks_per_file.y;  ++y_block)
    for (Count_t  x_block=0;  x_block <  blocks_per_file.x;  ++x_block, ++block_idx)
    {
        // Calculate values for the block
        auto block_site = Dimensions{.Ls=Ls_block, .x=x_block, .y=y_block, .z=z_block, .t=t_block};
        auto fine_origin = block_site*sites_per_block;
        auto is_even_origin = !this->isSiteOdd(fine_origin);
        const std::vector<Dimensions>* site_blocks_ptr;
        if (is_even_origin)
            site_blocks_ptr = &this->sites_in_even_origin_block;
        else
            site_blocks_ptr = &this->sites_in_odd_origin_block;
        out.emplace_back
        (
            SiteBlock
            {
                .fine_origin = fine_origin,
                .coarse_site = block_site,
                .is_even_origin = is_even_origin,
                .sites_in_block = *site_blocks_ptr, // This should init a reference
                .offset_basis_fp32   = base_fp32_offset + block_idx*block_size_fp32,
                .offset_basis_fp16   = base_fp16_offset + block_idx*block_size_fp16,
                .offset_coefficients = base_coef_offset + block_idx*block_size_coef
            }
        );
    }
    return out;
}

// generateSitesInBlock
template<bool make_even>
std::vector<Dimensions> Metadata::generateSitesInBlock()
{
    // Useless references, but reminds us that these are members of this object
    auto& sites_per_block = this->sites_per_block;

    // Would prefer to do a lazy iterator but it's probably a lot of effort for a comparatively small memory saving
    std::vector<Dimensions> out;
    out.reserve(this->stored_sites_per_block);
    for (Count_t Ls=0; Ls < sites_per_block.Ls; ++Ls)
    for (Count_t  t=0;  t <  sites_per_block.t;  ++t)
    for (Count_t  z=0;  z <  sites_per_block.z;  ++z)
    for (Count_t  y=0;  y <  sites_per_block.y;  ++y)
    for (Count_t  x=0;  x <  sites_per_block.x;  ++x)
    {
        auto site = Dimensions{.Ls=Ls, .x=x, .y=y, .z=z, .t=t};
        if (make_even ^ this->isSiteOdd(site)) // XOR allows this function to be used for both even and odd site generation
            out.emplace_back(site);
    }

    return out;
}


// ##### //
// UTILS //
// ##### //

// File parsing methods

const bool Metadata::isSiteOdd(const Dimensions& site) noexcept
{
    return (site.x + site.y + site.z + site.t) % 2;
}


// ############## //
// OFFSET PARSING //
// ############## //

const Count_t Metadata::getBlockIdxOfFileSite(const Dimensions& file_site) const noexcept
{
    auto& bounding_volume = this->sites_per_block;
    auto position_within_block = file_site / bounding_volume;
    return flattenDimensionsLsSlow(position_within_block, this->blocks_per_file);
}

const Count_t Metadata::getBlockSiteIdxOfFileSite(const Dimensions& file_site) const noexcept
{
    auto& bounding_volume = this->sites_per_block;
    auto index_in_each_block_direction = file_site % bounding_volume;

    return flattenDimensionsLsSlow(index_in_each_block_direction, this->sites_per_block) / 2; //  1/2 for red-black: Odd sites only
}

const BasisDataLocationInfo Metadata::getDataLocationOfFileSite(const Dimensions& file_site, Count_t which_basis_vector) const noexcept
{
    return BasisDataLocationInfo
    {
        .block_idx=this->getBlockIdxOfFileSite(file_site),
        .basis_vector_idx=which_basis_vector,
        .block_site_idx=this->getBlockSiteIdxOfFileSite(file_site)
    };
}

const Offset_t Metadata::getFP32BlockBasisVectorOffset(Count_t block_idx, Count_t which_basis_vector) const noexcept
{
    auto& block = this->blocks[block_idx];
    Offset_t basis_vector_offset = which_basis_vector * this->fp32_sizes.basis_vector_size;
    return block.offset_basis_fp32 + basis_vector_offset;
}

const Offset_t Metadata::getFP32FileOffsetOfDataLocation(const BasisDataLocationInfo& data_loc) const noexcept
{
    Offset_t block_basis_vector_offset = this->getFP32BlockBasisVectorOffset(data_loc.block_idx, data_loc.basis_vector_idx);
    Offset_t block_site_offset = data_loc.block_site_idx * Metadata::fp32_fermion_size;
    return block_basis_vector_offset + block_site_offset;
}

const Offset_t Metadata::getFP16BlockBasisVectorOffset(Count_t block_idx, Count_t which_basis_vector) const noexcept
{
    auto& block = this->blocks[block_idx];
    Offset_t basis_vector_offset = (which_basis_vector - this->eigenvector_info.n_basis_fp32) * this->fp16_sizes.basis_vector_size;
    return block.offset_basis_fp16 + basis_vector_offset;
}

const Offset_t Metadata::getFP16FileOffsetOfDataLocation(const BasisDataLocationInfo& data_loc) const noexcept
{
    Offset_t block_basis_vector_offset = this->getFP16BlockBasisVectorOffset(data_loc.block_idx, data_loc.basis_vector_idx);
    Offset_t block_site_offset = data_loc.block_site_idx * Metadata::fp16_fermion_size;
    return block_basis_vector_offset + block_site_offset;
}

const Count_t Metadata::getBlockIdxOfFileBlock(const Dimensions& file_block) const noexcept
{
    auto idx = flattenDimensionsLsSlow(file_block, this->blocks_per_file);
    return idx;
}

const CoarseVectorDataLocationInfo Metadata::getDataLocationOfFileBlock(const Dimensions& file_site, Count_t which_eigenvector) const noexcept
{
    return CoarseVectorDataLocationInfo
    {
        .block_idx=this->getBlockIdxOfFileBlock(file_site),
        .eigenvector_idx=which_eigenvector
    };
}

const Offset_t Metadata::getCoarseEigenvectorFileOffsetOfDataLocation(const CoarseVectorDataLocationInfo& data_loc) const noexcept
{
    return this->getCoarseEigenvectorOffset(data_loc.eigenvector_idx, data_loc.block_idx);
}

const Offset_t Metadata::getCoarseEigenvectorOffset(Count_t which_eigenvector, Count_t block_idx) const noexcept
{
    const auto& block = this->blocks[block_idx];
    Offset_t eigenvector_offset = this->coarse_sizes.blocks_for_eigenvector_size * which_eigenvector;
    return block.offset_coefficients + eigenvector_offset;
}

// ################# //
// Directory Getters //
// ################# //

// getDirectoryFromFile
const Count_t Metadata::getDirectoryFromFile(Count_t file_idx) const noexcept
{
    return file_idx / this->files_per_directory;
}

// getDirectoryStructure
const std::vector<FileLocationInfo> Metadata::getDirectoryStructure() const noexcept
{
    std::vector<FileLocationInfo> out;
    out.reserve(this->total_files);

    for (Count_t file_idx=0; file_idx < this->total_files; ++file_idx)
    {
        out.emplace_back(FileLocationInfo{.directory=this->getDirectoryFromFile(file_idx), .file=file_idx});
    }

    return out;
}

// ###################### //
// File Info - Fine Basis //
// ###################### //

// getDirectoryOfLatticeSite
const Count_t Metadata::getDirectoryOfLatticeSite(const Dimensions& lattice_site) const
{
    return this->getDirectoryFromFile(this->getFileOfLatticeSite(lattice_site));
}

// getFileOfLatticeSite
const Count_t Metadata::getFileOfLatticeSite(const Dimensions& lattice_site) const
{
    auto fpd = grid_size / this->sites_per_file;
    auto external_dims = lattice_site / this->sites_per_file;
    return flattenDimensionsLsFast(external_dims, fpd);
}

// getFirstLatticeSiteOfFile
const Dimensions Metadata::getFirstLatticeSiteOfFile(Count_t file_id) const noexcept
{
    auto fpd = this->grid_size / this->sites_per_file;
    auto t_idx = file_id / (fpd.Ls*fpd.x*fpd.y*fpd.z);
    file_id -= t_idx*(fpd.Ls*fpd.x*fpd.y*fpd.z);
    auto z_idx = file_id / (fpd.Ls*fpd.x*fpd.y);
    file_id -= z_idx*(fpd.Ls*fpd.x*fpd.y);
    auto y_idx = file_id / (fpd.Ls*fpd.x);
    file_id -= y_idx*(fpd.Ls*fpd.x);
    auto x_idx = file_id / (fpd.Ls);
    file_id -= x_idx*(fpd.Ls);
    auto Ls_idx = file_id;

    return Dimensions{Ls_idx, x_idx, y_idx, z_idx, t_idx} * this->sites_per_file;
}

// ######################## //
// File Info - Coarse Evecs //
// ######################## //

// getDirectoryOfLatticeSite
const Count_t Metadata::getDirectoryOfCoarseSite(const Dimensions& lattice_site) const noexcept
{
    return this->getDirectoryFromFile(this->getFileOfCoarseSite(lattice_site));
}

// getFileOfLatticeSite
const Count_t Metadata::getFileOfCoarseSite(const Dimensions& coarse_site) const noexcept
{
    auto fpd = (grid_size / this->sites_per_block) / this->blocks_per_file;
    auto external_dims = coarse_site / this->blocks_per_file;
    return flattenDimensionsLsFast(external_dims, fpd);
}

// getFirstLatticeSiteOfFile
const Dimensions Metadata::getFirstCoarseSiteOfFile(Count_t file_id) const noexcept
{
    auto fpd = (grid_size / this->sites_per_block) / this->blocks_per_file;
    auto t_idx = file_id / (fpd.Ls*fpd.x*fpd.y*fpd.z);
    file_id -= t_idx*(fpd.Ls*fpd.x*fpd.y*fpd.z);
    auto z_idx = file_id / (fpd.Ls*fpd.x*fpd.y);
    file_id -= z_idx*(fpd.Ls*fpd.x*fpd.y);
    auto y_idx = file_id / (fpd.Ls*fpd.x);
    file_id -= y_idx*(fpd.Ls*fpd.x);
    auto x_idx = file_id / (fpd.Ls);
    file_id -= x_idx*(fpd.Ls);
    auto Ls_idx = file_id;

    return Dimensions{Ls_idx, x_idx, y_idx, z_idx, t_idx} * this->blocks_per_file;
}

//####################//
// EXPORTED FUNCTIONS //
//####################//
Metadata readMetadataFile(const std::string& filepath, const Dimensions& grid_size)
{
    MetadataReader meta_reader(filepath);
    return meta_reader.read(grid_size);
}
