#include <cassert>
#include <cstdio>
#include <iostream>

#include <Grid/Grid.h>
#include <Hadrons/Application.hpp>
#include <Hadrons/EigenPack.hpp>
#include <openssl/sha.h>

#include "EigenpackReader/EigenpackReader.hpp"
#include "EigenpackReader/Eigenvalues/Eigenvalues.hpp"
#include "EigenpackReader/GridEigenpackReader.hpp"
#include "EigenpackReader/FileReader/FileReader.hpp"

#include "Global.h"

// #ifndef NUM_BASIS_VECTORS
//     #define NUM_BASIS_VECTORS 30
// #endif

// #ifndef MAX_CONCURRENT_BASIS_VECTORS
//     #define MAX_CONCURRENT_BASIS_VECTORS 2
// #endif

// #ifndef MAX_CONCURRENT_EIGEN_VECTORS
//     #define MAX_CONCURRENT_EIGEN_VECTORS 5
// #endif

void printoutFermion(float* ferm)
{
    std::cout 
    << ferm[ 0] << " " << ferm[ 1] << " "
    << ferm[ 2] << " " << ferm[ 3] << " "
    << ferm[ 4] << " " << ferm[ 5] << " "
    << ferm[ 6] << " " << ferm[ 7] << " "
    << ferm[ 8] << " " << ferm[ 9] << " "
    << ferm[10] << " " << ferm[11] << " "
    << ferm[12] << " " << ferm[13] << " "
    << ferm[14] << " " << ferm[15] << " "
    << ferm[16] << " " << ferm[17] << " "
    << ferm[18] << " " << ferm[19] << " "
    << ferm[20] << " " << ferm[21] << " "
    << ferm[22] << " " << ferm[23] << " "
    << std::endl;
}

template<bool isFP32>
void printoutSite(EigenpackReader& er, const Dimensions& site, Count_t which_basis_vector)
{
    auto loc = er.metadata.getFileOfLatticeSite(site);
    std::array<float, 24> ferm;
    if (isFP32)
        ferm = er.filereaders[loc].readLatticeFermionFP32AtLatticeSite(site, which_basis_vector);
    else
        ferm = er.filereaders[loc].readLatticeFermionFP16AtLatticeSite(site, which_basis_vector);
    printoutFermion(&ferm[0]);
}



void printoutCoefs(EigenpackReader& er, const Dimensions& site, Count_t which_eigenvector)
{
    auto loc = er.metadata.getFileOfCoarseSite(site);
    auto res = er.filereaders[loc].readCoarseEigenvectorAtCoarseSite(site, which_eigenvector);
    
    for (const auto& elem : res)
        std::cout << elem << " ";
    std::cout << std::endl;
}

struct Inputs
{
    int Ls=0;
    std::string mpi_layout = "1.1.1.1";
    std::string grid_layout = "";
    std::string eigenpack_location = "";
    std::string out_location = "";
    int config=-1;
    WriteMode writemode = WriteMode::INVALID;
};

void printUsage()
{
    std::cout << Grid::GridLogMessage << "Usage: " << std::endl;
    std::cout << Grid::GridLogMessage << "[Required] -i: Location of compressed eigenpack." << std::endl;
    std::cout << Grid::GridLogMessage << "[Required] -o: Output folder." << std::endl;
    std::cout << Grid::GridLogMessage << "[Required] --Ls: Ls size." << std::endl;
    std::cout << Grid::GridLogMessage << "[Required] --grid: Grid size." << std::endl;
    std::cout << Grid::GridLogMessage << "[Required] --outmode: [none, basic, hadrons]." << std::endl;
    std::cout << Grid::GridLogMessage << "[Required] --config: Configuration number." << std::endl;
    std::cout << Grid::GridLogMessage << "[Optional] --mpi: Grid MPI Layout." << std::endl;
}



Inputs parseArgv(int argc, char* argv[])
{
    using namespace Grid;
    Inputs inputs;

    if (argc == 1)
    {
        printUsage();
        throw std::runtime_error("Input error.");
    }

    if (!(argc % 2))
    {
        std::cout << Grid::GridLogMessage << "Invalid number of inputs." << std::endl;
        printUsage();
        throw std::runtime_error("Input error.");
    }

    for (int i=1; i < argc; i+=2)
    {
        std::cout << "Parsing" << std::endl;
        std::string this_arg = std::string(argv[i]);
        std::string next_arg = std::string(argv[i+1]);
        if (this_arg == "--mpi")
        {
            std::cout << GridLogMessage << "Setting MPI to " << next_arg << std::endl;
            inputs.mpi_layout = next_arg;
        }
        else if (this_arg == "--grid")
        {
            std::cout << GridLogMessage << "Setting Grid to " << next_arg << std::endl;
            inputs.grid_layout = next_arg;
        }
        else if (this_arg == "--Ls")
        {
            std::cout << GridLogMessage << "Setting Ls to " << next_arg << std::endl;
            inputs.Ls = std::stoi(next_arg);
        }
        else if (this_arg == "-i")
        {
            std::cout << GridLogMessage << "Setting input path to " << next_arg << std::endl;
            inputs.eigenpack_location = next_arg;
        }
        else if (this_arg == "-o")
        {
            std::cout << GridLogMessage << "Setting output path to " << next_arg << std::endl;
            inputs.out_location = next_arg;
        }
        else if (this_arg == "--outmode")
        {
            std::cout << GridLogMessage << "Setting outmode to " << next_arg << std::endl;
            if (next_arg == "none")
                inputs.writemode = WriteMode::NONE;
            else if (next_arg == "basic")
                inputs.writemode = WriteMode::BASIC;
            else if (next_arg == "hadrons")
                inputs.writemode = WriteMode::HADRONS;
            else
            {
                std::cout << Grid::GridLogMessage << "Unrecognised input: " << next_arg << std::endl;
                printUsage();
                throw std::runtime_error("Input error.");
            }
        }
        else if (this_arg == "--config")
        {
            std::cout << GridLogMessage << "Setting config path to " << next_arg << std::endl;
            inputs.config = std::stoi(next_arg);
        }
        else
        {
            std::cout << Grid::GridLogMessage << "Unrecognised input: " << next_arg << std::endl;
            printUsage();
            throw std::runtime_error("Input error.");
        }
    }


    if (inputs.grid_layout.size() == 0)
    {
        std::cout << Grid::GridLogMessage << "--grid argument missing." << std::endl;
        printUsage();
        throw std::runtime_error("Input error.");
    }
    if (inputs.eigenpack_location.size() == 0)
    {
        std::cout << Grid::GridLogMessage << "-i argument missing." << std::endl;
        printUsage();
        throw std::runtime_error("Input error.");
    }
    if (inputs.out_location.size() == 0)
    {
        std::cout << Grid::GridLogMessage << "-o argument missing." << std::endl;
        printUsage();
        throw std::runtime_error("Input error.");
    }
    if (inputs.Ls == 0 )
    {
        std::cout << Grid::GridLogMessage << "--Ls argument missing." << std::endl;
        printUsage();
        throw std::runtime_error("Input error.");
    }
    if (inputs.writemode == WriteMode::INVALID)
    {
        std::cout << Grid::GridLogMessage << "--outmode argument missing." << std::endl;
        printUsage();
        throw std::runtime_error("Input error.");
    }

    return inputs;
}

Dimensions getGridSize(const Inputs& inputs)
{
    auto grid_str = inputs.grid_layout;

    std::array<Count_t, 4> pos;
    for (int i=0; i < pos.size(); ++i)
    {
        int strpos = grid_str.find(".");
        pos[i] = std::stoi(grid_str.substr(0, strpos));
        grid_str = grid_str.substr(strpos+1, grid_str.size());
    }

    Dimensions out{static_cast<Count_t>(inputs.Ls), pos[0], pos[1], pos[2], pos[3]};
    return out;
}


Dimensions getMPISize(const Inputs& inputs)
{
    auto mpi_str = inputs.mpi_layout;

    std::array<Count_t, 4> pos;
    for (int i=0; i < pos.size(); ++i)
    {
        int strpos = mpi_str.find(".");
        pos[i] = std::stoi(mpi_str.substr(0, strpos));
        mpi_str = mpi_str.substr(strpos+1, mpi_str.size());
    }

    Dimensions out{1, pos[0], pos[1], pos[2], pos[3]};
    return out;
}


// << ISSUES >>
// -> Is a vector of arrays guaranteed to create a contiguous memory block? It does under Clang 11.0.0, but is it guaranteed by the standard?
int main(int argc, char* argv[])
{
    using namespace Grid;

    // Fails to compile without these three lines because of a missing symbol?!!
    std::array<unsigned char, 256> d;
    std::array<unsigned char, 256> m;
    SHA256(&d[0], 256, &m[0]);

    // Init the input variables
    Inputs inputs;
    inputs = parseArgv(argc, argv);
    auto grid_size = getGridSize(inputs);

    static const auto DO_WRITE = inputs.writemode;

    // Init eigenpack reader
    GridEigenpackReader<NUM_BASIS_VECTORS, MAX_CONCURRENT_BASIS_VECTORS, MAX_CONCURRENT_EIGEN_VECTORS> gepackrdr(inputs.eigenpack_location, grid_size, DO_WRITE);

    // Consistency checks on inputs
    auto& evi = gepackrdr.metadata.eigenvector_info;
    if (evi.n_basis_fp32 % MAX_CONCURRENT_BASIS_VECTORS)
        throw std::runtime_error
        (
            "Number of FP32 basis vectors ("             +
            std::to_string(evi.n_basis_fp32)             +
            ") not divisible by batch read size ("       +
            std::to_string(MAX_CONCURRENT_BASIS_VECTORS) +
            ")."
        );
    if (evi.n_basis_fp16 % MAX_CONCURRENT_BASIS_VECTORS)
        throw std::runtime_error
        (
            "Number of FP16 basis vectors ("             +
            std::to_string(evi.n_basis_fp16)             +
            ") not divisible by batch read size ("       +
            std::to_string(MAX_CONCURRENT_BASIS_VECTORS) +
            ")."
        );
    if (evi.n_eigen % MAX_CONCURRENT_EIGEN_VECTORS)
        throw std::runtime_error
        (
            "Number of coarse eigenvectors ("            +
            std::to_string(evi.n_eigen)                  +
            ") not divisible by batch read size ("       +
            std::to_string(MAX_CONCURRENT_EIGEN_VECTORS) +
            ")."
        );

    if (NUM_BASIS_VECTORS != gepackrdr.metadata.eigenvector_info.n_basis_total)
        throw std::runtime_error
        (
            "Number of input basis vectors ("                                 +
            std::to_string(NUM_BASIS_VECTORS)                                 +
            ") not equal to that defined in the metadata ("                   +
            std::to_string(gepackrdr.metadata.eigenvector_info.n_basis_total) +
            ")."
        );
    
    {
        // Check that no more than 2^31 bytes will be writen per proc to avoid
        // integer overflows in MPI
        uint64_t   evec_size       = grid_size.volume() * Metadata::fp32_fermion_size / 2;
        float      evec_size_gb    = evec_size / (1024*1024*1024); 
        uint32_t   min_mpi_procs   = static_cast<uint32_t>(std::ceil(evec_size / ((uint64_t)1 << 31)));
        Dimensions mpi_size        = getMPISize(inputs);
        uint32_t   input_mpi_procs = mpi_size.volume();
        if (min_mpi_procs > input_mpi_procs)
        {
            std::string msg = "Not enough MPI processes: initialised with '" + inputs.mpi_layout       +
                              "' (" + std::to_string(input_mpi_procs) + " total), but at least "       +
                              std::to_string(min_mpi_procs) + " are required to successfully write a " + 
                              std::to_string(evec_size_gb) + "GiB eigenvector.";
            std::cout <<GridLogMessage << msg << std::endl;
            throw std::runtime_error(msg);
        }
        else
        {
            std::cout << GridLogMessage << "Number of MPI procs OK" << std::endl;
        }
        
        
        // Check that the eigenpack files will not be accessed by multiple MPI processes.
        // The code does not currently support this because calculating offsets into other
        // MPI ranks is annoying.
        float local_grid_size_ls   = static_cast<float>(grid_size.Ls) / mpi_size.Ls;
        float local_grid_size_x    = static_cast<float>(grid_size.x) / mpi_size.x;
        float local_grid_size_y    = static_cast<float>(grid_size.y) / mpi_size.y;
        float local_grid_size_z    = static_cast<float>(grid_size.z) / mpi_size.z;
        float local_grid_size_t    = static_cast<float>(grid_size.t) / mpi_size.t;
        Dimensions local_grid_size = grid_size / mpi_size;
        Dimensions remainders      = local_grid_size % gepackrdr.metadata.sites_per_file;
        
        std::string bad_sizes = "";
        if (remainders.Ls != 0)
        {
            bad_sizes += "Local Ls dimension is not factorisable by sites per eigenpack file (" +
                         std::to_string(local_grid_size_ls) + " vs " + std::to_string(gepackrdr.metadata.sites_per_file.Ls) + ")\n";
            std::cout << GridLogMessage << "Local Ls dimension NOT OK " << std::to_string(local_grid_size_ls) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.Ls) << std::endl;
        }
        else
        {
            std::cout << GridLogMessage << "Local Ls dimension OK " << std::to_string(local_grid_size_ls) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.Ls) << std::endl;
        }
        if (remainders.x != 0)
        {
            bad_sizes += "Local x dimension is not factorisable by sites per eigenpack file (" +
                         std::to_string(local_grid_size_x) + " vs " + std::to_string(gepackrdr.metadata.sites_per_file.x) + ")\n";
            std::cout << GridLogMessage << "Local x dimension NOT OK " << std::to_string(local_grid_size_x) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.x) << std::endl;
        }
        else
        {
            std::cout << GridLogMessage << "Local x dimension OK " << std::to_string(local_grid_size_x) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.x) << std::endl;
        }
        if (remainders.y != 0)
        {
            bad_sizes += "Local y dimension is not factorisable by sites per eigenpack file (" +
                         std::to_string(local_grid_size_y) + " vs " + std::to_string(gepackrdr.metadata.sites_per_file.y) + ")\n";
            std::cout << GridLogMessage << "Local y dimension NOT OK " << std::to_string(local_grid_size_y) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.y) << std::endl;
        }
        else
        {
            std::cout << GridLogMessage << "Local y dimension OK " << std::to_string(local_grid_size_y) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.y) << std::endl;
        }
        if (remainders.z != 0)
        {
            bad_sizes += "Local Ls dimension is not factorisable by sites per eigenpack file (" +
                         std::to_string(local_grid_size_z) + " vs " + std::to_string(gepackrdr.metadata.sites_per_file.z) + ")\n";
            std::cout << GridLogMessage << "Local z dimension NOT OK " << std::to_string(local_grid_size_z) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.z) << std::endl;
        }
        else
        {
            std::cout << GridLogMessage << "Local z dimension OK " << std::to_string(local_grid_size_z) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.z) << std::endl;
        }
        if (remainders.t != 0)
        {
            bad_sizes += "Local t dimension is not factorisable by sites per eigenpack file (" +
                         std::to_string(local_grid_size_t) + " vs " + std::to_string(gepackrdr.metadata.sites_per_file.t) + ")\n";
            std::cout << GridLogMessage << "Local t dimension NOT OK " << std::to_string(local_grid_size_t) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.t) << std::endl;
        }
        else
        {
            std::cout << GridLogMessage << "Local t dimension OK " << std::to_string(local_grid_size_t) + "/" + std::to_string(gepackrdr.metadata.sites_per_file.t) << std::endl;
        }
        
        if (!bad_sizes.empty())
        {
            bad_sizes += "Re-run with an MPI decomposition that does not cause eigenpack files to cross MPI ranks.";
            std::cout << GridLogMessage << bad_sizes << std::endl;
            throw std::runtime_error(bad_sizes);
        }
    }
    
    //#####################//
    // GRID INITIALISATION //
    //#####################//

    // Set up Grid
    int grid_argc = 4;
    char* grid_argv[4];
    grid_argv[0] = const_cast<char*>("--mpi");
    grid_argv[1] = const_cast<char*>(inputs.mpi_layout.c_str());
    grid_argv[2] = const_cast<char*>("--grid");
    grid_argv[3] = const_cast<char*>(inputs.grid_layout.c_str());
    char** _grid_argv = grid_argv;
    Grid_init(&grid_argc, &_grid_argv);

    // Read and rewrite the Hadrons eigenpack twice to ensure that it isn't corrupted
    // when being written to disk
    Grid::BinaryIO::latticeWriteMaxRetry = 2;

    // Make the Fine Grid
    std::cout << GridLogMessage << "Initialising Fine Grid" << std::endl;
    auto& env = Hadrons::Environment::getInstance();
    env.createGrid<vComplexF>(grid_size.Ls);
    auto* fgrid = env.getRbGrid<vComplexF>(grid_size.Ls);

    const auto& spb = gepackrdr.metadata.sites_per_block;
    std::cout << GridLogMessage << "Initialising Coarse Grid with block size " << spb << std::endl;
    const std::vector<int> coarse_dims = {(int)spb.x, (int)spb.y, (int)spb.z, (int)spb.t};
    env.createCoarseGrid<vComplexF>(coarse_dims, grid_size.Ls);
    auto* cgrid = env.getCoarseGrid<vComplexF>(coarse_dims);

    auto& dims = fgrid->GlobalDimensions();
    auto& cdims = cgrid->GlobalDimensions();
    auto simd_layout = GridDefaultSimd(4,vComplex::Nsimd());
    auto mpi_layout = GridDefaultMpi();
    int64_t threads = GridThread::GetThreads();
    std::cout << GridLogMessage << "Grid dims:  " <<  dims[0] << " " <<  dims[1] << " " <<  dims[2] << " " <<  dims[3] << " " << dims[4] << std::endl;
    std::cout << GridLogMessage << "CGrid dims: " << cdims[0] << " " << cdims[1] << " " << cdims[2] << " " << cdims[3] << std::endl;
    std::cout << GridLogMessage << "Default MPI layout is: " << mpi_layout[0] << " " << mpi_layout[1] << " " << mpi_layout[2] << " " << mpi_layout[3] << std::endl;
    std::cout << GridLogMessage << "Reading from: " << inputs.eigenpack_location << std::endl;
    std::stringstream ss;
    ss << "Outputting to: ";
    switch (inputs.writemode)
    {
    case WriteMode::NONE:
        ss << "Nowhere, output disabled.";
        break;
    case WriteMode::BASIC:
        ss << inputs.out_location << "; BASIC output mode.";
        break;
    case WriteMode::HADRONS:
        ss << inputs.out_location << "; Hadrons CoarseEigenpack output mode.";
        break;
    case WriteMode::INVALID:
    default:
        ss << "Error!";
        throw std::runtime_error("");
        break;
    } 
    std::cout << GridLogMessage << ss.str() << std::endl;

    //##############//
    // FILE READING //
    //##############//
    std::cout << GridLogMessage << "About to begin converting..." << std::endl;
    try
    {
        gepackrdr.read(fgrid, cgrid, inputs.out_location, inputs.config);
    }
    catch(const std::exception& e)
    {
        std::cout << "Encountered exception: " << e.what() << std::endl;
        throw e;
    }
    
    std::cout << GridLogMessage << "Conversion complete!" << std::endl;

    Grid_finalize();
}
