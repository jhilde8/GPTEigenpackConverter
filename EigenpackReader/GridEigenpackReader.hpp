#pragma once

#include <Grid/Grid.h>
#include <Hadrons/Application.hpp>
#include <Hadrons/EigenPack.hpp>
#include "Eigenvalues/Eigenvalues.hpp"
#include "EigenpackReaderBase.hpp"
#include "Validation/BasicBinary.hpp"


template<typename T>
void PrintProcessesFilereaders(const std::vector<T>& arr, int proc_count)
{
    int rank;
    int sz = arr.size();
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank != 0)
    {
        const int* data = arr.data();
        MPI_Send(data, sz, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    {    
        // RANK 0
        std::cout << Grid::GridLogMessage << "FileReader Allocation [" << std::to_string(rank) << " ::";
        for (int i=0; i < sz; ++i)
            std::cout << " " << arr[i];
        std::cout << "]";

        // RANK >0
        for (int p=1; p < proc_count; ++p)
        {
            // Print object
            std::cout << " [" << std::to_string(p) << " ::";
            std::vector<int> recv(sz);
            MPI_Recv(recv.data(), sz, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i=0; i < sz; ++i)
                std::cout << " " << recv[i];
            std::cout << "]";
        }
        std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);   
}

template<typename T> // Template only here because I'm too lazy to write this in the cpp file
void PrintArrayFromAllProcesses(const std::string& msg, const std::vector<T>& arr, int proc_count)
{
    int rank;
    int sz = arr.size();
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank != 0)
    {
        MPI_Send(arr.data(), sz, MPI_INT, 0, 0, MPI_COMM_WORLD);
        int msg_length = (int)msg.length();
        MPI_Send(&msg_length, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(msg.c_str(), msg_length, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    {    
        // RANK 0
        std::cout << "[PROC " << std::to_string(rank) << "] :: " << msg;
        for (int i=0; i < sz; ++i)
            std::cout << " " << arr[i];
        std::cout << std::endl;

        // RANK >0
        for (int p=1; p < proc_count; ++p)
        {
            // Print Message
            int msg_length;
            MPI_Recv(&msg_length, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<char> chardata(msg_length);
            MPI_Recv(chardata.data(), msg_length, MPI_CHAR, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "[PROC " << std::to_string(p) << "] :: " << chardata.data();

            // Print object
            std::vector<int> recv(sz);
            MPI_Recv(recv.data(), sz, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i=0; i < sz; ++i)
                std::cout << " " << recv[i];
            std::cout << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);   
}

// template<typename T> // Template only here because I'm too lazy to write this in the cpp file
// void PrintOnMainProcess(const T& msg, int proc_idx)
// {
//     MPI_Barrier(MPI_COMM_WORLD);
//     int msg_length = (int)msg.length();
//     MPI_Send(&msg_length, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
//     const char* chardata = msg.c_str();
//     MPI_Send(chardata, msg_length, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
//     MPI_Barrier(MPI_COMM_WORLD);
//     if(rank == 0)
//     { 
//         // Print Message
//         int msg_length;
//         MPI_Recv(&msg_length, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         char chardata[msg_length];
//         MPI_Recv(chardata, msg_length, MPI_CHAR, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         std::cout << "[PROC " << std::to_string(proc_idx) << "] :: " << chardata;
//         std::cout << std::endl;
        
//     }
//     MPI_Barrier(MPI_COMM_WORLD);   
// }

enum class WriteMode
{
    INVALID,
    NONE,
    BASIC,
    HADRONS,
    SCIDAC
};

typedef void                  (FileReader::*SeekBasisFunc)(Count_t, Count_t);
typedef std::vector<Fermion_t>(FileReader::*ReadBasisFunc)(Count_t, std::vector<char>&);

template<Count_t NUM_BASIS_VECTORS, Count_t NUM_CONCURRENT_BASIS_VECTORS, Count_t NUM_CONCURRENT_EIGEN_VECTORS>
class GridEigenpackReader : public EigenpackReaderBase
{
private:
    typedef Grid::Hadrons::CoarseFermionEigenPack<Grid::FIMPLF, NUM_CONCURRENT_BASIS_VECTORS> Epack_t;

    const WriteMode DO_WRITE;

public:
    GridEigenpackReader(const std::string& eigenpack_location, const Dimensions& grid_size, WriteMode DO_WRITE);

    template<SeekBasisFunc SEEK_TO_VECTOR, ReadBasisFunc READ_DATA>
    void readBasisBase(Offset_t vector_size, Count_t initial_vector, Count_t num_vectors_in_section, Grid::GridRedBlackCartesian* fgrid, Epack_t& eigenpack, const std::string& out_path, int config);
    template<SeekBasisFunc SEEK_TO_VECTOR, ReadBasisFunc READ_DATA>
    void readBasisBaseBatch(Count_t batch_idx, std::vector<char>& read_buf, Count_t initial_vector, Count_t num_vectors_in_section, Grid::GridRedBlackCartesian* fgrid, GridEigenpackReader::Epack_t& eigenpack, const std::string& out_path);

    void read(Grid::GridRedBlackCartesian* fgrid, Grid::GridCartesian* cgrid, const std::string& out_path, int config);
    void readFP32Basis(Grid::GridRedBlackCartesian* fgrid, Epack_t& eigenpack, const std::string& out_path, int config);
    void readFP16Basis(Grid::GridRedBlackCartesian* fgrid, Epack_t& eigenpack, const std::string& out_path, int config);
    void readCoarseEigenvectors(Grid::GridRedBlackCartesian* fgrid, Grid::GridCartesian* cgrid, const std::string& out_path, int config);
};

template<Count_t NUM_BASIS_VECTORS, Count_t NUM_CONCURRENT_BASIS_VECTORS, Count_t NUM_CONCURRENT_EIGEN_VECTORS>
GridEigenpackReader<NUM_BASIS_VECTORS, NUM_CONCURRENT_BASIS_VECTORS, NUM_CONCURRENT_EIGEN_VECTORS>::GridEigenpackReader(const std::string& eigenpack_location, const Dimensions& grid_size, WriteMode DO_WRITE) 
    : EigenpackReaderBase(eigenpack_location, grid_size)
    , DO_WRITE{DO_WRITE}
{
    
}

template<Count_t NUM_BASIS_VECTORS, Count_t NUM_CONCURRENT_BASIS_VECTORS, Count_t NUM_CONCURRENT_EIGEN_VECTORS>
void GridEigenpackReader<NUM_BASIS_VECTORS, NUM_CONCURRENT_BASIS_VECTORS, NUM_CONCURRENT_EIGEN_VECTORS>::read(Grid::GridRedBlackCartesian* fgrid, Grid::GridCartesian* cgrid, const std::string& out_path, int config) 
{   
    using namespace Grid;

    // Read the basis vectors
    {
        Epack_t eigenpack;
        eigenpack.resize(NUM_CONCURRENT_BASIS_VECTORS, 1, fgrid, cgrid); // Segfaults if coarse size set to 0
        for(auto& basis_lattice : eigenpack.evec)
            basis_lattice.Checkerboard() = Odd;
        this->readFP32Basis(fgrid, eigenpack, out_path, config);
        this->readFP16Basis(fgrid, eigenpack, out_path, config);
    }

    // Read the coarse eigenvectors
    this->readCoarseEigenvectors(fgrid, cgrid, out_path, config);
}

template<Count_t NUM_BASIS_VECTORS, Count_t NUM_CONCURRENT_BASIS_VECTORS, Count_t NUM_CONCURRENT_EIGEN_VECTORS>
template<SeekBasisFunc SEEK_TO_VECTOR, ReadBasisFunc READ_DATA>
void GridEigenpackReader<NUM_BASIS_VECTORS, NUM_CONCURRENT_BASIS_VECTORS, NUM_CONCURRENT_EIGEN_VECTORS>::readBasisBase(Offset_t vector_size, Count_t initial_vector, Count_t num_vectors_in_section, Grid::GridRedBlackCartesian* fgrid, GridEigenpackReader::Epack_t& eigenpack, const std::string& out_path, int config)
{
    using namespace Grid;

    int n_procs = 1;
    for (const auto& proc_count : GridDefaultMpi())
        n_procs *= proc_count;

    auto this_proc = fgrid->ThisRank();

    auto vstart = fgrid->LocalStarts().toVector();
    Dimensions lstart{
        .Ls=(unsigned int)vstart[0],
        .x=(unsigned int)vstart[1]*2, 
        .y=(unsigned int)vstart[2],
        .z=(unsigned int)vstart[3],
        .t=(unsigned int)vstart[4]
    };

    auto vdims = fgrid->LocalDimensions().toVector();
    Dimensions ldims{
        .Ls=(unsigned int)vdims[0],
        .x=(unsigned int)vdims[1]*2, 
        .y=(unsigned int)vdims[2],
        .z=(unsigned int)vdims[3],
        .t=(unsigned int)vdims[4]
    };

    Dimensions origin_site{.Ls=0, .x=0, .y=0, .z=0, .t=0 };

    Dimensions pdims = lstart + ldims;
    
    std::vector<int> ldimsvec = {(int)ldims.Ls, (int)ldims.x, (int)ldims.y, (int)ldims.z, (int)ldims.t};
    std::cout << GridLogMessage << "Starting read on process " << (this_proc+1) << "/" << n_procs << std::endl;

    // Read fine eigenvalues
    auto fine_eigenvalues = readFineEigenvalues(this->eigenpack_location);

    // Divide up the number of eigenvectors into batches and read each batch into memory in sequence
    auto number_of_iterations = num_vectors_in_section / NUM_CONCURRENT_BASIS_VECTORS;
    std::vector<char> read_buf;
    read_buf.resize(NUM_CONCURRENT_BASIS_VECTORS*vector_size);
    for (Count_t batch_idx=0; batch_idx < number_of_iterations; ++batch_idx)
    {
        double total_fopen_time          = 0;
        double total_read_time           = 0;
        double total_poke_incl_loop_time = 0;
        double total_poke_prep_time      = 0;
        double total_poke_time           = 0;
        double total_poke_excl_loop_time = 0;
        double total_registereval_time   = 0;
        double time_taken_seconds = 0;

        time_taken_seconds -= usecond();
        auto start_vector = batch_idx*NUM_CONCURRENT_BASIS_VECTORS;
        Offset_t read_size;
        for (Count_t fr_idx=0; fr_idx < this->file_locations.size(); ++fr_idx)
        {
            float old_fopen_time = total_fopen_time;
            total_fopen_time -= usecond();
            auto fr = this->getRAIIFileReader(fr_idx); // Open file, auto-close when out of scope

            if (!(((lstart <= fr.fine_origin).sum() == 5) & ((fr.fine_origin < pdims).sum() == 5)))
            {
                total_fopen_time = old_fopen_time;
                continue;
            }
            PrintProcessesFilereaders(std::vector<int>{(int)fr_idx}, n_procs);

            total_fopen_time += usecond();
            read_size = 0;
            auto& n_sites = this->metadata.stored_sites_per_block;
            iSpinColourVector<ComplexF> fermion_buffer;

            for (Count_t block_idx=0; block_idx < fr.metadata.blocks.size(); ++block_idx)
            {
                auto& block = fr.metadata.blocks[block_idx];
                auto block_origin_lattice_site = block.fine_origin + fr.fine_origin;
                
                (fr.*SEEK_TO_VECTOR)(block_idx, start_vector + initial_vector);
                auto start_pos = fr.tell();

                total_read_time -= usecond();
                auto data = (fr.*READ_DATA)(NUM_CONCURRENT_BASIS_VECTORS, read_buf);
                total_read_time += usecond();
                
                auto bstart = block.sites_in_block[fr.metadata.blocks.size()-1];
                Dimensions bdims{
                    .Ls=(unsigned int)bstart.Ls,
                    .x=(unsigned int)bstart.x*2,
                    .y=(unsigned int)bstart.y,
                    .z=(unsigned int)bstart.z,
                    .t=(unsigned int)bstart.t
                };

                total_poke_incl_loop_time -= usecond();
                for (int batch_vector=0; batch_vector < NUM_CONCURRENT_BASIS_VECTORS; ++batch_vector)
                {
                    total_registereval_time -= usecond();
                    eigenpack.eval[batch_vector] = fine_eigenvalues[batch_vector + start_vector + initial_vector];
                    total_registereval_time += usecond();

                    for (int block_site_idx=0; block_site_idx < n_sites; ++block_site_idx)
                    {
                        total_poke_excl_loop_time -= usecond();
                        total_poke_prep_time      -= usecond();
                        // Get the loop variables
                        auto& fine_site = block.sites_in_block[block_site_idx];
                        auto local_fermion_idx = batch_vector * n_sites + block_site_idx;
                        auto lattice_site = fine_site + block_origin_lattice_site;

                        // Extract the fermion data
                        auto& fermion_data = data[local_fermion_idx];

                        // Convert Fermion to iSpinColourVector
                        int idx = 0;
                        for (int spin=0; spin < 4; ++spin)
                        for (int colour=0; colour < 3; ++colour, idx+=2)
                        {
                            fermion_buffer()(spin)(colour) = ComplexF(fermion_data[idx], fermion_data[idx+1]);
                        }
                        
                        Dimensions local_site = lattice_site - lstart;
                        Coordinate grid_site({ (int)local_site.Ls, (int)local_site.x, (int)local_site.y, (int)local_site.z, (int)local_site.t });

                        total_poke_prep_time += usecond();
                        total_poke_time      -= usecond();

                        if (!(((origin_site <= fine_site).sum() == 5) & ((fine_site < pdims).sum() == 5)))
                        {
                            //PrintOnMainProcess("SITE INDEX ERROR!");
                            std::stringstream stream;
                            stream << "SITE INDEX ERROR!";
                            stream << " " << origin_site.Ls;
                            stream << " " << origin_site.x;
                            stream << " " << origin_site.y;
                            stream << " " << origin_site.z;
                            stream << " " << origin_site.t;
                            stream << " |";
                            stream << " " << fine_site.Ls;
                            stream << " " << fine_site.x;
                            stream << " " << fine_site.y;
                            stream << " " << fine_site.z;
                            stream << " " << fine_site.t;
                            stream << " |";
                            stream << " " << pdims.Ls;
                            stream << " " << pdims.x;
                            stream << " " << pdims.y;
                            stream << " " << pdims.z;
                            stream << " " << pdims.t;
                            throw std::runtime_error(stream.str());
                        }

                        pokeLocalSite(fermion_buffer, eigenpack.evec[batch_vector], grid_site);
                        total_poke_time += usecond();
                        total_poke_excl_loop_time += usecond();
                    }
                }
                total_poke_incl_loop_time += usecond();
                read_size += fr.tell() - start_pos;
            }
        }
        read_size *= this->file_locations.size();
        time_taken_seconds += usecond();

        total_fopen_time          /= 1E6;
        total_read_time           /= 1E6;
        total_poke_incl_loop_time /= 1E6;
        total_poke_excl_loop_time /= 1E6;
        total_poke_prep_time      /= 1E6;
        total_poke_time           /= 1E6;
        total_registereval_time   /= 1E6;
        time_taken_seconds        /= 1E6;

        auto gb_per_sec = ((double)read_size / (1024*1024*1024)) / (time_taken_seconds);
        auto read_gb_per_sec = ((double)read_size / (1024*1024*1024)) / (total_read_time);

        std::cout << GridLogMessage 
        << "Processed batch "
        << batch_idx+1 << "/" << number_of_iterations
        << " of " 
        << NUM_CONCURRENT_BASIS_VECTORS 
        << " basis vectors in " 
        << time_taken_seconds 
        << " @ "
        << gb_per_sec
        << " GiB / s." 
        << std::endl;

        std::cout << GridLogMessage
        << ">>> Spent " 
        << total_fopen_time
        << " seconds opening files and initialising readers ["
        << std::round((total_fopen_time/time_taken_seconds) * 100 * 100)  / 100
        << "% of total]"
        << std::endl;

        std::cout << GridLogMessage
        << ">>> Spent " 
        << total_read_time
        << " seconds in file I/O ["
        << std::round((total_read_time/time_taken_seconds) * 100 * 100)  / 100
        << "% of total]"
        << " @ "
        << read_gb_per_sec
        << " GiB / s." 
        << std::endl;

        std::cout << GridLogMessage
        << ">>> Spent " 
        << total_poke_incl_loop_time
        << " seconds arranging read data into Grid data structures and poking (including loop time) ["
        << std::round((total_poke_incl_loop_time/time_taken_seconds) * 100 * 100)  / 100
        << "% of total]"
        << std::endl;

        std::cout << GridLogMessage
        << ">>> >>> Spent " 
        << total_registereval_time
        << " registering eigenvalues ["
        << std::round((total_registereval_time/total_poke_incl_loop_time) * 100 * 100)  / 100
        << "% of loop time]"
        << std::endl;

        std::cout << GridLogMessage
        << ">>> >>> Spent " 
        << total_poke_excl_loop_time
        << " seconds arranging read data into Grid data structures and poking (excluding loop time) ["
        << std::round((total_poke_excl_loop_time/total_poke_incl_loop_time) * 100 * 100)  / 100
        << "% of loop time]"
        << std::endl;

        std::cout << GridLogMessage
        << ">>> >>> >>> Spent " 
        << total_poke_prep_time
        << " seconds arranging read data into Grid data structures ["
        << std::round((total_poke_prep_time/total_poke_excl_loop_time) * 100 * 100)  / 100
        << "% of loop time]"
        << std::endl;
    
        std::cout << GridLogMessage
        << ">>> >>> >>> Spent " 
        << total_poke_time
        << " seconds poking data ["
        << std::round((total_poke_time/total_poke_excl_loop_time) * 100 * 100)  / 100
        << "% of loop time]"
        << std::endl;

        std::cout << GridLogMessage
        << "### Spent " 
        << std::round(((total_fopen_time+total_read_time+total_poke_incl_loop_time)/time_taken_seconds) * 100 * 100)  / 100
        << "% of total batch execution time in monitored code blocks."
        << std::endl;

        // Write EVs out
        fgrid->Barrier();
        for (int evec_idx=0; evec_idx < NUM_CONCURRENT_BASIS_VECTORS; ++evec_idx)
            std::cout << "Norm: " << norm2(eigenpack.evec[evec_idx]) << std::endl;

        start_vector += initial_vector;
        if (DO_WRITE == WriteMode::HADRONS)
        {
            eigenpack.record.operatorXml="<OperatorXML><DWFPar>GPTEigenpack</DWFPar></OperatorXML>";
            eigenpack.record.solverXml="<SolverXML><LocalCoherenceLanczosPar>GPTEigenpackLCLP</LocalCoherenceLanczosPar></SolverXML>";
            eigenpack.writeFine(out_path + "vec", true, start_vector, start_vector + NUM_CONCURRENT_BASIS_VECTORS, config);
        }
        else if (DO_WRITE == WriteMode::BASIC)
        {
            for (int batch_vector=0; batch_vector < NUM_CONCURRENT_BASIS_VECTORS; ++batch_vector)
            {
                auto basis_vector = start_vector + batch_vector;
                BasicBinary bin(out_path + "basis" + std::to_string(basis_vector) + ".bin");
                bin.write(eigenpack.evec[batch_vector], this->metadata.grid_size);
            }
            std::cout << GridLogMessage << "Completed Basic write." << std::endl;
        }
        fgrid->Barrier();
    }

    std::cout << GridLogMessage << "Completed basis section read on process." << std::endl;

    std::vector<char>().swap(read_buf); // Free the buffer 
}

template<Count_t NUM_BASIS_VECTORS, Count_t NUM_CONCURRENT_BASIS_VECTORS, Count_t NUM_CONCURRENT_EIGEN_VECTORS>
void GridEigenpackReader<NUM_BASIS_VECTORS, NUM_CONCURRENT_BASIS_VECTORS, NUM_CONCURRENT_EIGEN_VECTORS>::readFP32Basis(Grid::GridRedBlackCartesian* fgrid, GridEigenpackReader::Epack_t& eigenpack, const std::string& out_path, int config)
{
    this->readBasisBase<&FileReader::seekToFP32BlockBasisVector, &FileReader::readNextBasisVectorsFP32>
    (
        this->metadata.fp32_sizes.basis_vector_size, 
        0,
        this->metadata.eigenvector_info.n_basis_fp32,
        fgrid, 
        eigenpack, 
        out_path,
        config
    );
}

template<Count_t NUM_BASIS_VECTORS, Count_t NUM_CONCURRENT_BASIS_VECTORS, Count_t NUM_CONCURRENT_EIGEN_VECTORS>
void GridEigenpackReader<NUM_BASIS_VECTORS, NUM_CONCURRENT_BASIS_VECTORS, NUM_CONCURRENT_EIGEN_VECTORS>::readFP16Basis(Grid::GridRedBlackCartesian* fgrid, GridEigenpackReader::Epack_t& eigenpack, const std::string& out_path, int config)
{
    this->readBasisBase<&FileReader::seekToFP16BlockBasisVector, &FileReader::readNextBasisVectorsFP16>
    (
        this->metadata.fp16_sizes.basis_vector_size, 
        this->metadata.eigenvector_info.n_basis_fp32,
        this->metadata.eigenvector_info.n_basis_fp16,
        fgrid, 
        eigenpack, 
        out_path,
        config
    );
}

template<Count_t NUM_BASIS_VECTORS, Count_t NUM_CONCURRENT_BASIS_VECTORS, Count_t NUM_CONCURRENT_EIGEN_VECTORS>
void GridEigenpackReader<NUM_BASIS_VECTORS, NUM_CONCURRENT_BASIS_VECTORS, NUM_CONCURRENT_EIGEN_VECTORS>::readCoarseEigenvectors(Grid::GridRedBlackCartesian* fgrid, Grid::GridCartesian* cgrid, const std::string& out_path, int config)
{
    // Get proc count
    using namespace Grid;

    int n_procs = 1;
    for (const auto& proc_count : GridDefaultMpi())
        n_procs *= proc_count;

    // Create eigenpack holder
    Hadrons::CoarseFermionEigenPack<FIMPLF, NUM_BASIS_VECTORS, FIMPLF> eigenpack;
    eigenpack.resize(0, NUM_CONCURRENT_EIGEN_VECTORS, fgrid, cgrid);

    // Read coarse eigenvalues
    auto coarse_eigenvalues = readCoarseEigenvalues(this->eigenpack_location);

    // Handle COEF
    iVector<ComplexF, NUM_BASIS_VECTORS> coef_buffer;

    // Dims
    auto vstart = cgrid->LocalStarts().toVector();
    Dimensions lstart{
        .Ls=0,
        .x=(unsigned int)vstart[0],
        .y=(unsigned int)vstart[1],
        .z=(unsigned int)vstart[2],
        .t=(unsigned int)vstart[3]
    };

    auto vdims = cgrid->LocalDimensions().toVector();
    Dimensions ldims{
        .Ls=0,
        .x=(unsigned int)vdims[0], 
        .y=(unsigned int)vdims[1],
        .z=(unsigned int)vdims[2],
        .t=(unsigned int)vdims[3]
    };

    Dimensions pdims = lstart + ldims;

    // Currently only reading one basis vector's worth of coefs at a time: if it's a performance issue, we can re-evaluate
    auto number_of_iterations_coef = this->metadata.eigenvector_info.n_eigen / NUM_CONCURRENT_EIGEN_VECTORS;

    std::vector<Count_t> file_indices;
    for (Count_t fr_idx=0; fr_idx < this->file_locations.size(); ++fr_idx)
    {
        auto fr = this->getRAIIFileReader(fr_idx); // Open file, auto-close when out of scope
        if ((((lstart <= fr.block_origin).sum4() == 4) & ((fr.block_origin < pdims).sum4() == 4)))
            file_indices.push_back(fr_idx);
    }

    for (Count_t batch_idx=0; batch_idx < number_of_iterations_coef; ++batch_idx)
    {
        Offset_t read_size=0;
        auto time_taken_seconds = -usecond();
        auto start_vector = batch_idx * NUM_CONCURRENT_EIGEN_VECTORS;
        for (const auto& fr_idx : file_indices)
        {
            auto fr = this->getRAIIFileReader(fr_idx); // Open file, auto-close when out of scope
            for (int batch_vector=0; batch_vector < NUM_CONCURRENT_EIGEN_VECTORS; ++batch_vector)
            {
                auto start_offset = fr.tell();

                auto eigenvector = start_vector + batch_vector;
                fr.seekToCoarseEigenvector(eigenvector);
                for (auto& block: fr.metadata.blocks)
                {
                    // Get the loop variables
                    auto coarse_site = block.coarse_site + fr.block_origin;
                    auto local_site = coarse_site - lstart;
                    // Extract the fermion data
                    auto coef_data = fr.readNextCoarseEigenvector();

                    Coordinate grid_site({ (int)local_site.x, (int)local_site.y, (int)local_site.z, (int)local_site.t });
                    for (int basis_vector=0; basis_vector < fr.metadata.eigenvector_info.n_basis_total; ++basis_vector)
                    {
                        coef_buffer(basis_vector) = ComplexF(coef_data[2*basis_vector], coef_data[2*basis_vector + 1]);    
                    }

                    pokeLocalSite(coef_buffer, eigenpack.evecCoarse[batch_vector], grid_site);
                }
                
                eigenpack.evalCoarse[batch_vector] = coarse_eigenvalues[batch_vector + start_vector];
                read_size += fr.tell() - start_offset;
            }
        }
        
        time_taken_seconds += usecond();
        time_taken_seconds /= 1000000;

        // Print out stats
        auto gb_per_sec = (((double)read_size) / (1024*1024*1024)) / (time_taken_seconds);

        std::cout << GridLogMessage 
        << "Processed batch "
        << batch_idx+1 << "/" << number_of_iterations_coef
        << " of " 
        << NUM_CONCURRENT_EIGEN_VECTORS 
        << " basis vectors in " 
        << time_taken_seconds 
        << " @ "
        << gb_per_sec
        << " GiB / s."
        << std::endl;
        // Write EVs out
        fgrid->Barrier();

        if (DO_WRITE == WriteMode::HADRONS)
        {
            eigenpack.record.operatorXml="<OperatorXML><DWFPar>GPTEigenpack</DWFPar></OperatorXML>";
            eigenpack.record.solverXml="<SolverXML><LocalCoherenceLanczosPar>GPTEigenpackLCLP</LocalCoherenceLanczosPar></SolverXML>";
            eigenpack.writeCoarse(out_path + "vec", true, start_vector, start_vector + NUM_CONCURRENT_EIGEN_VECTORS, config);
        }
        else if (DO_WRITE == WriteMode::BASIC)
        {
            for (int batch_vector=0; batch_vector < NUM_CONCURRENT_EIGEN_VECTORS; ++batch_vector)
            {
                auto basis_vector = start_vector + batch_vector;
                BasicBinary bin(out_path + "eigen" + std::to_string(basis_vector) + ".bin");
                bin.writeCoarse(eigenpack.evecCoarse[batch_vector], this->metadata.grid_size / this->metadata.sites_per_block, this->metadata.eigenvector_info.n_basis_total);
            }
        }

    }

    std::cout << GridLogMessage << "Completed coarse section read on process " << fgrid->ThisRank() << "." << std::endl;
}
