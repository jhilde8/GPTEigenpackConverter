#pragma once

#include "EigenpackReaderBase.hpp"
#include "FileReader/FileReader.hpp"
#include "Metadata/Metadata.hpp"

class EigenpackReader : public EigenpackReaderBase
{
public:
    std::vector<FileReader> filereaders;
    
    EigenpackReader(const std::string& eigenpack_location, const Dimensions& grid_size) 
        : EigenpackReaderBase(eigenpack_location, grid_size) 
    {
        this->filereaders.reserve(this->file_locations.size());
        for (Count_t i=0; i < this->file_locations.size(); ++i)
            this->filereaders.emplace_back(this->getRAIIFileReader(i));
    };
    Fermion_t readFP32BasisVector(const Dimensions& lattice_site, Count_t which_basis_vector);
    Fermion_t readFP16BasisVector(const Dimensions& lattice_site, Count_t which_basis_vector);
    std::vector<float> readCoarseEigenvector(const Dimensions& coarse_site, Count_t which_eigenvector);
};
