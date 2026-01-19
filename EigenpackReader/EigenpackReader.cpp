#include "EigenpackReader.hpp"


EigenpackReader::Fermion_t EigenpackReader::readFP32BasisVector(const Dimensions& lattice_site, Count_t which_basis_vector)
{
    auto loc = this->metadata.getFileOfLatticeSite(lattice_site);
    return this->filereaders[loc].readLatticeFermionFP32AtLatticeSite(lattice_site, which_basis_vector);
}

EigenpackReader::Fermion_t EigenpackReader::readFP16BasisVector(const Dimensions& lattice_site, Count_t which_basis_vector)
{
    auto loc = this->metadata.getFileOfLatticeSite(lattice_site);
    return this->filereaders[loc].readLatticeFermionFP16AtLatticeSite(lattice_site, which_basis_vector);
}

std::vector<float> EigenpackReader::readCoarseEigenvector(const Dimensions& coarse_site, Count_t which_eigenvector)
{
    auto loc = this->metadata.getFileOfCoarseSite(coarse_site);
    return this->filereaders[loc].readCoarseEigenvectorAtCoarseSite(coarse_site, which_eigenvector);
}
