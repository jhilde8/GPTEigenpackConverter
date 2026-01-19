#pragma once

#include "FileReader/FileReader.hpp"
#include "Metadata/Metadata.hpp"

class EigenpackReaderBase
{
protected:
    typedef std::array<float, 24> Fermion_t;
public:
    const std::string eigenpack_location;
    const Metadata metadata;
    std::vector<FileLocationInfo> file_locations;
    
    EigenpackReaderBase(const std::string& eigenpack_location, const Dimensions& grid_size);
    FileReader getRAIIFileReader(Count_t idx);
};
