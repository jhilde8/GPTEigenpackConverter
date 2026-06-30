#pragma once

#include <vector>

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
    bool zip_mode;
    std::vector<Offset_t> zip_entry_offsets; // indexed by global file index

    EigenpackReaderBase(const std::string& eigenpack_location, const Dimensions& grid_size);
    FileReader getRAIIFileReader(Count_t idx);
};
