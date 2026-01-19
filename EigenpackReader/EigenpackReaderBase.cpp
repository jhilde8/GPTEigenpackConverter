#include "EigenpackReaderBase.hpp"
#include "../Utils/StringManip.hpp"

EigenpackReaderBase::EigenpackReaderBase(const std::string& eigenpack_location, const Dimensions& grid_size)
    : eigenpack_location{eigenpack_location}
    , metadata{readMetadataFile(eigenpack_location + "/metadata.txt", grid_size)}
    , file_locations{this->metadata.getDirectoryStructure()}
{}

FileReader EigenpackReaderBase::getRAIIFileReader(Count_t idx)
{
    auto& loc = this->file_locations[idx];

    auto filepath = this->eigenpack_location + "/" + countToDirectory(loc.directory) + "/" + countToFilename(loc.file);


    return FileReader{ loc.file, filepath, this->metadata };
}

