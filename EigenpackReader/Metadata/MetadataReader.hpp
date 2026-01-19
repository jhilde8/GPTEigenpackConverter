#pragma once

#include <fstream>
#include <unordered_map>

#include "Metadata.hpp"

class MetadataReader
{
protected:
    typedef std::unordered_map<std::string, Count_t> MetadataLookup;
    std::fstream bytestream;
    std::string data_directory;
public:
    MetadataReader(const std::string& filepath);
    ~MetadataReader();
    Metadata read(const Dimensions& grid_size);
    void validateMetadata(const MetadataLookup&);
};
