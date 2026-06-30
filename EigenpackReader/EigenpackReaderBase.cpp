#include <fstream>

#include "EigenpackReaderBase.hpp"
#include "../Utils/StringManip.hpp"
#include "../Utils/ZipReader.hpp"

EigenpackReaderBase::EigenpackReaderBase(const std::string& eigenpack_location, const Dimensions& grid_size)
    : eigenpack_location{eigenpack_location}
    , metadata{readMetadataFile(eigenpack_location + "/metadata.txt", grid_size)}
    , file_locations{this->metadata.getDirectoryStructure()}
    , zip_mode{false}
{
    auto first_zip = eigenpack_location + "/" + countToDirectory(0) + ".zip";
    if (std::ifstream(first_zip).good())
    {
        zip_mode = true;
        zip_entry_offsets.resize(this->metadata.total_files);

        for (Count_t dir = 0; dir < Metadata::num_directories; ++dir)
        {
            auto zip_path   = eigenpack_location + "/" + countToDirectory(dir) + ".zip";
            auto offsets    = readZipEntryOffsets(zip_path);
            Count_t base    = dir * this->metadata.files_per_directory;

            for (Count_t f = 0; f < this->metadata.files_per_directory; ++f)
            {
                auto entry_name = countToDirectory(dir) + "/" + countToFilename(base + f);
                auto it = offsets.find(entry_name);
                if (it == offsets.end())
                    throw std::runtime_error("Entry not found in " + zip_path + ": " + entry_name);
                zip_entry_offsets[base + f] = it->second;
            }
        }
    }
}

FileReader EigenpackReaderBase::getRAIIFileReader(Count_t idx)
{
    auto& loc = this->file_locations[idx];

    if (zip_mode)
    {
        auto zip_path   = this->eigenpack_location + "/" + countToDirectory(loc.directory) + ".zip";
        auto base_offset = zip_entry_offsets[loc.file];
        return FileReader{ loc.file, zip_path, base_offset, this->metadata };
    }

    auto filepath = this->eigenpack_location + "/" + countToDirectory(loc.directory) + "/" + countToFilename(loc.file);
    return FileReader{ loc.file, filepath, this->metadata };
}

