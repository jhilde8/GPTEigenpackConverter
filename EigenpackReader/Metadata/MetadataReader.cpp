#include "MetadataReader.hpp"

//###################//
// UTILITY FUNCTIONS //
//###################//

inline uint32_t try_stoi(const std::string& key, const std::string& in)
{
    try
    {
        return std::stoi(in);
    }
    catch(const std::invalid_argument& e)
    {
        throw std::invalid_argument("ERROR: Unable to convert metadata \'" + key + "\' entry \'" + in + "\' to an integer.");
    }
}

//#######################//
// METADATA READER CLASS //
//#######################//

/// This is an auxilliary class intended to be inaccessible outside of the translation unit.
/// The purpose is to read a metadata.txt file and construct a Metadata object from it.


// RAII Constructor
// Opens the metadata file at {data_directory}/metadata.txt.
// Does not perform a read - call read() to read the contents and return a Metadata object.
// The only reason this is a class and not a set of functions is for RAII,
// which may not be necessary since std::fstream is supposed to be RAII.
MetadataReader::MetadataReader(const std::string& metadata_path)
{
    this->bytestream.open(metadata_path, std::ios_base::in);
    if(!this->bytestream.is_open())
        throw std::runtime_error("Unable to open file: \'" + metadata_path + "\'.");
}

// RAII Destructor
// Closes the filestream, which may not be necessary since std::fstream is allegedly RAII.
MetadataReader::~MetadataReader()
{
    this->bytestream.close();
}

// read
// Reads the metadata file that the class has been opened with.
// Returns a Metadata object that can be used to init the FileReaders.
Metadata MetadataReader::read(const Dimensions& grid_size)
{
    MetadataReader::MetadataLookup readin;

    // Read in data
    std::string line;
    while(std::getline(this->bytestream, line))
    {
        // Ignore any metadata entry beginning with "crc32"
        // Maybe we'll add it back in for validation checks later
        static const std::string ignore_string = "crc32";
        if (line.substr(0, ignore_string.size()) == ignore_string)
            continue;
        
        // Parse everything else and put it into a lookup table
        uint32_t separator_idx = line.find("=");
        std::string key = line.substr(0, separator_idx-1);
        auto value = try_stoi(key, line.substr(separator_idx+1, line.length()));
        readin[key] = value;
    }

    // Check it contains all we need to construct the Metadata class
    this->validateMetadata(readin);

    // Construct + return, RVO should kick in but will be a vanishingly small optimisation
    return Metadata(grid_size, readin);

}

// validateMetadata
// Checks that all required data have been read from the metadata file.
void MetadataReader::validateMetadata(const MetadataReader::MetadataLookup& readin)
{
    // This is a range-based for loop, it just looks weird due to the long array initialiser
    for 
    (
        const std::string& element : 
        {
             "s[0]", "s[1]", "s[2]", "s[3]", "s[4]",
             "b[0]", "b[1]", "b[2]", "b[3]", "b[4]",
            "nb[0]","nb[1]","nb[2]","nb[3]","nb[4]",
            "neig", "nkeep", "nkeep_single", "blocks", "FP16_COEF_EXP_SHARE_FLOATS"
        }
    )
    {
        if (!readin.count(element))
            throw std::runtime_error("Could not find required member \'" + element + "\' in Metadata file.");
    }
}
