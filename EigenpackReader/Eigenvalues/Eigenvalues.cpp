#include <fstream>

#include "Eigenvalues.hpp"

inline uint32_t try_stoi(const std::string& in)
{
    try
    {
        return std::stoi(in);
    }
    catch(const std::invalid_argument& e)
    {
        throw std::invalid_argument("ERROR: Unable to convert \'" + in + "\' to an integer.");
    }
}

inline float try_stof(const std::string& in)
{
    try
    {
        return std::stof(in);
    }
    catch(const std::invalid_argument& e)
    {
        throw std::invalid_argument("ERROR: Unable to convert \'" + in + "\' to a float.");
    }
}

std::vector<float> readEigenvalues(const std::string& filepath)
{
    std::fstream bytestream;
    bytestream.open(filepath, std::ios_base::in);
    if(!bytestream.is_open())
        throw std::runtime_error("Unable to open file: \'" + filepath + "\'.");

        // Read in data
    std::string line;
    std::getline(bytestream, line);
    auto count = try_stoi(line);

    std::vector<float> out;
    out.reserve(count);
    while(std::getline(bytestream, line))
    {
        auto value = try_stof(line);
        out.push_back(value);
    }
    
    bytestream.close();
    return out;
}

std::vector<float> readCoarseEigenvalues(const std::string& eigenpack_location)
{
    return readEigenvalues(eigenpack_location + "/eigen-values.txt");
}

std::vector<float> readFineEigenvalues(const std::string& eigenpack_location)
{
    return readEigenvalues(eigenpack_location + "/eigen-values.txt");
}
