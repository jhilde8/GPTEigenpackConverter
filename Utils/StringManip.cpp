#include <iomanip>
#include <sstream>

#include "StringManip.hpp"

std::string leftPad(size_t value, size_t result_size, const std::string& character)
{
    std::stringstream ss;
    ss << std::setw(result_size) << std::setfill('0') << value;
    return ss.str();
}

std::string countToFilename(size_t counter)
{
    return leftPad(counter, 10, "0") + ".compressed";
}

std::string countToDirectory(size_t counter)
{
    return leftPad(counter, 2, "0");
}