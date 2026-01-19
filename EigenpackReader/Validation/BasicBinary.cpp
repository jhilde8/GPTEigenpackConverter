#include "BasicBinary.hpp"


BasicBinary::BasicBinary(const std::string& filepath) 
    : filepath{filepath} 
{
    this->bytestream.open(filepath);
}

BasicBinary::~BasicBinary()
{
    this->bytestream.close();
}
