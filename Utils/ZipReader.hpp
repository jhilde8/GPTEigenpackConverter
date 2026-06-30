#pragma once

#include <string>
#include <unordered_map>

#include "Typedefs.hpp"

// Parse the central directory of a store-only zip file and return the absolute byte
// offset of the raw data for each entry, keyed by entry filename. Handles ZIP64.
// The bytes at each returned offset are identical to the original uncompressed file.
std::unordered_map<std::string, Offset_t> readZipEntryOffsets(const std::string& zip_path);
