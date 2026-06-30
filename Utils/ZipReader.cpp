#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "ZipReader.hpp"

namespace {

uint16_t read16(const char* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
uint32_t read32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint64_t read64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// ZIP record signatures
constexpr uint32_t SIG_LFH       = 0x04034b50;
constexpr uint32_t SIG_CD        = 0x02014b50;
constexpr uint32_t SIG_EOCD      = 0x06054b50;
constexpr uint32_t SIG_EOCD64    = 0x06064b50;
constexpr uint32_t SIG_EOCD64LOC = 0x07064b50;

// ZIP64 extra field tag
constexpr uint16_t TAG_ZIP64 = 0x0001;

// Sentinel values in standard fields indicating the real value is in a ZIP64 extension
constexpr uint16_t ZIP64_SENTINEL16 = 0xFFFF;
constexpr uint32_t ZIP64_SENTINEL32 = 0xFFFFFFFF;

// End of Central Directory record (22 bytes fixed)
constexpr int EOCD_SIZE             = 22;
constexpr int EOCD_OFF_TOTAL_ENTRIES = 10;
constexpr int EOCD_OFF_CD_SIZE      = 12;
constexpr int EOCD_OFF_CD_OFFSET    = 16;

// ZIP64 End of Central Directory Locator (20 bytes fixed)
constexpr int EOCD64_LOC_SIZE       = 20;
constexpr int EOCD64_LOC_OFF_OFFSET = 8;

// ZIP64 End of Central Directory record (56 bytes fixed)
constexpr int EOCD64_SIZE           = 56;
constexpr int EOCD64_OFF_N_ENTRIES  = 32;
constexpr int EOCD64_OFF_CD_SIZE    = 40;
constexpr int EOCD64_OFF_CD_OFFSET  = 48;

// Central Directory record (46 bytes fixed, followed by filename + extra + comment)
constexpr int CD_FIXED_SIZE         = 46;
constexpr int CD_OFF_COMP_SIZE      = 20;
constexpr int CD_OFF_UNCOMP_SIZE    = 24;
constexpr int CD_OFF_FNAME_LEN      = 28;
constexpr int CD_OFF_EXTRA_LEN      = 30;
constexpr int CD_OFF_COMMENT_LEN    = 32;
constexpr int CD_OFF_LFH_OFFSET     = 42;
constexpr int CD_OFF_FNAME          = 46;

// Local File Header (30 bytes fixed, followed by filename + extra + file data)
constexpr int LFH_FIXED_SIZE        = 30;
constexpr int LFH_OFF_FNAME_LEN     = 26;
constexpr int LFH_OFF_EXTRA_LEN     = 28;

// ZIP64 extra field: 2-byte tag + 2-byte size + data
constexpr int ZIP64_EXTRA_HEADER_SIZE = 4;

} // namespace

std::unordered_map<std::string, Offset_t> readZipEntryOffsets(const std::string& zip_path)
{
    std::ifstream f(zip_path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open zip file: " + zip_path);

    f.seekg(0, std::ios::end);
    const int64_t file_size = static_cast<int64_t>(f.tellg());
    if (file_size < EOCD_SIZE)
        throw std::runtime_error("File too small to be a zip: " + zip_path);

    // Locate the EOCD record by searching backwards from the end of the file.
    // Fast path: no zip comment, so EOCD is the last 22 bytes.
    // Fall back to a full backwards search to handle unexpected comments (up to 65535 bytes).
    const int64_t max_comment = 65535;
    const int64_t search_size = std::min(file_size, static_cast<int64_t>(EOCD_SIZE + max_comment));
    std::vector<char> tail(search_size);
    f.seekg(file_size - search_size);
    f.read(tail.data(), search_size);

    int64_t eocd_pos = -1;
    for (int64_t i = search_size - EOCD_SIZE; i >= 0; --i)
    {
        if (read32(tail.data() + i) == SIG_EOCD)
        {
            eocd_pos = (file_size - search_size) + i;
            break;
        }
    }
    if (eocd_pos < 0)
        throw std::runtime_error("EOCD record not found in: " + zip_path);

    const char* eocd = tail.data() + (eocd_pos - (file_size - search_size));

    uint64_t n_entries = read16(eocd + EOCD_OFF_TOTAL_ENTRIES);
    uint64_t cd_size   = read32(eocd + EOCD_OFF_CD_SIZE);
    uint64_t cd_offset = read32(eocd + EOCD_OFF_CD_OFFSET);

    // If any field is at its sentinel value, this is a ZIP64 archive.
    if (n_entries == ZIP64_SENTINEL16 || cd_size == ZIP64_SENTINEL32 || cd_offset == ZIP64_SENTINEL32)
    {
        // The ZIP64 EOCD locator sits immediately before the standard EOCD.
        if (eocd_pos < EOCD64_LOC_SIZE)
            throw std::runtime_error("ZIP64 EOCD locator not found in: " + zip_path);

        char locator[EOCD64_LOC_SIZE];
        f.seekg(eocd_pos - EOCD64_LOC_SIZE);
        f.read(locator, EOCD64_LOC_SIZE);
        if (read32(locator) != SIG_EOCD64LOC)
            throw std::runtime_error("ZIP64 EOCD locator signature mismatch in: " + zip_path);

        const uint64_t eocd64_off = read64(locator + EOCD64_LOC_OFF_OFFSET);

        char eocd64[EOCD64_SIZE];
        f.seekg(static_cast<std::streamoff>(eocd64_off));
        f.read(eocd64, EOCD64_SIZE);
        if (read32(eocd64) != SIG_EOCD64)
            throw std::runtime_error("ZIP64 EOCD signature mismatch in: " + zip_path);

        n_entries = read64(eocd64 + EOCD64_OFF_N_ENTRIES);
        cd_size   = read64(eocd64 + EOCD64_OFF_CD_SIZE);
        cd_offset = read64(eocd64 + EOCD64_OFF_CD_OFFSET);
    }

    // Read the entire central directory in one shot.
    std::vector<char> cd(cd_size);
    f.seekg(static_cast<std::streamoff>(cd_offset));
    f.read(cd.data(), static_cast<std::streamsize>(cd_size));

    std::unordered_map<std::string, Offset_t> entry_offsets;
    entry_offsets.reserve(n_entries);

    size_t pos = 0;
    for (uint64_t i = 0; i < n_entries; ++i)
    {
        if (pos + CD_FIXED_SIZE > cd.size())
            throw std::runtime_error("Central directory truncated in: " + zip_path);
        if (read32(cd.data() + pos) != SIG_CD)
            throw std::runtime_error("Central directory signature mismatch in: " + zip_path);

        const uint16_t fname_len   = read16(cd.data() + pos + CD_OFF_FNAME_LEN);
        const uint16_t extra_len   = read16(cd.data() + pos + CD_OFF_EXTRA_LEN);
        const uint16_t comment_len = read16(cd.data() + pos + CD_OFF_COMMENT_LEN);
        const uint32_t cd_csize    = read32(cd.data() + pos + CD_OFF_COMP_SIZE);
        const uint32_t cd_usize    = read32(cd.data() + pos + CD_OFF_UNCOMP_SIZE);
        uint64_t lfh_offset        = read32(cd.data() + pos + CD_OFF_LFH_OFFSET);

        std::string fname(cd.data() + pos + CD_OFF_FNAME, fname_len);

        // If the local file header offset is at its sentinel value, the real offset
        // is in the ZIP64 extra field. Fields appear in order: original size, compressed
        // size, local header offset - but only the fields whose CD counterpart is at
        // its sentinel value are present.
        if (lfh_offset == ZIP64_SENTINEL32)
        {
            const char* extra = cd.data() + pos + CD_OFF_FNAME + fname_len;
            size_t ep = 0;
            while (ep + ZIP64_EXTRA_HEADER_SIZE <= static_cast<size_t>(extra_len))
            {
                const uint16_t tag  = read16(extra + ep);
                const uint16_t size = read16(extra + ep + 2);
                if (tag == TAG_ZIP64)
                {
                    const char* z64 = extra + ep + ZIP64_EXTRA_HEADER_SIZE;
                    size_t skip = 0;
                    if (cd_usize == ZIP64_SENTINEL32) skip += 8;
                    if (cd_csize == ZIP64_SENTINEL32) skip += 8;
                    lfh_offset = read64(z64 + skip);
                    break;
                }
                ep += ZIP64_EXTRA_HEADER_SIZE + size;
            }
            if (lfh_offset == ZIP64_SENTINEL32)
                throw std::runtime_error("ZIP64 local header offset not found for entry: " + fname);
        }

        // Read the local file header to get its actual extra field length, which can
        // differ from the central directory extra field length.
        char lfh[LFH_FIXED_SIZE];
        f.seekg(static_cast<std::streamoff>(lfh_offset));
        f.read(lfh, LFH_FIXED_SIZE);
        if (read32(lfh) != SIG_LFH)
            throw std::runtime_error("Local file header signature mismatch for entry: " + fname);

        const uint16_t lfh_fname_len = read16(lfh + LFH_OFF_FNAME_LEN);
        const uint16_t lfh_extra_len = read16(lfh + LFH_OFF_EXTRA_LEN);
        entry_offsets[fname] = lfh_offset + LFH_FIXED_SIZE + lfh_fname_len + lfh_extra_len;

        pos += CD_FIXED_SIZE + fname_len + extra_len + comment_len;
    }

    return entry_offsets;
}
