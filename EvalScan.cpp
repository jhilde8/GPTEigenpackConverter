/*
 * EvalScan.cpp, part of GPTEigenpackConverter
 *
 * Standalone diagnostic: scans a Hadrons-format eigenpack (as written by
 * this converter's writeFine/writeCoarse output) and reports statistics on
 * the eval[] array -- min/max/NaN/Inf/negative/zero/duplicate counts --
 * WITHOUT materializing any of the (large) eigenvector lattice field data.
 *
 * Each vector's eigenvalue is stored as small LIME/XML metadata attached
 * ahead of the (large) binary field payload in the SciDAC record
 * (see Grid::ScidacReader::readScidacFieldRecord in
 * Grid/parallelIO/IldgIO.h). This tool replicates that header-reading
 * sequence but calls skipPastBinaryRecord() instead of materializing the
 * field, so a full ~2200-vector eigenpack scan should take seconds rather
 * than the tens of minutes a full parallel field read costs -- no Grid
 * Cartesian objects are ever constructed.
 *
 * Usage:
 *   EvalScan [--fineDir <dir> --nFine <n>] [--coarseDir <dir> --nCoarse <n>]
 *
 * <dir> is the directory holding that section's v<k>.bin files as written
 * by writeFine/writeCoarse, e.g. ".../vec_fine.1500" or
 * ".../vec_coarse.1500". Reads v<k>.bin for k in [0,n). Either section can
 * be omitted to scan just the other one.
 */

#include <Grid/Grid.h>
#include <Hadrons/EigenPack.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace Grid;

struct EvalStats
{
    unsigned int        count     = 0;
    unsigned int        nanCount  = 0;
    unsigned int        infCount  = 0;
    unsigned int        negCount  = 0;
    unsigned int        zeroCount = 0;
    double               minVal    = std::numeric_limits<double>::infinity();
    double               maxVal    = -std::numeric_limits<double>::infinity();
    std::vector<double> vals;
};

// Reads just the VecRecord (index + eval) for a single-vector SciDAC file,
// skipping the binary field payload entirely.
Hadrons::VecRecord readEvalOnly(const std::string &filename)
{
    ScidacReader        binReader;
    std::string         fileXml;
    FieldMetaData        header;
    scidacRecord         privateRecord;
    Hadrons::VecRecord   vecRecord;

    binReader.open(filename);
    binReader.readLimeObject(fileXml, std::string(SCIDAC_FILE_XML));
    binReader.readLimeObject(header, std::string("FieldMetaData"), std::string(GRID_FORMAT));
    binReader.readLimeObject(vecRecord, vecRecord.SerialisableClassName(), std::string(SCIDAC_RECORD_XML));
    binReader.readLimeObject(privateRecord, privateRecord.SerialisableClassName(), std::string(SCIDAC_PRIVATE_RECORD_XML));
    binReader.skipPastBinaryRecord();
    binReader.close();

    return vecRecord;
}

void scanSection(const std::string &dirStem, unsigned int n, EvalStats &stats)
{
    for (unsigned int k = 0; k < n; ++k)
    {
        std::string filename = dirStem + "/v" + std::to_string(k) + ".bin";
        auto        rec      = readEvalOnly(filename);

        if (rec.index != k)
        {
            std::cout << GridLogMessage << "WARNING: " << filename
                      << " has index " << rec.index << ", expected " << k
                      << std::endl;
        }

        double eval = rec.eval;

        stats.vals.push_back(eval);
        stats.count++;
        if (std::isnan(eval)) stats.nanCount++;
        if (std::isinf(eval)) stats.infCount++;
        if (eval < 0.)        stats.negCount++;
        if (eval == 0.)       stats.zeroCount++;
        if (!std::isnan(eval) && !std::isinf(eval))
        {
            stats.minVal = std::min(stats.minVal, eval);
            stats.maxVal = std::max(stats.maxVal, eval);
        }

        if ((k % 200) == 0)
        {
            std::cout << GridLogMessage << "  scanned " << k << "/" << n << std::endl;
        }
    }
}

void report(const std::string &label, const EvalStats &stats)
{
    std::cout << GridLogMessage << "==== " << label << " (" << stats.count
              << " vectors) ====" << std::endl;
    std::cout << GridLogMessage << "  min           = " << stats.minVal << std::endl;
    std::cout << GridLogMessage << "  max           = " << stats.maxVal << std::endl;
    std::cout << GridLogMessage << "  NaN           = " << stats.nanCount << std::endl;
    std::cout << GridLogMessage << "  Inf           = " << stats.infCount << std::endl;
    std::cout << GridLogMessage << "  negative      = " << stats.negCount << std::endl;
    std::cout << GridLogMessage << "  exactly zero  = " << stats.zeroCount << std::endl;

    // Duplicate check: an eval reused across vectors (e.g. a mis-paired
    // index during conversion) shows up as repeated exact values once
    // sorted, which should essentially never happen for genuine Lanczos
    // Ritz values.
    auto sorted = stats.vals;
    std::sort(sorted.begin(), sorted.end());

    unsigned int dupGroups = 0, dupTotal = 0;
    for (size_t i = 1; i < sorted.size(); ++i)
    {
        if (sorted[i] == sorted[i - 1])
        {
            if (i < 2 || sorted[i] != sorted[i - 2]) dupGroups++;
            dupTotal++;
        }
    }
    std::cout << GridLogMessage << "  exact duplicates = " << dupTotal
              << " (in " << dupGroups << " group(s))" << std::endl;
}

int main(int argc, char *argv[])
{
    Grid_init(&argc, &argv);

    std::string  fineDir;
    std::string  coarseDir;
    unsigned int nFine   = 0;
    unsigned int nCoarse = 0;

    if (GridCmdOptionExists(argv, argv + argc, "--fineDir"))
        fineDir = GridCmdOptionPayload(argv, argv + argc, "--fineDir");
    if (GridCmdOptionExists(argv, argv + argc, "--coarseDir"))
        coarseDir = GridCmdOptionPayload(argv, argv + argc, "--coarseDir");
    if (GridCmdOptionExists(argv, argv + argc, "--nFine"))
        nFine = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nFine"));
    if (GridCmdOptionExists(argv, argv + argc, "--nCoarse"))
        nCoarse = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nCoarse"));

    if ((fineDir.empty() && coarseDir.empty()) ||
        (!fineDir.empty() && nFine == 0) ||
        (!coarseDir.empty() && nCoarse == 0))
    {
        std::cout << GridLogMessage
                  << "Usage: EvalScan [--fineDir <dir> --nFine <n>] [--coarseDir <dir> --nCoarse <n>]"
                  << std::endl;
        std::cout << GridLogMessage
                  << "  <dir> is the directory containing that section's v<k>.bin files, "
                  << "e.g. \".../vec_fine.1500\" or \".../vec_coarse.1500\"."
                  << std::endl;
        Grid_finalize();
        return EXIT_FAILURE;
    }

    if (!fineDir.empty())
    {
        EvalStats fineStats;

        std::cout << GridLogMessage << "Scanning fine/basis section: " << fineDir << std::endl;
        scanSection(fineDir, nFine, fineStats);
        report("fine/basis", fineStats);
    }

    if (!coarseDir.empty())
    {
        EvalStats coarseStats;

        std::cout << GridLogMessage << "Scanning coarse section: " << coarseDir << std::endl;
        scanSection(coarseDir, nCoarse, coarseStats);
        report("coarse", coarseStats);
    }

    Grid_finalize();

    return EXIT_SUCCESS;
}
