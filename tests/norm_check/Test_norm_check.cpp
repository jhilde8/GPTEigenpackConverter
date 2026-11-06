#include <Grid/Grid.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>

using namespace Grid;

typedef LatticeFermionF          Field;
typedef Field::vector_object     vobj;
typedef vobj::scalar_object      sobj;

struct LimeScan
{
    uint64_t    payloadOffset = 0;
    uint64_t    payloadBytes  = 0;
    std::string normXml;
    std::string checksumXml;
};

static uint64_t readBigEndian64(const unsigned char *p)
{
    uint64_t v = 0;

    for (int i = 0; i < 8; ++i)
    {
        v = (v << 8) | p[i];
    }

    return v;
}

static LimeScan scanLime(const std::string &filename)
{
    LimeScan      scan;
    std::ifstream in(filename, std::ios::binary);
    uint64_t      off = 0;
    unsigned char h[144];

    if (!in.is_open())
    {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    while (in.seekg(off), in.read(reinterpret_cast<char *>(h), 144))
    {
        uint64_t    n    = readBigEndian64(h + 8);
        std::string type(reinterpret_cast<char *>(h + 16), strnlen(reinterpret_cast<char *>(h + 16), 128));
        uint64_t    data = off + 144;

        std::cout << GridLogMessage << "LIME record @" << off << " " << type << " " << n << " bytes" << std::endl;
        if (type == "ildg-binary-data" || type == "scidac-binary-data")
        {
            scan.payloadOffset = data;
            scan.payloadBytes  = n;
        }
        else if (type == "FieldNormMetaData" || type == "scidac-checksum")
        {
            std::string text(n, '\0');

            in.seekg(data);
            in.read(&text[0], n);
            (type == "FieldNormMetaData" ? scan.normXml : scan.checksumXml) = text;
        }
        off = data + ((n + 7) / 8) * 8;
    }

    return scan;
}

struct RankSums
{
    double rankInner;
    double localNorm2Lanes;
    double rawComponents;
};

static RankSums localSums(const Field &field)
{
    RankSums  s;
    GridBase *g     = field.Grid();
    int       nsimd = g->Nsimd();
    auto      ln    = localNorm2(field);
    int       nflt  = sizeof(sobj) / sizeof(RealF);

    s.rankInner       = real(rankInnerProduct(field, field));
    s.localNorm2Lanes = 0.;
    s.rawComponents   = 0.;
    {
        autoView(f_v, field, CpuRead);
        autoView(l_v, ln, CpuRead);

        for (uint64_t ss = 0; ss < g->oSites(); ++ss)
        {
            for (int lane = 0; lane < nsimd; ++lane)
            {
                sobj         site = extractLane(lane, f_v[ss]);
                const RealF *p    = reinterpret_cast<const RealF *>(&site);

                for (int i = 0; i < nflt; ++i)
                {
                    s.rawComponents += static_cast<double>(p[i]) * static_cast<double>(p[i]);
                }
                s.localNorm2Lanes += static_cast<double>(real(TensorRemove(extractLane(lane, l_v[ss]))));
            }
        }
    }

    return s;
}

int main(int argc, char *argv[])
{
    Grid_init(&argc, &argv);

    std::string filename;
    int         Ls = 12;

    if (!GridCmdOptionExists(argv, argv + argc, "--file"))
    {
        std::cerr << "Usage: Test_norm_check --file <v.bin> [--Ls 12] --grid X.Y.Z.T --mpi a.b.c.d" << std::endl;
        Grid_finalize();
        return EXIT_FAILURE;
    }
    filename = GridCmdOptionPayload(argv, argv + argc, "--file");
    if (GridCmdOptionExists(argv, argv + argc, "--Ls"))
    {
        Ls = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--Ls"));
    }

    GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                         GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);
    Field                  field(FrbGrid);
    LimeScan               scan;
    uint64_t               expected = sizeof(sobj) * FrbGrid->_gsites;

    if (FrbGrid->IsBoss())
    {
        scan = scanLime(filename);
        std::cout << GridLogMessage << "FieldNormMetaData: " << scan.normXml << std::endl;
        std::cout << GridLogMessage << "scidac-checksum:   " << scan.checksumXml << std::endl;
    }
    FrbGrid->Broadcast(0, &scan.payloadOffset, sizeof(scan.payloadOffset));
    FrbGrid->Broadcast(0, &scan.payloadBytes, sizeof(scan.payloadBytes));
    if (scan.payloadBytes != expected)
    {
        std::cerr << "Payload is " << scan.payloadBytes << " bytes, grid expects " << expected << std::endl;
        Grid_finalize();
        return EXIT_FAILURE;
    }

    uint32_t                       nersc, csuma, csumb;
    BinarySimpleMunger<sobj, sobj> munge;

    BinaryIO::readLatticeObject<vobj, sobj>(field, filename, munge, scan.payloadOffset,
                                            getFormatString<vobj>(), nersc, csuma, csumb);
    field.Checkerboard() = Odd;
    std::cout << GridLogMessage << "Read checksums: scidac A " << std::hex << csuma
              << " B " << csumb << std::dec << std::endl;

    RankSums local = localSums(field);
    double   A     = norm2(field);
    double   D     = local.rankInner;
    double   B     = local.localNorm2Lanes;
    double   C     = local.rawComponents;

    FrbGrid->GlobalSum(D);
    FrbGrid->GlobalSum(B);
    FrbGrid->GlobalSum(C);

    std::cout << std::setprecision(15);
    std::cout << GridLogMessage << "A norm2(field)                              " << A << std::endl;
    std::cout << GridLogMessage << "D rankInnerProduct + GlobalSum              " << D << std::endl;
    std::cout << GridLogMessage << "B localNorm2, lanes summed in double        " << B << std::endl;
    std::cout << GridLogMessage << "C raw float components squared, in double   " << C << std::endl;

    int                 nrank = FrbGrid->ProcessorCount();
    int                 me    = FrbGrid->ThisRank();
    std::vector<double> perD(nrank, 0.), perB(nrank, 0.), perC(nrank, 0.);

    perD[me] = local.rankInner;
    perB[me] = local.localNorm2Lanes;
    perC[me] = local.rawComponents;
    FrbGrid->GlobalSumVector(perD.data(), nrank);
    FrbGrid->GlobalSumVector(perB.data(), nrank);
    FrbGrid->GlobalSumVector(perC.data(), nrank);

    if (FrbGrid->IsBoss())
    {
        std::vector<int> order(nrank);

        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b)
        {
            return std::abs(perC[a] - perD[a]) > std::abs(perC[b] - perD[b]);
        });
        std::cout << GridLogMessage << "Ranks by |C - D| (rank: C  B  D  C-D):" << std::endl;
        for (int i = 0; i < std::min(nrank, 8); ++i)
        {
            int r = order[i];

            std::cout << GridLogMessage << "  " << r << ": " << perC[r] << "  " << perB[r]
                      << "  " << perD[r] << "  " << perC[r] - perD[r] << std::endl;
        }
    }

    Grid_finalize();

    return EXIT_SUCCESS;
}
