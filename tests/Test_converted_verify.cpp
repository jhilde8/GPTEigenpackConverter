/*
 * Test_converted_verify.cpp
 *
 * Standalone data-integrity check for the output of GPTEigenpackConverter.
 * Intended to run right after the converter finishes, on the same CPU
 * allocation/build (e.g. Andes), as a quick pass/fail before committing to
 * a full production a2a solve.
 *
 * Checks that a block-promoted coarse eigenvector is an approximate
 * eigenvector of Mpc^dag Mpc, i.e. reproduces what
 * LocalCoherenceLanczos::getFineEvecEval() + testCoarse() check right after
 * a real Lanczos run (Grid/algorithms/iterative/LocalCoherenceLanczos.h:
 * 358-378, 444-448). This is NOT the same as testing the raw fine ("_fine")
 * basis vectors directly: those are only converged Ritz vectors of a
 * Chebyshev-accelerated operator, kept as a *spanning subspace* for the
 * near-null space, and are not individually expected to be good
 * eigenvectors of the true (unaccelerated) Mpc^dag Mpc -- reconstructing
 * one from a lone fine vector's stored eigenvalue can be many orders of
 * magnitude off even for perfectly healthy data (confirmed empirically:
 * both the 64I data under test *and* 48I, an independently-validated
 * working ensemble, showed the same catastrophic mismatch under a raw
 * fine-vector-only check). The physically meaningful eigenvector only
 * exists after combining ALL fine basis vectors, weighted by one row of
 * the coarse eigenvector:
 *
 *   blockPromote(evecCoarse[i], evec, subspace);  // subspace = full fine basis
 *   eval = evalCoarse[i];
 *
 * which is exactly what this test now does, reading the full fine basis
 * (all sizeFine vectors from "<filestem>_fine") plus the first nCheck
 * coarse eigenvectors (the lowest/most physically relevant, from
 * "<filestem>_coarse"/v0.bin upward).
 *
 * Reads via Hadrons::CoarseFermionEigenPack<FImpl, nBasis>, the same reader
 * class LoadCoarseFermionEigenPack200F/etc. use in production -- no
 * separate coarse-grid module needed, since the coarse grid geometry
 * (SpaceTimeGrid::makeFourDimGrid/makeFiveDimGrid + a valid SIMD/MPI
 * layout for the coarse dimensions) doesn't need to match whatever layout
 * Hadrons' own Environment happened to pick; blockPromote/blockProject and
 * Grid's parallel binary I/O are both layout-independent. Grid geometry is
 * otherwise built directly from --grid/--mpi like any plain Grid program.
 *
 * --ens selects which ensemble's action/eigenpack parameters to use.
 * These are no longer separate flags (Ls, Schur convention): a mismatched
 * combination (wrong Schur convention or Ls for a given action) is not
 * meaningful, so both are bundled into the --ens preset:
 *   - 64I (default): standard Mobius, mass/M5/b/c/Ls/boundary/twist taken
 *     from par.CGl.1500.xml's mdwf_l/mdwff_l (b=1.5, c=0.5, Ls=12).
 *     nBasis=200 (sizeFine), matches production's epack_coarse
 *     (MIO::LoadCoarseFermionEigenPack200F, blockSize 4.4.4.4.12). Schur
 *     convention: SchurDiagOneOperator (matches GPT's schur_complement_one,
 *     which is what produced this eigenpack).
 *   - 48I: z-Mobius (MAction::ZMobiusDWF in production), mass/M5/b/c/omega/
 *     Ls/boundary/twist taken from the 48I zmdwf_l XML block (b=1.0, c=0.0,
 *     Ls=14, 14 complex omega coefficients). nBasis=400 (sizeFine),
 *     blockSize 4.3.3.3.14. Schur convention: SchurDiagTwoOperator (48I is
 *     Hadrons' default production convention). Used as a control: 48I is a
 *     working, already-validated production ensemble, so if this same test
 *     harness passes cleanly on 48I's converted vectors, that isolates a
 *     64I-specific failure to something about that specific converted pack
 *     (or its conversion) rather than a bug in this test itself.
 * --gauge/--filestem/--traj/--grid/--mpi remain independent of --ens.
 *
 * Motivation: production a2a solves show a very bad deflation guess -- CG
 * barely moves off its undeflated trajectory in 400 inner iterations. The
 * guesser wiring (EigenPackLCDecompress -> ExactDeflation) was checked and
 * matches production, action parameters/Ls/boundary/twist were checked
 * against par.CGl.1500.xml and match exactly, the gauge field/trajectory
 * were confirmed correct, and DiagOne vs DiagTwo only changes the
 * reconstructed eigenvalue by a fraction of a percent -- so this test now
 * checks the one thing that hadn't actually been checked correctly yet:
 * whether the fully-reconstructed (fine + coarse) eigenvectors are what
 * they claim to be.
 *
 * Usage:
 *   ./Test_converted_verify --grid 64.64.64.128 --mpi 4.4.4.4 \
 *       --gauge /path/to/ckpoint_lat \
 *       --filestem /path/to/converted/vec \
 *       --traj 1500 --nCheck 5 --resid 1e-3 --ens 64I
 *
 * A vector that's intact should reconstruct an eigenvalue close to the
 * stored one and pass the residual target; loosen/tighten --resid to
 * whatever tolerance the coarse Lanczos was actually run at. An O(1)
 * residual on a vector regardless of --resid means that reconstruction
 * simply isn't an eigenvector of this operator (corrupt read, index/
 * ordering mismatch, or a real bug in the fine or coarse data).
 */
#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedLanczos.h>
#include <Hadrons/EigenPack.hpp>

using namespace Grid;
using namespace Hadrons;

// Runs the Mpc^dag Mpc residual check on block-promoted coarse
// eigenvectors, for a given Schur convention (SchurOp is
// SchurDiagOneOperator<FMat, Field> or SchurDiagTwoOperator<FMat, Field>,
// fully specified by the caller). FMat/Field/CoarseField are deduced from
// Ddwf/subspace/evecCoarse, so this works unchanged for both the 64I
// (Mobius) and 48I (z-Mobius) action types. Returns the number of vectors
// that failed.
template <typename SchurOp, typename FMat, typename Field, typename CoarseField>
unsigned int checkVectors(FMat &Ddwf, std::vector<Field> &subspace,
                          std::vector<CoarseField> &evecCoarse, std::vector<RealD> &evalCoarse,
                          unsigned int nCheck, double resid, const std::string &label)
{
    SchurOp                                      schurOp(Ddwf);
    PlainHermOp<Field>                           hermOp(schurOp);
    ImplicitlyRestartedLanczosHermOpTester<Field> tester(hermOp);
    unsigned int                                 nFail = 0;
    Field                                        evec(subspace[0].Grid());

    evec.Checkerboard() = subspace[0].Checkerboard();

    std::cout << GridLogMessage << "Checking " << nCheck
              << " block-promoted coarse eigenvector(s) against Mpc^dag Mpc (" << label << "), "
              << "target residual " << resid << std::endl;

    for (unsigned int i = 0; i < nCheck; ++i)
    {
        blockPromote(evecCoarse[i], evec, subspace);

        RealD evalStored = evalCoarse[i];
        RealD evalRecon  = evalStored;
        int   conv       = tester.TestConvergence(i, resid, evec, evalRecon, 1.0);
        RealD relDiff    = (evalStored != 0.)
                          ? std::abs(evalRecon - evalStored) / std::abs(evalStored)
                          : std::abs(evalRecon - evalStored);

        std::cout << GridLogMessage << "  evec " << i
                  << ": stored eval = "        << evalStored
                  << ", reconstructed eval = " << evalRecon
                  << ", rel eval diff = "      << relDiff
                  << (conv ? "  [OK]" : "  [FAIL]") << std::endl;

        if (!conv)
        {
            ++nFail;
        }
    }

    std::cout << GridLogMessage << nFail << " / " << nCheck
              << " coarse eigenvectors FAILED the residual check (" << label << ")" << std::endl;

    return nFail;
}

int main(int argc, char *argv[])
{
    std::string  gaugeFile = "";
    std::string  filestem  = "";
    int          traj      = -1;
    unsigned int nCheck    = 5; // number of coarse eigenvectors to promote and check
    double       resid     = 1e-3;
    std::string  ens       = "64I";

    if (GridCmdOptionExists(argv, argv + argc, "--gauge"))
        gaugeFile = GridCmdOptionPayload(argv, argv + argc, "--gauge");
    if (GridCmdOptionExists(argv, argv + argc, "--filestem"))
        filestem = GridCmdOptionPayload(argv, argv + argc, "--filestem");
    if (GridCmdOptionExists(argv, argv + argc, "--traj"))
        traj = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--traj"));
    if (GridCmdOptionExists(argv, argv + argc, "--nCheck"))
        nCheck = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nCheck"));
    if (GridCmdOptionExists(argv, argv + argc, "--resid"))
        resid = std::stod(GridCmdOptionPayload(argv, argv + argc, "--resid"));
    if (GridCmdOptionExists(argv, argv + argc, "--ens"))
        ens = GridCmdOptionPayload(argv, argv + argc, "--ens");

    if (gaugeFile.empty() || filestem.empty() || (ens != "64I" && ens != "48I"))
    {
        std::cerr << "Usage: " << argv[0]
                  << " --grid X.Y.Z.T --mpi x.y.z.t"
                  << " --gauge <config> --filestem <converter filestem, no _fine/_coarse suffix>"
                  << " [--traj N] [--nCheck N] [--resid X] [--ens 64I|48I]" << std::endl;
        exit(EXIT_FAILURE);
    }

    Grid_init(&argc, &argv);
    GridLogIRL.Active(true); // per-vector residual line comes from here

    // ------------------------------------------------------------------
    // Ensemble presets: action parameters, eigenpack geometry (nBasis is a
    // compile-time template parameter below, so it's fixed per branch, not
    // a variable here), and Schur convention. 64I: standard Mobius, matches
    // par.CGl.1500.xml's mdwf_l/mdwff_l exactly, GPT's schur_complement_one
    // -> SchurDiagOneOperator. 48I: z-Mobius, matches the production
    // zmdwf_l block (14 complex omega coefficients, one per Ls slice),
    // Hadrons' default -> SchurDiagTwoOperator.
    // ------------------------------------------------------------------
    RealD                 mass, M5, b, c;
    unsigned int          Ls;
    std::string           schurConv;
    AcceleratorVector<Complex, Nd> boundary;
    AcceleratorVector<Real, Nd>    twist;
    std::vector<ComplexD> omega; // only used for ens == 48I
    std::vector<int>      blockSize4d;
    int                   blockLs;

    if (ens == "48I")
    {
        Ls        = 14;
        mass      = 0.00078;
        M5        = 1.8;
        b         = 1.0;
        c         = 0.0;
        boundary  = std::vector<Complex>{1., 1., 1., 1.};
        twist     = std::vector<Real>{0., 0., 0., 0.};
        schurConv = "diagtwo";
        omega     = {
            ComplexD(1.4789834351796358e+00, -0.000000000000000e+00),
            ComplexD(1.347049274947458e+00,  -0.000000000000000e+00),
            ComplexD(1.1273467425761714e+00, -0.000000000000000e+00),
            ComplexD(8.91777252638092e-01,   -0.000000000000000e+00),
            ComplexD(6.798157283073448e-01,  -0.000000000000000e+00),
            ComplexD(5.06488728896523e-01,   -0.000000000000000e+00),
            ComplexD(3.71660686773301e-01,   -0.000000000000000e+00),
            ComplexD(2.6925807172929905e-01, -0.000000000000000e+00),
            ComplexD(1.9187135204541664e-01, -0.000000000000000e+00),
            ComplexD(1.516448422854592e-01,  -0.000000000000000e+00),
            ComplexD(1.3000288981245012e-01,  4.732386585347784e-02),
            ComplexD(1.3000288981245012e-01, -4.732386585347784e-02),
            ComplexD(8.653188911012571e-02,   1.0514692271165546e-01),
            ComplexD(8.653188911012571e-02,  -1.0514692271165546e-01),
        };
        // nBasis (sizeFine) = 400, sizeCoarse = 2000.
        blockSize4d = {4, 3, 3, 3};
        blockLs     = 14;
    }
    else if (ens == "64I")
    {
        Ls          = 12;
        mass        = 0.000678;
        M5          = 1.8;
        b           = 1.5;
        c           = 0.5;
        boundary    = std::vector<Complex>{1., 1., 1., -1.};
        twist       = std::vector<Real>{0., 0., 0., 0.};
        schurConv   = "diagone";
        // nBasis (sizeFine) = 200, sizeCoarse = 2000.
        blockSize4d = {4, 4, 4, 4};
        blockLs     = 12;
    }
    else
    {
        std::cout << "Incorrect ensemble param. This test only accepts 48I and 64I." << std::endl;
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------
    // Grids, straight off --grid/--mpi. Gauge is read in double precision
    // (matches NerscIO/production), then precision-cast down -- everything
    // downstream (action, eigenpack, operator) is single precision,
    // matching production's mdwff_l/vec_fine. Gauge field type/grids are
    // the same real-valued ones for both ens presets -- z-Mobius only
    // changes the fermion action/field type, not the gauge representation.
    //
    // The coarse grid is built directly here (not through Hadrons'
    // Environment) using the same block-decomposition arithmetic
    // Environment::createCoarseGrid uses; its SIMD/MPI layout doesn't need
    // to match whatever Hadrons picked when the file was written, only the
    // global coarse dimensions matter (see top-of-file comment).
    // ------------------------------------------------------------------
    GridCartesian          *UGrid    = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());

    GridCartesian          *UGridF   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian   *UrbGridF = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);
    GridCartesian           *FGridF   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGridF);
    GridRedBlackCartesian   *FrbGridF = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGridF);

    Coordinate fineDim = GridDefaultLatt();
    Coordinate coarseDim(4);
    for (int d = 0; d < 4; ++d)
    {
        coarseDim[d] = fineDim[d] / blockSize4d[d];
    }
    int cLs = Ls / blockLs;

    GridCartesian *CGrid4dF = SpaceTimeGrid::makeFourDimGrid(coarseDim, GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    GridCartesian *CGrid5dF = SpaceTimeGrid::makeFiveDimGrid(cLs, CGrid4dF);

    // ------------------------------------------------------------------
    // Real production gauge config, single precision (matches mdwff_l).
    // ------------------------------------------------------------------
    LatticeGaugeFieldD Umu(UGrid);
    LatticeGaugeFieldF UmuF(UGridF);
    FieldMetaData       header;
    NerscIO::readConfiguration(Umu, header, gaugeFile);
    precisionChange(UmuF, Umu);

    // ------------------------------------------------------------------
    // Full fine basis (all sizeFine vectors) + first nCheck coarse
    // eigenvectors (lowest/most physically relevant).
    // ------------------------------------------------------------------
    unsigned int nFail;

    if (ens == "48I")
    {
        typedef typename ZMobiusFermionF::FermionField LatticeFermionZF;

        typename ZMobiusFermionF::ImplParams implParams;
        implParams.boundary_phases = boundary;
        implParams.twist_n_2pi_L   = twist;
        ZMobiusFermionF Ddwf(UmuF, *FGridF, *FrbGridF, *UGridF, *UrbGridF,
                             mass, M5, omega, b, c, implParams);

        Hadrons::CoarseFermionEigenPack<ZFIMPLF, 400> epack(400, nCheck, FrbGridF, CGrid5dF);
        epack.readFine(filestem, true, traj);
        epack.readCoarse(filestem, true, traj);

        if (schurConv == "diagone")
        {
            nFail = checkVectors<SchurDiagOneOperator<ZMobiusFermionF, LatticeFermionZF>>(
                Ddwf, epack.evec, epack.evecCoarse, epack.evalCoarse, nCheck, resid, "SchurDiagOneOperator");
        }
        else
        {
            nFail = checkVectors<SchurDiagTwoOperator<ZMobiusFermionF, LatticeFermionZF>>(
                Ddwf, epack.evec, epack.evecCoarse, epack.evalCoarse, nCheck, resid, "SchurDiagTwoOperator");
        }
    }
    else // 64I
    {
        typename MobiusFermionF::ImplParams implParams;
        implParams.boundary_phases = boundary;
        implParams.twist_n_2pi_L   = twist;
        MobiusFermionF Ddwf(UmuF, *FGridF, *FrbGridF, *UGridF, *UrbGridF,
                            mass, M5, b, c, implParams);

        Hadrons::CoarseFermionEigenPack<FIMPLF, 200> epack(200, nCheck, FrbGridF, CGrid5dF);
        epack.readFine(filestem, true, traj);
        epack.readCoarse(filestem, true, traj);

        if (schurConv == "diagone")
        {
            nFail = checkVectors<SchurDiagOneOperator<MobiusFermionF, LatticeFermionF>>(
                Ddwf, epack.evec, epack.evecCoarse, epack.evalCoarse, nCheck, resid, "SchurDiagOneOperator");
        }
        else
        {
            nFail = checkVectors<SchurDiagTwoOperator<MobiusFermionF, LatticeFermionF>>(
                Ddwf, epack.evec, epack.evecCoarse, epack.evalCoarse, nCheck, resid, "SchurDiagTwoOperator");
        }
    }

    Grid_finalize();

    return (nFail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
