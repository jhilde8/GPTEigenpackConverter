/*
 * Test_converted_verify.cpp
 *
 * Standalone data-integrity check for the fine (local coherence basis)
 * output of GPTEigenpackConverter. Intended to run right after the
 * converter finishes, on the same CPU allocation/build (e.g. Andes), as a
 * quick pass/fail before committing to a full production a2a solve.
 *
 * Reads a handful of fine basis vectors straight off disk via
 * Hadrons::FermionEigenPack<FIMPLF> (the exact reader
 * CoarseEigenPack::readFine() uses, since the converter writes to
 * "<filestem>_fine.<traj>" in that same format) -- no coarse grid, no
 * coarse coefficients, no CoarseFermionEigenPack involved. Grid geometry is
 * built directly from --grid/--mpi like any plain Grid program; Grid's
 * parallel I/O is layout-independent (lexicographic on disk), so this
 * doesn't need to match the SIMD decomposition the converter happened to
 * use -- only the global lattice dimensions matter, and those come from
 * the same --grid argument the converter was run with. Confirmed the
 * write side goes through the same general-purpose Grid ScidacWriter path
 * (GridEigenpackReader.hpp's WriteMode::HADRONS branch calls
 * eigenpack.writeFine(), i.e. Hadrons::EigenPack::write() ->
 * EigenPackIo::writePack()), so there is no reader-side local-volume
 * constraint beyond passing the correct global --grid dimensions -- that
 * constraint (matching MPI ranks to GPT's file split) is specific to the
 * converter itself, not to reading its output back.
 *
 * Checks that each loaded vector is an approximate eigenvector of
 * Mpc^dag Mpc, exactly as LocalCoherenceLanczos::testFine() checks right
 * after a real Lanczos run (Grid/algorithms/iterative/LocalCoherenceLanczos.h:
 * 347-356). Reuses the same ImplicitlyRestartedLanczosHermOpTester, so the
 * residual math matches production's own Lanczos convergence test bit for
 * bit. --schur selects which Schur convention Mpc is built from
 * (SchurDiagTwoOperator, Hadrons' default -- matches par.CGl.*.xml's
 * RBPrecCG/ExactDeflation as of this writing -- or SchurDiagOneOperator,
 * matching GPT's schur_complement_one); default is diagtwo.
 *
 * --ens selects which ensemble's action parameters to build Mpc from:
 *   - 64i (default): standard Mobius, mass/M5/b/c/Ls/boundary/twist taken
 *     from par.CGl.1500.xml's mdwf_l/mdwff_l (b=1.5, c=0.5, Ls=12).
 *   - 48i: z-Mobius (MAction::ZMobiusDWF in production), mass/M5/b/c/omega/
 *     Ls/boundary/twist taken from the 48I zmdwf_l XML block (b=1.0, c=0.0,
 *     Ls=14, 14 complex omega coefficients). Used as a control: 48I is a
 *     working, already-validated production ensemble, so if this same test
 *     harness passes cleanly on 48I's converted vectors, that isolates the
 *     64I failure to something about that specific converted pack (or its
 *     conversion) rather than a bug in this test itself.
 * --gauge/--filestem/--traj/--grid/--mpi/--Ls remain independent of --ens
 * (Ls defaults to the ensemble's own value -- 12 for 64i, 14 for 48i -- but
 * --Ls still overrides either). Note b=1.0,c=0.0 on 48I (b+c=1) does not
 * match the commonly-quoted "b+c=2" standard some 64I documentation also
 * cites for b=1.5,c=0.5 -- i.e. the documented "standard" b+c does not
 * necessarily match what a given ensemble's Lanczos was actually run with,
 * which is why b/c are bundled into the --ens preset rather than assumed.
 *
 * Motivation: production a2a solves show a very bad deflation guess -- CG
 * barely moves off its undeflated trajectory in 400 inner iterations. The
 * guesser wiring (EigenPackLCDecompress -> ExactDeflation) was checked and
 * matches production, action parameters/Ls/boundary/twist were checked
 * against par.CGl.1500.xml and match exactly, the gauge field/trajectory
 * were confirmed correct, and DiagOne vs DiagTwo only changes the
 * reconstructed eigenvalue by a fraction of a percent (not the 1e5-1e6
 * relative blowup actually observed) -- so the remaining suspects are
 * either the unresolved zip-vs-non-zip byte disagreement seen partway
 * through v0.bin, or something specific to the converted 64I data itself,
 * which the 48I control run is meant to help isolate.
 *
 * Usage:
 *   ./Test_converted_verify --grid 64.64.64.128 --mpi 4.4.4.4 \
 *       --gauge /path/to/ckpoint_lat \
 *       --filestem /path/to/converted/vec \
 *       --traj 1500 --nCheck 5 --resid 1e-3 --schur diagtwo --ens 64i
 *
 * A vector that's intact should reconstruct an eigenvalue close to the
 * stored one and pass the residual target; loosen/tighten --resid to
 * whatever tolerance the fine Lanczos was actually run at. An O(1) residual
 * on a vector regardless of --resid means that vector simply isn't an
 * eigenvector of this operator (corrupt read, index/ordering mismatch).
 */
#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedLanczos.h>
#include <Hadrons/EigenPack.hpp>

using namespace Grid;
using namespace Hadrons;

// Runs the Mpc^dag Mpc residual check for a given Schur convention (SchurOp
// is SchurDiagOneOperator<FMat, Field> or SchurDiagTwoOperator<FMat, Field>,
// fully specified by the caller). FMat/Field are deduced from Ddwf/vecs, so
// this works unchanged for both the 64I (Mobius) and 48I (z-Mobius) action
// types. Returns the number of vectors that failed.
template <typename SchurOp, typename FMat, typename Field>
unsigned int checkVectors(FMat &Ddwf, std::vector<Field> &vecs, std::vector<RealD> &evals,
                          unsigned int nCheck, double resid, const std::string &label)
{
    SchurOp                                      schurOp(Ddwf);
    PlainHermOp<Field>                           hermOp(schurOp);
    ImplicitlyRestartedLanczosHermOpTester<Field> tester(hermOp);
    unsigned int                                 nFail = 0;

    std::cout << GridLogMessage << "Checking " << nCheck
              << " fine basis vector(s) against Mpc^dag Mpc (" << label << "), "
              << "target residual " << resid << std::endl;

    for (unsigned int k = 0; k < nCheck; ++k)
    {
        RealD evalStored = evals[k];
        RealD evalRecon  = evalStored;
        int   conv       = tester.TestConvergence(k, resid, vecs[k], evalRecon, 1.0);
        RealD relDiff    = (evalStored != 0.)
                          ? std::abs(evalRecon - evalStored) / std::abs(evalStored)
                          : std::abs(evalRecon - evalStored);

        std::cout << GridLogMessage << "  evec " << k
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
              << " fine basis vectors FAILED the residual check (" << label << ")" << std::endl;

    return nFail;
}

int main(int argc, char *argv[])
{
    std::string  gaugeFile = "";
    std::string  filestem  = "";
    int          traj      = -1;
    unsigned int nCheck    = 5;
    double       resid     = 1e-3;
    unsigned int Ls        = 0; // 0 = "use the --ens default", set below
    std::string  schurConv = "diagtwo";
    std::string  ens       = "64i";

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
    if (GridCmdOptionExists(argv, argv + argc, "--schur"))
        schurConv = GridCmdOptionPayload(argv, argv + argc, "--schur");
    if (GridCmdOptionExists(argv, argv + argc, "--ens"))
        ens = GridCmdOptionPayload(argv, argv + argc, "--ens");
    if (GridCmdOptionExists(argv, argv + argc, "--Ls"))
        Ls = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--Ls"));

    if (gaugeFile.empty() || filestem.empty()
        || (schurConv != "diagone" && schurConv != "diagtwo")
        || (ens != "64i" && ens != "48i"))
    {
        std::cerr << "Usage: " << argv[0]
                  << " --grid X.Y.Z.T --mpi x.y.z.t"
                  << " --gauge <config> --filestem <converter filestem, no _fine suffix>"
                  << " [--traj N] [--nCheck N] [--resid X] [--Ls N]"
                  << " [--schur diagone|diagtwo] [--ens 64i|48i]" << std::endl;
        exit(EXIT_FAILURE);
    }

    Grid_init(&argc, &argv);
    GridLogIRL.Active(true); // per-vector residual line comes from here

    // ------------------------------------------------------------------
    // Ensemble action presets. 64I: standard Mobius, matches par.CGl.1500.
    // xml's mdwf_l/mdwff_l exactly. 48I: z-Mobius, matches the production
    // zmdwf_l block (14 complex omega coefficients, one per Ls slice).
    // ------------------------------------------------------------------
    RealD              mass, M5, b, c;
    std::vector<Real>  boundary, twist;
    std::vector<ComplexD> omega; // only used for ens == 48i

    if (ens == "48i")
    {
        if (Ls == 0) Ls = 14;
        mass     = 0.00078;
        M5       = 1.8;
        b        = 1.0;
        c        = 0.0;
        boundary = {1., 1., 1., 1.};
        twist    = {0., 0., 0., 0.};
        omega    = {
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
        // Remaining 48I eigenpack info (not yet used by this fine-only
        // check, recorded here for when the coarse-vector check is added):
        // sizeFine=400, sizeCoarse=2000, blockSize=4.3.3.3.14.
    }
    else // 64i
    {
        if (Ls == 0) Ls = 12;
        mass     = 0.000678;
        M5       = 1.8;
        b        = 1.5;
        c        = 0.5;
        boundary = {1., 1., 1., -1.};
        twist    = {0., 0., 0., 0.};
    }

    // ------------------------------------------------------------------
    // Grids, straight off --grid/--mpi. Gauge is read in double precision
    // (matches NerscIO/production), then precision-cast down -- everything
    // downstream (action, eigenpack, operator) is single precision,
    // matching production's mdwff_l/vec_fine. Gauge field type/grids are
    // the same real-valued ones for both ens presets -- z-Mobius only
    // changes the fermion action/field type, not the gauge representation.
    // ------------------------------------------------------------------
    GridCartesian          *UGrid    = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());

    GridCartesian          *UGridF   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian   *UrbGridF = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);
    GridCartesian           *FGridF   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGridF);
    GridRedBlackCartesian   *FrbGridF = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGridF);

    // ------------------------------------------------------------------
    // Real production gauge config, single precision (matches mdwff_l).
    // ------------------------------------------------------------------
    LatticeGaugeFieldD Umu(UGrid);
    LatticeGaugeFieldF UmuF(UGridF);
    FieldMetaData       header;
    NerscIO::readConfiguration(Umu, header, gaugeFile);
    precisionChange(UmuF, Umu);

    // ------------------------------------------------------------------
    // Fine basis vectors ONLY -- points straight at "<filestem>_fine", the
    // exact file CoarseEigenPack::readFine() would read. Loads only the
    // first nCheck vectors (v0.bin upward) -- v0.bin is where the byte
    // disagreement was found.
    // ------------------------------------------------------------------
    unsigned int nFail;

    if (ens == "48i")
    {
        typedef typename ZMobiusFermionF::FermionField LatticeFermionZF;

        typename ZMobiusFermionF::ImplParams implParams;
        implParams.boundary_phases = boundary;
        implParams.twist_n_2pi_L   = twist;
        ZMobiusFermionF Ddwf(UmuF, *FGridF, *FrbGridF, *UGridF, *UrbGridF,
                             mass, M5, omega, b, c, implParams);

        Hadrons::FermionEigenPack<ZFIMPLF> epack(nCheck, FrbGridF);
        epack.read(filestem + "_fine", true, traj);

        if (schurConv == "diagone")
        {
            nFail = checkVectors<SchurDiagOneOperator<ZMobiusFermionF, LatticeFermionZF>>(
                Ddwf, epack.evec, epack.eval, nCheck, resid, "SchurDiagOneOperator");
        }
        else
        {
            nFail = checkVectors<SchurDiagTwoOperator<ZMobiusFermionF, LatticeFermionZF>>(
                Ddwf, epack.evec, epack.eval, nCheck, resid, "SchurDiagTwoOperator");
        }
    }
    else // 64i
    {
        typename MobiusFermionF::ImplParams implParams;
        implParams.boundary_phases = boundary;
        implParams.twist_n_2pi_L   = twist;
        MobiusFermionF Ddwf(UmuF, *FGridF, *FrbGridF, *UGridF, *UrbGridF,
                            mass, M5, b, c, implParams);

        Hadrons::FermionEigenPack<FIMPLF> epack(nCheck, FrbGridF);
        epack.read(filestem + "_fine", true, traj);

        if (schurConv == "diagone")
        {
            nFail = checkVectors<SchurDiagOneOperator<MobiusFermionF, LatticeFermionF>>(
                Ddwf, epack.evec, epack.eval, nCheck, resid, "SchurDiagOneOperator");
        }
        else
        {
            nFail = checkVectors<SchurDiagTwoOperator<MobiusFermionF, LatticeFermionF>>(
                Ddwf, epack.evec, epack.eval, nCheck, resid, "SchurDiagTwoOperator");
        }
    }

    Grid_finalize();

    return (nFail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
