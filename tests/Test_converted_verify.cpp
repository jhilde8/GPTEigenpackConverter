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
 * the same --grid argument the converter was run with.
 *
 * Checks that each loaded vector is an approximate eigenvector of
 * Mpc^dag Mpc (SchurDiagTwoOperator, Hadrons' default Schur convention --
 * matches par.CGl.*.xml's RBPrecCG/ExactDeflation), exactly as
 * LocalCoherenceLanczos::testFine() checks right after a real Lanczos run
 * (Grid/algorithms/iterative/LocalCoherenceLanczos.h:347-356). Reuses the
 * same ImplicitlyRestartedLanczosHermOpTester, so the residual math matches
 * production's own Lanczos convergence test bit for bit.
 *
 * Motivation: production a2a solves show a very bad deflation guess -- CG
 * barely moves off its undeflated trajectory in 400 inner iterations. The
 * guesser wiring (EigenPackLCDecompress -> ExactDeflation) was checked and
 * matches production, so the remaining suspects are the converted data
 * itself: either the unresolved zip-vs-non-zip byte disagreement seen
 * partway through v0.bin, or a coefficient-ordering mismatch in the coarse
 * read (out of scope for this test -- fine vectors only).
 *
 * Usage:
 *   ./Test_converted_verify --grid 64.64.64.128 --mpi 4.4.4.4 \
 *       --gauge /path/to/ckpoint_lat \
 *       --filestem /path/to/converted/vec \
 *       --traj 1500 --nCheck 5 --resid 1e-3
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

int main(int argc, char *argv[])
{
    std::string  gaugeFile = "";
    std::string  filestem  = "";
    int          traj      = -1;
    unsigned int nCheck    = 5;
    double       resid     = 1e-3;
    unsigned int Ls        = 12;

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
    if (GridCmdOptionExists(argv, argv + argc, "--Ls"))
        Ls = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--Ls"));

    if (gaugeFile.empty() || filestem.empty())
    {
        std::cerr << "Usage: " << argv[0]
                  << " --grid X.Y.Z.T --mpi x.y.z.t"
                  << " --gauge <config> --filestem <converter filestem, no _fine suffix>"
                  << " [--traj N] [--nCheck N] [--resid X] [--Ls N]" << std::endl;
        exit(EXIT_FAILURE);
    }

    Grid_init(&argc, &argv);
    GridLogIRL.Active(true); // per-vector residual line comes from here

    const RealD mass = 0.000678;
    const RealD M5   = 1.8;
    const RealD b    = 1.5;
    const RealD c    = 0.5;

    // ------------------------------------------------------------------
    // Grids, straight off --grid/--mpi. Gauge is read in double precision
    // (matches NerscIO/production), then precision-cast down -- everything
    // downstream (action, eigenpack, operator) is single precision,
    // matching production's mdwff_l/vec_fine.
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
    // Single-precision Mobius DWF action, standard 64I parameters, matching
    // production's mdwff_l (par.CGl.1500.xml).
    // ------------------------------------------------------------------
    typename MobiusFermionF::ImplParams implParams;
    implParams.boundary_phases = {1., 1., 1., -1.};
    implParams.twist_n_2pi_L   = {0., 0., 0., 0.};
    MobiusFermionF Ddwf(UmuF, *FGridF, *FrbGridF, *UGridF, *UrbGridF, mass, M5, b, c, implParams);

    SchurDiagTwoOperator<MobiusFermionF, LatticeFermionF>   schurOp(Ddwf);
    PlainHermOp<LatticeFermionF>                            hermOp(schurOp);
    ImplicitlyRestartedLanczosHermOpTester<LatticeFermionF> tester(hermOp);

    // ------------------------------------------------------------------
    // Fine basis vectors ONLY -- points straight at "<filestem>_fine", the
    // exact file CoarseEigenPack::readFine() would read. Loads only the
    // first nCheck vectors (v0.bin upward) -- v0.bin is where the byte
    // disagreement was found.
    // ------------------------------------------------------------------
    Hadrons::FermionEigenPack<FIMPLF> epack(nCheck, FrbGridF);
    epack.read(filestem + "_fine", true, traj);

    unsigned int nFail = 0;

    std::cout << GridLogMessage << "Checking " << nCheck
              << " fine basis vector(s) against Mpc^dag Mpc (SchurDiagTwoOperator), "
              << "target residual " << resid << std::endl;

    for (unsigned int k = 0; k < nCheck; ++k)
    {
        RealD evalStored = epack.eval[k];
        RealD evalRecon  = evalStored;
        int   conv       = tester.TestConvergence(k, resid, epack.evec[k], evalRecon, 1.0);
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
              << " fine basis vectors FAILED the residual check" << std::endl;

    Grid_finalize();

    return (nFail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
