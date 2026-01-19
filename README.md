# GPTEigenpackConverter

This is a public upload of a code to convert Local Coherence eigenpacks from the [GPT](https://github.com/lehner/gpt) format to a Scidac format readable by [Grid](https://github.com/paboyle/Grid) and [Hadrons](https://github.com/aportelli/Hadrons.git).

The GPT format contains both FP32 data and bitpacked float data. The FP32 data is bit-for-bit preserved, but the bitpacked data is not supported in the target format and is decompressed during conversion. The converted eigenpack therefore entirely comprises FP32 data.

This code was written a considerable amount of time before the public release, and although functional, the source is quite poorly written. This is one reason for a highly belated release. Given that it is unclear when an opportunity to clean up (and re-validate) the codebase may arise, it is now made available due to its potential utility.

### Dependencies
CPU builds of:
- [Grid](https://github.com/paboyle/Grid)
- [Hadrons](https://github.com/aportelli/Hadrons.git)

### Compilation
The source includes standard autotools build scripts and can be compiled as follows:
```bash
./bootstrap.sh.
mkdir build; cd build
../configure
--prefix=<path/to/install/programme>
--with-grid=<path/to/grid>
--with-hadrons=<path/to/hadrons>
CC=... # Whichever compiler was used for Grid & Hadrons
CXX=... # Whichever compiler was used for Grid & Hadrons
nbasis=250
# Number of basis (fine) vectors. Must exactly equal the number of basis vectors used by the eigenpack.
concbasis=2
# Number of basis (fine) vectors to convert per batch. Low numbers harm performance, but use less memory.
conceigen=100
# Number of (coarse) eigenvectors to convert per batch. Low numbers harm performance, but use less memory.
```

In the above, `nbasis`, `concbasis`, and `conceigen` are mandatory configuration arguments.
- `nbasis` must match the number of basis vectors in the GPT eigenpack, meaning that the programme must be recompiled for eigenpacks with different basis counts.
- `concbasis` is the number of basis vectors that will be read from disk and converted at once. Increasing this number will reduce the overall runtime, but requires space for a larger memory buffer.
- `conceigen` is the number of coarse eigenvectors that will be read from disk and converted at once. Increasing this number will reduce the overall runtime, but requires space for a larger memory buffer.

This can then be built and installed with `make`

```bash
make
make install
```

### Running

The converter can be ran as follows:
```bash
path/to/GPTEigenpackConverter
# Fine lattice size
--grid 64.64.64.128
# MPI layout for the conversion
--mpi 1.1.2.2
# Lattice Ls extent
--Ls 12
# Configuration number
--config 1000
# Path to GPT eigenpack (containing metadata.txt)
--i /path/to/input
# Directory to output Scidac eigenpack to
--o /path/to/output
# Which format to output data in ('none' or 'hadrons')
--outmode <fmt>
```
It is recommended to do this on CPUs to more easily use large numbers of MPI ranks. This is because each MPI process must not write more than 2 GiB to disk to prevent integer overflows, and you must therefore use at least (eigenvector size)/2 MPI ranks to convert. This could be alleviated with future work.

### Acknowledging

If this code has enabled scientific work that has led to a peer-reviewed publication, it would be appreciated if the role played by this code could be acknowledged with a statement in the manuscript.
