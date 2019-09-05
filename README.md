# Libretracker: Libre, Free and Open-Source Eyetracking based on TIMM method

![Sample output](libretracker.gif)

## Introduction

Libretracker optimizes the [Tristan Hume's original code](https://github.com/trishume/eyeLike) of eye tracking algorithm by Fabian Timm (see references).

## Building

Prerequisites for Fedora-like distros:

```
sudo dnf install git cmake opencv-devel ocl-icd-devel
```

Prerequisites for Debian-like distros:

```
sudo apt-get install git cmake libopencv-dev ocl-icd-opencl-dev
```

Special prerequisites:

 * Needs at least GNU C/C++/Fortran 8.3.0 for AVX512 code generation

Older Ubuntu distributions might use PPAs to access newer compilers:

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo add-apt-repository ppa:ubuntu-toolchain-r/ppa
sudo apt-get update
sudo apt-get install gcc-8 g++-8 build-essential
```

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 80
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-5 50
```

Compiling:

```console
git clone --recursive https://github.com/dmikushin/libretracker.git
cd libretracker
mkdir build
cd build
cmake ..
make -j4
```

## Deployment

This version of libretracker reads the input image series from `data/images/trial_0/raw` directory, and saves outputs to `data/images/trial_0/processed`:

```console
./libretracker
```

If a GPU is found, it is used by default. CPU version could be enforced explicitly:

```console
USE_CUDA=0 USE_OPENCL=0 ./libretracker
```

The CPU version is single-threaded by default, unless the `OMP_NUM_THREADS` is given:

```console
OMP_NUM_THREADS=4 USE_CUDA=0 USE_OPENCL=0 ./libretracker
```

Note libretracker's mulththreading backend is Intel TBB (not OpenMP), yet using the `OMP_NUM_THREADS` for unification.

# Performance results

The results below are obtained on the provided test dataset. The time values are per frame, averaged from the entire dataset.

Intel(R) Core(TM) i7-3610QM CPU @ 2.30GHz:

| TIMM version                | time, ms  |
| --------------------------- | --------- |
| basic code                  | 36.07     |
| AVX vectorization, 1 thread | 13.63     |
| AVX vectorization, 4 thread |  5.92     |
| NVIDIA GTX1060 GPU          |  3.48     |

## References

Timm and Barth. Accurate eye centre localisation by means of gradients. In Proceedings of the Int. Conference on Computer Theory and Applications (VISAPP), volume 1, pages 125-130, Algarve, Portugal, 2011. INSTICC.

