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

