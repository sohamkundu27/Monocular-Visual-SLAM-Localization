#!/usr/bin/env bash
# Build the C++ dependencies of the monocular SLAM port from source into a
# user-local prefix, without root.
#
# Versions are chosen to match what the Python reference implementation runs
# against, so the two implementations can be compared for numerical parity:
#
#   OpenCV   4.13.0   == cv2.__version__ of the opencv-python 4.13.0.92 wheel
#   OpenBLAS 0.3.15   == the OpenBLAS bundled inside that wheel (cv2 uses it as
#                        its LAPACK HAL for SVD / GEMM on large matrices)
#   GTSAM    4.2.2    == the gtsam 4.2.2 wheel (the Python pin is >=4.2,<4.3)
#   Boost    1.87.0   == the Boost bundled inside that gtsam wheel
#   Eigen    3.4.0    (GTSAM is built with GTSAM_USE_SYSTEM_EIGEN=ON against it)
#
# OpenCV is configured like the wheel where it can affect numerics: IPP ICV on,
# LAPACK via OpenBLAS on, Eigen OFF (cv2 reports "Eigen: NO", and with Eigen
# the USAC solvers take different code paths), pthreads parallel backend,
# default SSE3 baseline with runtime dispatch, no fast-math.
#
# Usage:
#   tools/build_deps.sh [PREFIX]          # default PREFIX=$HOME/opt/mvslam-deps
#   FC=/path/to/gfortran tools/build_deps.sh
#
# Afterwards configure the project with
#   cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=$PREFIX/install
set -euo pipefail

ROOT="${1:-$HOME/opt/mvslam-deps}"
SRC="$ROOT/src"
BLD="$ROOT/build"
PREFIX="$ROOT/install"
JOBS="${JOBS:-$(nproc)}"
FC="${FC:-gfortran}"
STEPS="${STEPS:-eigen openblas boost opencv gtsam}"

EIGEN_VER=3.4.0
OPENBLAS_VER=0.3.15
BOOST_VER=1.87.0
OPENCV_VER=4.13.0
GTSAM_VER=4.2.2

mkdir -p "$SRC" "$BLD" "$PREFIX"

fetch() {  # fetch <url> <archive-name> <extracted-dir>
    local url="$1" archive="$SRC/$2" dir="$SRC/$3"
    if [[ ! -d "$dir" ]]; then
        [[ -f "$archive" ]] || curl -fL --retry 3 -o "$archive" "$url"
        tar -xf "$archive" -C "$SRC"
    fi
}

build_eigen() {
    fetch "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VER/eigen-$EIGEN_VER.tar.gz" \
        "eigen-$EIGEN_VER.tar.gz" "eigen-$EIGEN_VER"
    cmake -S "$SRC/eigen-$EIGEN_VER" -B "$BLD/eigen" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
    cmake --build "$BLD/eigen" --target install
}

build_openblas() {
    fetch "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$OPENBLAS_VER/OpenBLAS-$OPENBLAS_VER.tar.gz" \
        "OpenBLAS-$OPENBLAS_VER.tar.gz" "OpenBLAS-$OPENBLAS_VER"
    # Same build flavour as the wheel's "OpenBLAS 0.3.15 DYNAMIC_ARCH NO_AFFINITY":
    # TARGET only sets the build-time baseline (0.3.15 cannot auto-detect recent CPUs);
    # kernels are selected at runtime for the host CPU, pthreads, no OpenMP.
    # The Fortran runtime may live outside the default linker path (e.g. an
    # unprivileged extraction of the gfortran packages), so point ld at it.
    local gfdir
    gfdir="$(dirname "$("$FC" -print-file-name=libgfortran.so)")"
    make -C "$SRC/OpenBLAS-$OPENBLAS_VER" -j"$JOBS" LDFLAGS="-L$gfdir" \
        DYNAMIC_ARCH=1 TARGET=PRESCOTT NO_AFFINITY=1 USE_OPENMP=0 USE_THREAD=1 NUM_THREADS=64 \
        FC="$FC" NO_STATIC=1 libs netlib shared
    make -C "$SRC/OpenBLAS-$OPENBLAS_VER" PREFIX="$PREFIX" LDFLAGS="-L$gfdir" FC="$FC" \
        DYNAMIC_ARCH=1 TARGET=PRESCOTT NO_AFFINITY=1 USE_OPENMP=0 NO_STATIC=1 install
    # The Fortran runtime must be findable next to libopenblas at runtime.
    local libgf
    libgf="$(readlink -f "$("$FC" -print-file-name=libgfortran.so)")"
    if [[ "$libgf" == /* && -f "$libgf" ]]; then
        [[ -e /lib/x86_64-linux-gnu/libgfortran.so.5 ]] || cp -L "$libgf" "$PREFIX/lib/libgfortran.so.5"
    fi
}

build_boost() {
    local us="${BOOST_VER//./_}"
    fetch "https://archives.boost.io/release/$BOOST_VER/source/boost_$us.tar.gz" \
        "boost_$us.tar.gz" "boost_$us"
    pushd "$SRC/boost_$us" >/dev/null
    ./bootstrap.sh --prefix="$PREFIX" \
        --with-libraries=serialization,system,filesystem,thread,program_options,date_time,timer,chrono,regex,atomic
    ./b2 -j"$JOBS" --build-dir="$BLD/boost" variant=release link=shared threading=multi \
        cxxflags=-fPIC install
    popd >/dev/null
}

build_opencv() {
    fetch "https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VER.tar.gz" \
        "opencv-$OPENCV_VER.tar.gz" "opencv-$OPENCV_VER"
    cmake -S "$SRC/opencv-$OPENCV_VER" -B "$BLD/opencv" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DBUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,flann \
        -DBUILD_SHARED_LIBS=ON -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
        -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_DOCS=OFF \
        -DWITH_IPP=ON -DWITH_EIGEN=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_PTHREADS_PF=ON \
        -DWITH_LAPACK=ON -DLAPACK_IMPL=OpenBLAS \
        -DLAPACK_LIBRARIES="$PREFIX/lib/libopenblas.so" -DLAPACK_INCLUDE_DIR="$PREFIX/include" \
        -DLAPACK_CBLAS_H=cblas.h -DLAPACK_LAPACKE_H=lapacke.h \
        -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON \
        -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF \
        -DWITH_AVIF=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_GTK=OFF \
        -DWITH_QT=OFF -DWITH_VTK=OFF -DWITH_PROTOBUF=OFF -DWITH_CUDA=OFF \
        -DENABLE_FAST_MATH=OFF -DCPU_BASELINE=SSE3 \
        -DCMAKE_INSTALL_RPATH="$PREFIX/lib"
    cmake --build "$BLD/opencv" -j"$JOBS" --target install
}

build_gtsam() {
    fetch "https://github.com/borglab/gtsam/archive/refs/tags/$GTSAM_VER.tar.gz" \
        "gtsam-$GTSAM_VER.tar.gz" "gtsam-$GTSAM_VER"
    cmake -S "$SRC/gtsam-$GTSAM_VER" -B "$BLD/gtsam" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DBOOST_ROOT="$PREFIX" -DBoost_NO_SYSTEM_PATHS=ON \
        -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_BUILD_TIMING_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_BUILD_DOCS=OFF \
        -DGTSAM_USE_SYSTEM_EIGEN=ON -DEigen3_DIR="$PREFIX/share/eigen3/cmake" \
        -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
        -DCMAKE_INSTALL_RPATH="$PREFIX/lib"
    cmake --build "$BLD/gtsam" -j"$JOBS" --target install
}

for step in $STEPS; do
    echo "==> building $step"
    "build_$step"
done
echo "Dependencies installed under $PREFIX"
