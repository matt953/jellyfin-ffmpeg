#!/bin/bash

SCRIPT_REPO="https://github.com/matt953/edge264.git"
SCRIPT_COMMIT="0b7b7375f043a614162c7e2ff362d5c6a5342029"

ffbuild_enabled() {
    [[ $TARGET == mac* ]] && return 0
    return 1
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" edge264
    cd edge264

    make clean || true

    # Build object file (with logs variant for runtime intrinsics selection)
    make edge264.o edge264_headers_log.o VARIANTS=logs BUILD_TEST=no

    # Create static library from object files
    ar rcs libedge264.a edge264.o edge264_headers_log.o
    ranlib libedge264.a

    mkdir -p "$FFBUILD_PREFIX/lib"
    mkdir -p "$FFBUILD_PREFIX/include"
    cp libedge264.a "$FFBUILD_PREFIX/lib/"
    cp edge264.h "$FFBUILD_PREFIX/include/"
}

ffbuild_configure() {
    echo --enable-libedge264
}

ffbuild_unconfigure() {
    echo --disable-libedge264
}

ffbuild_cflags() {
    echo "-I$FFBUILD_PREFIX/include"
}

ffbuild_ldflags() {
    echo "-L$FFBUILD_PREFIX/lib"
}

ffbuild_libs() {
    echo -ledge264
}
