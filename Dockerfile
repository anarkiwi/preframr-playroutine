# preframr-playroutine: instrumented libsidplayfp + sidtrace + python tooling.
#
# Multi-stage so the expensive libsidplayfp compile is cached independently of
# the python test layer.

# Pinned libsidplayfp revision the instrumentation patch is generated against.
ARG LIBSIDPLAYFP_REF=47766e4cef3f835a3d17dac574f44831088010d4
# reSIDfp engine: an external dependency in current libsidplayfp. Provides the
# deterministic SID emulation (noise LFSR resets to a fixed 0x7fffff).
ARG LIBRESIDFP_REF=8498ac9470a10c4aada3916e2abfc44ca3d0f25d

#-----------------------------------------------------------------------------
# Stage 1: build instrumented libsidplayfp and the sidtrace tool.
#-----------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder
ARG LIBSIDPLAYFP_REF
ARG LIBRESIDFP_REF
ENV DEBIAN_FRONTEND=noninteractive
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
RUN apt-get update && apt-get install --no-install-recommends -yq \
        automake autoconf ca-certificates g++ git wget make pkg-config \
        libtool xa65 libgcrypt20-dev gettext && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Build the reSIDfp engine first so libsidplayfp's pkg-config check enables the
# residfp builder (the deterministic, high-quality SID emulation).
RUN git clone https://github.com/libsidplayfp/libresidfp && \
    cd libresidfp && \
    git checkout ${LIBRESIDFP_REF} && \
    autoreconf -ivf && ./configure && \
    make -j"$(nproc)" && make install && ldconfig

RUN git clone https://github.com/libsidplayfp/libsidplayfp && \
    cd libsidplayfp && \
    git checkout ${LIBSIDPLAYFP_REF} && \
    git submodule update --init --recursive

# Apply the instrumentation hooks, then build + install.
COPY patches/instrument.patch /src/instrument.patch
RUN cd /src/libsidplayfp && \
    git apply --whitespace=nowarn /src/instrument.patch && \
    autoreconf -ivf && \
    ./configure --without-exsid --without-usbsid && \
    make -j"$(nproc)" && make install && ldconfig && \
    test -f /usr/local/include/sidplayfp/builders/residfp.h

# Build the tracer against the installed lib (public headers) plus the
# in-tree instrument.h.
COPY app/sidtrace.cpp /src/app/sidtrace.cpp
RUN g++ -O2 -std=c++17 \
        -I/usr/local/include -I/src/libsidplayfp/src \
        /src/app/sidtrace.cpp \
        -o /usr/local/bin/sidtrace \
        -L/usr/local/lib -lsidplayfp && \
    ldconfig && \
    /usr/local/bin/sidtrace --help 2>&1 | head -1

#-----------------------------------------------------------------------------
# Stage 2: python tooling + tests on top of the runtime library.
#-----------------------------------------------------------------------------
FROM python:3.12-slim AS test
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -yq \
        libgcrypt20 libgomp1 ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Runtime library + tracer binary from the builder.
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/bin/sidtrace /usr/local/bin/sidtrace
RUN ldconfig

WORKDIR /work
COPY pyproject.toml /work/pyproject.toml
COPY preframr_playroutine /work/preframr_playroutine
COPY tests /work/tests
RUN pip install --no-cache-dir -e ".[test]"

# Sanity check at build time.
RUN sidtrace --help 2>&1 | head -1

ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]
