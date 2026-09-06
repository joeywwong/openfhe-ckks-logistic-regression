# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION} AS openfhe-build

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENFHE_REVISION=b2869aef5cf61afd364b3eaea748dcc8a7020b9c
ARG BUILD_JOBS=2

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/openfhe
RUN git init \
    && git remote add origin https://github.com/openfheorg/openfhe-development.git \
    && git fetch --depth 1 origin "${OPENFHE_REVISION}" \
    && git checkout --detach FETCH_HEAD \
    && git submodule update --init --depth 1 third-party/cereal

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_UNITTESTS=OFF \
        -DGIT_SUBMOD_AUTO=OFF \
    && cmake --build build --parallel "${BUILD_JOBS}" \
    && cmake --install build

FROM openfhe-build AS project-build

ARG BUILD_JOBS=2

WORKDIR /opt/openfhe-lab
COPY . .

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpenFHE_DIR=/usr/local/lib/OpenFHE \
    && cmake --build build --parallel "${BUILD_JOBS}"

FROM project-build AS test

RUN ctest --test-dir build --output-on-failure

FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=project-build /usr/local/lib/ /usr/local/lib/
COPY --from=project-build /opt/openfhe-lab/build/openfhe_lab_compare /usr/local/bin/openfhe_lab_compare
COPY --from=project-build --chown=10001:10001 /opt/openfhe-lab/data/ /opt/openfhe-lab/data/
COPY --from=project-build --chown=10001:10001 /opt/openfhe-lab/LICENSE /opt/openfhe-lab/THIRD_PARTY_NOTICES.md /opt/openfhe-lab/

RUN ldconfig \
    && mkdir -p /opt/openfhe-lab/results \
    && chown 10001:10001 /opt/openfhe-lab/results

WORKDIR /opt/openfhe-lab
USER 10001:10001

ENTRYPOINT ["openfhe_lab_compare"]
CMD ["--help"]
