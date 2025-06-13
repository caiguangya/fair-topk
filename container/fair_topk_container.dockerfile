FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade

RUN apt-get install -y build-essential g++ clang cmake vim tar wget unzip git libboost-dev xtensor-dev libgmp-dev libmpfr-dev

RUN unset DEBIAN_FRONTEND

RUN wget -v https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz && \
    tar -xvf gurobi11.0.3_linux64.tar.gz && \
    rm -f gurobi11.0.3_linux64.tar.gz && \
    cd gurobi1103/linux64/src/build && \
    sed -i 's/-O/-O3 -DNDEBUG/g' Makefile && \
    make && \
    rm -f *.o && \
    cd ../../../../ && \
    mkdir -p /usr/local/lib/gurobi /usr/local/include/gurobi && \
    cp -f gurobi1103/linux64/lib/libgurobi.so.11.0.3 /usr/local/lib/gurobi/libgurobi110.so && \
    cp -f gurobi1103/linux64/src/build/libgurobi_c++.a /usr/local/lib/gurobi && \
    mv -f gurobi1103/linux64/include/*.h /usr/local/include/gurobi && \
    mv -f gurobi1103 /opt/gurobi

RUN git clone --single-branch -b 3.4.0 https://gitlab.com/libeigen/eigen.git eigen && \
    cd eigen && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make install && \
    cd ../../ && \
    rm -rf eigen

RUN git clone https://github.com/khizmax/libcds.git libcds && \
    cd libcds && \
    git reset --hard 9985d2a87feaa3e92532e28f4ab762a82855a49c && \
    cmake -DCMAKE_BUILD_TYPE=Release . && \
    make install && \
    cd ../ && \
    rm -rf libcds

RUN git clone --single-branch -b v2.5.0 https://github.com/CLIUtils/CLI11.git CLI11 && \
    cd CLI11 && \
    cmake -DCMAKE_BUILD_TYPE=Release . && \
    make install -j && \
    cd ../ && \
    rm -rf CLI11

ENV LC_ALL=C
ENV LD_LIBRARY_PATH=/usr/local/lib64
