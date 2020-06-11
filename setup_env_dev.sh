#!/bin/bash

if [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
    # when the following lib files are ready, we think build ivf-hnsw done
    if [ ! -f ${PWD}/lib/libfaiss.a ]; then
        echo ivf-hnsw not build yet, build it first
        exit 1
    fi

    if [ ! -f ${PWD}/lib/libhnswlib.a ]; then
        echo ivf-hnsw not build yet, build it first
        exit 1
    fi

    if [ ! -f ${PWD}/lib/libivf-hnsw.a ]; then
        echo ivf-hnsw not build yet, build it first
        exit 1
    fi

    # following is not related with run program, it's used for build program which will use ivf-hnsw
    # as if ldconfig not use symbol link, must copy files
    sudo cp -f ${PWD}/lib/libfaiss.a /usr/local/lib/libfaiss.a
    sudo cp -f ${PWD}/lib/libhnswlib.a /usr/local/lib/libhnswlib.a
    sudo cp -f ${PWD}/lib/libivf-hnsw.a /usr/local/lib/libivf-hnsw.a

    sudo ldconfig
fi

if [ -L /usr/include/ivf-hnsw ] ; then
    sudo rm /usr/include/ivf-hnsw
    sudo ln -s ${PWD} /usr/include/ivf-hnsw
fi

if [ -L /usr/include/faiss ] ; then
    sudo rm /usr/include/faiss
    sudo ln -s ${PWD}/faiss/ /usr/include/faiss
fi

if [ -L /usr/include/hnswlib ] ; then
    sudo rm /usr/include/hnswlib
    sudo ln -s ${PWD}/hnswlib/ /usr/include/hnswlib
fi

if [ -L ${PWD}/visited_list_pool.h ] ; then
    sudo rm ${PWD}/visited_list_pool.h
    sudo ln -s ${PWD}/hnswlib/visited_list_pool.h ${PWD}/visited_list_pool.h
fi

if [ -L ${PWD}/hnswalg.h ] ; then
    sudo rm ${PWD}/hnswalg.h
    ln -s ${PWD}/hnswlib/hnswalg.h ${PWD}/hnswalg.h
fi

echo "success setup develop environment"
