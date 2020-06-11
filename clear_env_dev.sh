#!/bin/bash

RUN_SUDO=sudo

if [ ! -d "/usr/include/" ]; then
	echo "Not develop environment, no action for clear"
	exit 1
fi

if [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
    eval $RUN_SUDO rm -f /usr/local/lib/libfaiss.a
    eval $RUN_SUDO rm -f /usr/local/lib/libhnswlib.a
    eval $RUN_SUDO rm -f /usr/local/lib/libivf-hnsw.a

    eval $RUN_SUDO ldconfig
else
    unset RUN_SUDO
fi

if [ -L /usr/include/ivf-hnsw ] ; then
    eval $RUN_SUDO rm /usr/include/ivf-hnsw
fi

if [ -L /usr/include/faiss ] ; then
    eval $RUN_SUDO rm /usr/include/faiss
fi

if [ -L /usr/include/hnswlib ] ; then
    eval $RUN_SUDO rm /usr/include/hnswlib
fi

if [ -L ${PWD}/visited_list_pool.h ] ; then
    eval $RUN_SUDO rm ${PWD}/visited_list_pool.h
fi

if [ -L ${PWD}/hnswalg.h ] ; then
    eval $RUN_SUDO rm ${PWD}/hnswalg.h
fi

echo "success clear develop environment"
