#!/bin/bash

#RUN_SUDO=sudo
CURRENT_FASISS=faiss_release_o2

if [ "$(expr substr $(uname -s) 1 5)" != "Linux" ]; then
    unset RUN_SUDO
fi

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
	# make sure libpq-dev package installed, otherwise it will lead build failed
    eval $RUN_SUDO dpkg -s libpq-dev > /dev/null 2>&1
    if [  $? -ne 0 ];then
        echo "libpq-dev not installed, please install it before build"
        exit 1
    fi

    if [ ! -f ${PWD}/faisslib/${CURRENT_FASISS}/lib/libfaiss.a ]; then
        echo libfaiss.a dies not exist, project may broken !
        exit 1
    fi
fi

if [ -L ${PWD}/faisslib/faiss ]; then
    eval $RUN_SUDO rm ${PWD}/faisslib/faiss
fi

if [ -L ${PWD}/faisslib/lib ]; then
    eval $RUN_SUDO rm ${PWD}/faisslib/lib
fi

eval $RUN_SUDO ln -s ${PWD}/faisslib/${CURRENT_FASISS}/header ${PWD}/faisslib/faiss
eval $RUN_SUDO ln -s ${PWD}/faisslib/${CURRENT_FASISS}/lib ${PWD}/faisslib/lib

if [ -L ${PWD}/hnswalg.h ] ; then
    eval $RUN_SUDO rm ${PWD}/hnswalg.h
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/hnswalg.h ${PWD}/hnswalg.h
else
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/hnswalg.h ${PWD}/hnswalg.h
fi

echo "success setup develop environment"
