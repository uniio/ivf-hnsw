#!/bin/bash

RUN_SUDO=sudo

if [ ! -d "/usr/include/" ]; then
	echo "Not develop environment, no action for setup"
	exit 1
fi

if [ "$(expr substr $(uname -s) 1 5)" != "Linux" ]; then
    unset RUN_SUDO
fi

if [ -L /usr/include/ivf-hnsw ] ; then
    eval $RUN_SUDO rm /usr/include/ivf-hnsw
    eval $RUN_SUDO ln -s ${PWD} /usr/include/ivf-hnsw
else
    eval $RUN_SUDO ln -s ${PWD} /usr/include/ivf-hnsw
fi

if [ -L /usr/include/hnswlib ] ; then
    eval $RUN_SUDO rm /usr/include/hnswlib
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/ /usr/include/hnswlib
else
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/ /usr/include/hnswlib
fi

if [ -L ${PWD}/visited_list_pool.h ] ; then
    eval $RUN_SUDO rm ${PWD}/visited_list_pool.h
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/visited_list_pool.h ${PWD}/visited_list_pool.h
else
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/visited_list_pool.h ${PWD}/visited_list_pool.h
fi

if [ -L ${PWD}/hnswalg.h ] ; then
    eval $RUN_SUDO rm ${PWD}/hnswalg.h
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/hnswalg.h ${PWD}/hnswalg.h
else
    eval $RUN_SUDO ln -s ${PWD}/hnswlib/hnswalg.h ${PWD}/hnswalg.h
fi

if [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
    # make sure libpq-dev package installed, otherwise it will lead build failed
    eval $RUN_SUDO dpkg -s libpq-dev
    if [  $? -ne 0 ];then
        echo "libpq-dev not installed, please install it before build"
    fi

    # when the following lib files are ready, we think build ivf-hnsw done
    if [ ! -f ${PWD}/lib/libhnswlib.a ]; then
        echo ivf-hnsw not build yet, build it and run this script again
        exit 1
    fi

    if [ ! -f ${PWD}/lib/libivf-hnsw.a ]; then
        echo ivf-hnsw not build yet, build it and run this script again
        exit 1
    fi

    # following is not related with run program, it's used for build program which will use ivf-hnsw
    # as if ldconfig not use symbol link, must copy files
    eval $RUN_SUDO cp -f ${PWD}/lib/libhnswlib.a /usr/local/lib/libhnswlib.a
    eval $RUN_SUDO cp -f ${PWD}/lib/libivf-hnsw.a /usr/local/lib/libivf-hnsw.a

    eval $RUN_SUDO ldconfig
fi

echo "success setup develop environment"
