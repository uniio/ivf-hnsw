#!/bin/bash

RUN_SUDO=sudo

if [ ! -d "/usr/include/" ]; then
    echo "Not develop environment, no action for setup"
    exit 1
fi

if [ "$(expr substr $(uname -s) 1 5)" != "Linux" ]; then
    unset RUN_SUDO
fi

if [ -f ${PWD}/faisslib/faiss ]; then
    eval $RUN_SUDO rm ${PWD}/faisslib/faiss
fi
eval $RUN_SUDO ln -s ${PWD}/faisslib/faiss_987337 ${PWD}/faisslib/faiss

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    if [ ! -f ${PWD}/faisslib/faiss/lib/libfaiss.a ]; then
        echo libfaiss.a dies not exist, project may broken !
        exit 1
    fi
fi

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    eval $RUN_SUDO rm -f /usr/local/lib/libfaiss.a
    eval $RUN_SUDO cp -f ${PWD}/faisslib/faiss/lib/libfaiss.a /usr/local/lib/libfaiss.a
    eval $RUN_SUDO ldconfig
fi

eval $RUN_SUDO rm -r /usr/include/faiss
eval $RUN_SUDO ln -s ${PWD}/faisslib/faiss/header /usr/include/faiss
echo "setup_faiss_dev success"
