#!/bin/bash

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

data_nfs_src=$(cat /etc/fstab|grep nfs_shared|awk '{print $1}')
nfs_path_mnt="/mnt/nfs_shared"

if [ ${data_nfs_src} =  "" ]; then
    echo "NFS server info not found in /etc/fstab"
    exit 1
fi

if [ ! -d ${nfs_path_mnt} ]; then
    echo "test data mount point not ready, create it"
    sudo mkdir -p ${$nfs_path_mnt}
fi

if [ ! -d ${nfs_path_mnt}/vector1b/ ]; then
    mnt_cmd=$(echo sudo /bin/mount -t nfs ${data_nfs_src} ${nfs_path_mnt})
    # echo ${mnt_cmd}
    eval ${mnt_cmd}
    if [ $? != 0 ]; then
        echo Failed to mount nfs data
        exit 1
    fi
fi

rm -fr "${PWD}/data"

if [ ! -d "${PWD}/data" ]; then
    ln -s ${nfs_path_mnt}/vector1b "${PWD}/data"
fi

if [ ! -d "${PWD}/models/SIFT1B" ]; then
    mkdir -p "${PWD}/models/SIFT1B"
fi

if [ ! -d "${PWD}/models/DEEP1B" ]; then
    mkdir -p "${PWD}/models/DEEP1B"
fi

if [ x"$1" = x"reset" ]; then
    echo "Clear existed models data"
    rm -f ${PWD}/models/SIFT1B/*
    rm -f ${PWD}/models/DEEP1B/*
fi

# following is not related with run program, it's used for build program which will use ivf-hnsw
# as if ldconfig not use symbol link, must copy files
sudo cp -f ${PWD}/lib/libfaiss.a /usr/local/lib/libfaiss.a
sudo cp -f ${PWD}/lib/libhnswlib.a /usr/local/lib/libhnswlib.a
sudo cp -f ${PWD}/lib/libivf-hnsw.a /usr/local/lib/libivf-hnsw.a

sudo ldconfig

echo "success setup test environment"
