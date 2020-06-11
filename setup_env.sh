#!/bin/bash

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

echo "success setup test environment"
