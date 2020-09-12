#!/bin/bash
if [ "$(expr substr $(uname -s) 1 5)" != "Linux" ]; then
    # path in msys2 is not same as Linux
    # create directory and symbol link only to fix CDT error in eclipse
    if [ ! -d /usr/include/postgresql ]; then
	    mkdir /usr/include/postgresql
    fi

    for file in `ls -l /mingw64/include|grep 'Jun 22'|awk '{print $9}'`
    do
        ln -s /mingw64/include/$file /usr/include/postgresql/$file
    done
fi
