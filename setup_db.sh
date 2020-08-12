#!/bin/bash
if [ ! -f ${PWD}/dbsetup.sql ]; then
    echo file dbsetup.sql not exist
    exit 1
fi

export PGPASSWORD="postgres"

# drop previous database
dropdb -h localhost -p 5432 -U postgres servicedb

# create database for index service
createdb -h localhost -p 5432 -U postgres servicedb

# create system tables for index service
psql -d servicedb -U postgres -f ./dbsetup.sql
