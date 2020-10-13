#!/bin/bash

k="10"                 # Number of the closest vertices to search
nq="10000"             # Number of queries
path_data="/home/hzz/orcv_search/data"
path_q="${path_data}/split_1000/132/bigann_base_000.bvecs"
./search_index -k ${k} -nq ${nq} -path_q ${path_q}
