/*
 * build_precomputed_index.cpp
 *
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_DB.h>
#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>
#include <ivf-hnsw/hnswalg.h>
#include <ivf-hnsw/utils.h>

using std::cout;
using std::endl;

using namespace hnswlib;
using namespace ivfhnsw;

int main(int argc, char **argv) {
    system_conf_t sys_conf;
    pq_conf_t     pq_conf;
    int           rc;

    // Initialize database interface
    // Following code is 1st start point in service, get work path first
    Index_DB* db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
    rc = db_p->Connect();
    if (rc) {
        std::cout << "Failed to connect to Database server" << std::endl;
        exit(1);
    }
    rc = db_p->GetSysConfig(sys_conf);
    if (rc) {
        std::cout << "Failed to get system configuration" << std::endl;
        exit(1);
    }

    rc = db_p->GetLatestPQConf(pq_conf);
    if (rc) {
        std::cout << "Failed to get PQ configuration" << std::endl;
        exit(1);
    }

    //==================
    // Initialize Index
    //==================
    IndexIVF_HNSW_Grouping* index = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);
    index->load_quantizer(sys_conf, pq_conf);

    // build precomputed index for all of batch in system
    // TODO: in service1b the 2nd parameter should be batch number in active (ie. current used by service)
    rc = index->build_prcomputed_index(sys_conf, sys_conf.batch_max);
    if (rc) {
        std::cout << "Failed to build precomputed index" << std::endl;
        exit(1);
    }

    delete index;

    return 0;
}



