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
    Index_DB*     db_p = nullptr;
    IndexIVF_HNSW_Grouping* index = nullptr;

    // Initialize database interface
    // Following code is 1st start point in service, get work path first
    db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
    rc = db_p->Connect();
    if (rc) {
        std::cout << "Failed to connect to Database server" << std::endl;
        goto out;
    }
    rc = db_p->GetSysConfig(sys_conf);
    if (rc) {
        std::cout << "Failed to get system configuration" << std::endl;
        goto out;
    }

    rc = db_p->GetLatestPQConf(pq_conf);
    if (rc) {
        std::cout << "Failed to get PQ configuration" << std::endl;
        goto out;
    }

    //==================
    // Initialize Index
    //==================
    index = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);
    rc = index->load_quantizer(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load_quantizer" << std::endl;
        goto out;
    }

    // build precomputed index for all of batch in system
    rc = index->build_precomputed_index(sys_conf);
    if (rc) {
        std::cout << "Failed to build precomputed index" << std::endl;
        goto out;
    }

out:
    if (db_p) delete db_p;
    if (index) delete index;

    return rc;
}



