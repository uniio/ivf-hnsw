/*
 * build_index.cpp
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

int main(int argc, char** argv) {
    system_conf_t           sys_conf;
    pq_conf_t               pq_conf;
    Index_DB*               db_p     = nullptr;
    IndexIVF_HNSW_Grouping* index = nullptr;
    int                     rc;
    std::vector<size_t>     batchs_to_index = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    // Initialize database interface
    // Following code is 1st start point in service, get work path first
    db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
    rc   = db_p->Connect();
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
    index->do_opq = pq_conf.with_opq;

    //==========
    // Load PQ
    //==========

    rc = index->load_quantizer(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load quantizer" << std::endl;
        goto out;
    }
    index->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ codebooks" << std::endl;
        goto out;
    }

    rc = index->build_batchs_to_index(sys_conf, batchs_to_index);
    if (rc) {
        std::cout << "Failed to rebuild index 1" << std::endl;
        goto out;
    }

    std::cout << "Please check memory cost of index" << std::endl;
    while (1) {
        sleep(1);
    }

out:
    if (db_p != nullptr)
        delete db_p;
    if (index != nullptr)
        delete index;

    return rc;
}
