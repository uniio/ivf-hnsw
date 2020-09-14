/*
 * load_index.cpp
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
    Index_DB*               db_p  = nullptr;
    IndexIVF_HNSW_Grouping* index = nullptr;
    int                     rc;
    size_t                  ver, batch_start, batch_end;

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
    index->do_opq  = pq_conf.with_opq;

    //==========
    // Load PQ
    //==========
    rc = index->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ CodeBooks" << std::endl;
        goto out;
    }
    std::cout << "Success to load PQ CodeBooks" << std::endl;

    rc = db_p->GetLatestIndexInfo(ver, batch_start, batch_end);
    if (rc) {
        std::cout << "Failed to get vector batch info" << std::endl;
        goto out;
    }
    if (ver == 0) {
        std::cout << "No batch info" << std::endl;
        // valid version always start from 1
        rc  = 1;
        goto out;
    }

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

    rc = index->load_index(sys_conf, ver);
    if (rc) {
        std::cout << "Failed to load index" << std::endl;
        goto out;
    }
    std::cout << "Success load index" << std::endl;

out:
    if (db_p != nullptr) delete db_p;
    if (index != nullptr) delete index;

    return rc;
}
