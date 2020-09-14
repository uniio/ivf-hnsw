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
    Index_DB*               db_p  = nullptr;
    IndexIVF_HNSW_Grouping* index[2] = {nullptr, nullptr};
    int                     rc;
    size_t                  ver, batch_start, batch_end;
    std::vector<size_t>     batchs_to_index[2] = { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 }};

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
    index[0] = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);
    index[0]->do_opq = pq_conf.with_opq;
    index[1] = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);
    index[1]->do_opq = pq_conf.with_opq;

    //==========
    // Load PQ
    //==========
    rc = db_p->GetLatestIndexInfo(ver, batch_start, batch_end);
    if (rc) {
        std::cout << "Failed to get vector batch info" << std::endl;
        goto out;
    }
    if (ver == 0) {
        std::cout << "No batch info" << std::endl;
        // valid version always start from 1
    }

    rc = index[0]->load_quantizer(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load quantizer" << std::endl;
        goto out;
    }
    index[0]->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ codebooks" << std::endl;
        goto out;
    }
    rc = index[1]->load_quantizer(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load quantizer" << std::endl;
        goto out;
    }
    index[1]->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ codebooks" << std::endl;
        goto out;
    }
    rc = index[0]->build_batchs_to_index(sys_conf, batchs_to_index[0]);
    if (rc) {
        std::cout << "Failed to rebuild index 0" << std::endl;
        goto out;
    }
    rc = index[1]->build_batchs_to_index(sys_conf, batchs_to_index[1]);
    if (rc) {
        std::cout << "Failed to rebuild index 1" << std::endl;
        goto out;
    }

    std::cout << "Please check memory cost of two index" << std::endl;
    while (1) {
        sleep(1);
    }

out:
    if (db_p != nullptr) delete db_p;
    if (index[0] != nullptr)
        delete index[0];
    if (index[1] != nullptr)
        delete index[1];

    return rc;
}
