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

int main(int argc, char** argv) 
{
    int                     rc;
    system_conf_t           sys_conf;
    pq_conf_t               pq_conf;
    Index_DB*               db_p  = nullptr;
    IndexIVF_HNSW_Grouping* index = nullptr;
    size_t                  ver, batch_start, batch_end;
    std::vector<size_t>     batchs_to_index = {0};

    //==============
    // Parse Options
    //==============
    Parser opt = Parser(argc, argv);
    printf("k=%ld, nq=%ld, path_q=%s\n", opt.k, opt.nq, opt.path_q);
    std::vector<float>      massQ(opt.nq * 128);

    //==================================================================
    // Initialize database interface
    // Following code is 1st start point in service, get work path first
    //==================================================================
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

    //==============
    // Load Queries
    //==============
    std::cout << "Loading queries from " << opt.path_q << std::endl;
    //std::vector<float> massQ(nq * sys_conf.dim);
    {
        std::ifstream query_input(opt.path_q, std::ios::binary);
        readXvecFvec<uint8_t>(query_input, massQ.data(), sys_conf.dim, opt.nq);
    }

    //==================
    // Initialize Index
    //==================
    index = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);
    index->do_opq  = pq_conf.with_opq;

    //================
    // Load Quantizer
    //================
    rc = db_p->GetLatestIndexInfo(ver, batch_start, batch_end);
    if (rc) {
        std::cout << "Failed to get vector batch info" << std::endl;
        goto out;
    }
    if (ver == 0) {
        std::cout << "No batch info" << std::endl;
        goto out;
    }

    rc = index->load_quantizer(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load quantizer" << std::endl;
        goto out;
    }

    //================
    // Load PQ 
    //================
    index->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ codebooks" << std::endl;
        goto out;
    }

    //================
    // Load Index
    //================
#if 0
    rc = index->build_batchs_to_index_ex(sys_conf, batchs_to_index);
    if (rc) {
        std::cout << "Failed to build index" << std::endl;
        goto out;
    }
    std::cout << "Success to build index" << std::endl;
#else
    rc = index->load_index(sys_conf, ver);
    if (rc) {
        std::cout << "Failed to load index" << std::endl;
        goto out;
    }
    //============================================
    // Load Index Rotate quantizer
    // without this step , recall will alaways be 0
    //=============================================
    if (index->do_opq){
        std::cout << "Rotating centroids" << std::endl;
        index->rotate_quantizer();
    }
    std::cout << "Success load index" << std::endl;
#endif
    //=======================
    // Set SEARCH parameter
    //=======================
    index->nprobe = sys_conf.nprobe;
    index->max_codes = sys_conf.max_codes;
    index->do_pruning = sys_conf.do_pruning;
    index->quantizer->efSearch = sys_conf.efSearch;
    printf("nprobe=%ld, max_codes=%ld, do_pruning=%d, efSearch=%ld\n", 
        index->nprobe, index->max_codes, index->do_pruning, index->quantizer->efSearch);
    /* get batch list */
    index->load_sys_conf();
    /* get batch list size */
    rc = index->get_batchs_attr();
    if (rc) {
        std::cout << "Failed to Get Batch List File Size from DB" << std::endl;
        //goto out;
    }

    //==============
    // BEGIN SEARCH 
    //==============
    {
        size_t correct = 0;
        float  distances[opt.k];
        long   labels[opt.k];
        for (size_t i = 0; i < opt.nq; i++) {
            index->search(opt.k, massQ.data() + i*sys_conf.dim, distances, labels);
            std::cout << "query vector id: " << i << std::endl;
            for(size_t j = 0; j < opt.k; j++) {
                std::cout << "check search result " << labels[j];
                if((long)i == labels[j]) {
                    correct++;
                    std::cout << " match" << std::endl;
                } else {
                    std::cout << " not match" << std::endl;
                }
            }
        }
        std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.nq << std::endl;
    }

out:
    if (db_p != nullptr) delete db_p;
    if (index != nullptr) delete index;

    return rc;
}
