/*
 * load_index.cpp
 *
 */
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
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
    char                    path_vector[1024];
    size_t                  nq = 10000;
    size_t                  k  = 10;
    size_t                  correct = 0;
    std::vector<float>      massQ;
    size_t                  base_id_query;
    float                   distances[10];
    long                    labels[10];

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

    index         = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);
    index->do_opq = pq_conf.with_opq;

    ver = 1;

    rc = index->load_quantizer(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to build quantizer" << std::endl;
        rc = -1;
        goto out;
    }

    // rc = index->build_pq_files(sys_conf, pq_conf);
    // if (rc) {
    //     std::cout << "Failed to build PQ files" << std::endl;
    //     rc = -1;
    //     goto out;
    // }

    index->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ codebooks" << std::endl;
        goto out;
    }

    // rc = index->append_pq_info(sys_conf, pq_conf);
    // if (rc) {
    //     std::cout << "Failed to add PQ Info into database" << std::endl;
    //     rc = -1;
    //     goto out;
    // }

    // rc = index->build_precomputed_index(sys_conf);
    // if (rc) {
    //     std::cout << "Failed to build precomputed index" << std::endl;
    //     goto out;
    // }

    sprintf(path_vector, "%s/split_1000/bigann_base_%03lu.bvecs", sys_conf.path_base_data, 1L);
    std::cout << "Loading queries from " << path_vector << std::endl;

    massQ.resize(nq * sys_conf.dim);
    base_id_query = 1000000;
    {
        std::ifstream query_input(path_vector, std::ios::binary);
        readXvecFvec<uint8_t>(query_input, massQ.data(), sys_conf.dim, nq);
    }

    rc = index->build_batchs_to_index(sys_conf, 0, 1);
    if (rc) {
        std::cout << "Failed to build index" << std::endl;
        goto out;
    }
    std::cout << "Success to build index" << std::endl;

    //============================================
    // Load Index Rotate quantizer
    // without this step , recall will alaways be 0
    //=============================================
    if (index->do_opq) {
        std::cout << "Rotating centroids" << std::endl;
        index->rotate_quantizer();
    }

    index->nprobe              = sys_conf.nprobe;
    index->max_codes           = sys_conf.max_codes;
    index->do_pruning          = sys_conf.do_pruning;
    index->quantizer->efSearch = sys_conf.efSearch;
    for (size_t i = 0; i < nq; i++) {
        // std::vector<size_t> id_vectors;
        // index->search(k, massQ.data() + i * sys_conf.dim, id_vectors);
        index->search(k, massQ.data() + i * sys_conf.dim, distances, labels);
        std::cout << "query vector id: " << i << std::endl;
        for (size_t j = 0; j < k; j++) {
            // std::cout << "check search result " << id_vectors[j];
            std::cout << "check search result " << labels[j];
            if ((i + base_id_query) == labels[j]) {
                // if (((size_t)i + base_id_query) == id_vectors[j]) {
                    correct++;
                    std::cout << " match" << std::endl;
            }
                else {
                    std::cout << " not match" << std::endl;
                }
            }
    }
    std::cout << "Recall@" << k << ": " << 1.0f * correct / nq << std::endl;

out:
    if (db_p != nullptr) delete db_p;
    if (index != nullptr) delete index;

    return rc;
}
