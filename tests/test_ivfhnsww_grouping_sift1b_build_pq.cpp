#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>
#include <ivf-hnsw/hnswalg.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/utils.h>

using namespace hnswlib;
using namespace ivfhnsw;

//===========================================
// IVF-HNSW + Grouping (+ Pruning) on DEEP1B
//===========================================
// Note: during construction process,
// we use <groups_per_iter> parameter.
// Set it based on the capacity of your RAM
//===========================================
int main(int argc, char** argv)
{
    //===============
    // Parse Options
    //===============
    Parser opt = Parser(argc, argv);

    //==================
    // Load Groundtruth
    //==================
    std::cout << "Loading groundtruth from " << opt.path_gt << std::endl;
    std::vector<idx_t> massQA(opt.nq * opt.ngt);
    {
        std::ifstream gt_input(opt.path_gt, std::ios::binary);
        readXvec<idx_t>(gt_input, massQA.data(), opt.ngt, opt.nq);
    }
    //==============
    // Load Queries
    //==============
    std::cout << "Loading queries from " << opt.path_q << std::endl;
    std::vector<float> massQ(opt.nq * opt.d);
    {
        std::ifstream query_input(opt.path_q, std::ios::binary);
        readXvecFvec<uint8_t>(query_input, massQ.data(), opt.d, opt.nq);
    }

    // Initialize database interface
    // Following code is 1st start point in service, get work path first
    system_conf_t sys_conf;
    pq_conf_t pq_conf;
    int rc;
    Index_DB* db_p = nullptr;
    char path_full[1024];
    db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
    rc   = db_p->GetSysConfig(sys_conf);
    if (rc) {
        std::cout << "Failed to get system configuration" << std::endl;
        exit(1);
    }
    rc = db_p->GetLatestPQConf(pq_conf);
    if (rc) {
        std::cout << "Failed to get PQ configuration" << std::endl;
        exit(1);
    }

    // Make sure
    //==================
    // Initialize Index
    //==================
    IndexIVF_HNSW_Grouping *index = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc);

    // Construct quantizer, which will be used to build
    {
        char path_full[1024];
        char path_centroids[1024], path_info[1024], path_edges[1024];
        index->get_path_centroids(sys_conf, path_centroids);
        index->get_path_info(sys_conf, pq_conf, path_info);
        index->get_path_edges(sys_conf, pq_conf, path_edges);
        index->build_quantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);
    }

    index->do_opq = opt.do_opq;

    size_t code_size, nsubc;
    bool   with_opq;
    size_t pq_ver;
    //==========
    // Load PQ
    // In design, we will not Train PQ in service1b, it will be handled by seperate tool
    // TODO: may need discuss, if it's good design decision
    //==========
    // if (index->get_latest_pq_info(pq_ver, with_opq, code_size, nsubc)) 
    {
        std::cout << "Error when get PQ info from database" << std::endl;
        exit(1);
    }
    if (code_size == 0) {
        // code_size == 0 means not found record in database
        std::cout << "Error: not fond PQ info from database" << std::endl;
        exit(1);
    } else {
        std::cout << "Load latest PQ files with version: " << pq_ver << std::endl;

        // get PQ codebook path
        index->get_path_pq(sys_conf, pq_ver, path_full);
        std::cout << "Loading Residual PQ codebook from " << path_full
                  << std::endl;
        if (index->pq)
            delete index->pq;
        index->pq = faiss::read_ProductQuantizer(path_full);

        if (with_opq) {
            // get OPQ rotation matrix path
            index->get_path_opq_matrix(sys_conf, pq_ver, path_full);
            std::cout << "Loading Residual OPQ rotation matrix from " << path_full
                      << std::endl;
            index->opq_matrix = dynamic_cast<faiss::LinearTransform *>(
                faiss::read_VectorTransform(path_full));
        }

        // get norm PQ codebook path
        index->get_path_norm_pq(sys_conf, pq_ver, path_full);
        std::cout << "Loading Norm PQ codebook from " << path_full << std::endl;
        if (index->norm_pq)
            delete index->norm_pq;
        index->norm_pq = faiss::read_ProductQuantizer(path_full);
    }

    /*
     * batch_cur     batch of current vector file  
     * idx_cur       index of current precomputed index file
     * 
     * batch_cur < idx_cur, becuase precomputed index file generate by vector file
     */
    size_t batch_cur, idx_cur;
    char path_batch_idx[1024];
    //====================
    // Load precompute indices
    // In design, we will not build precompute indices, it will be handled by seperate tool
    // TODO: may need discuss, if it's good design decision
    // 
    //====================
    {
        int rc;
        std::vector<batch_info_t> batch_list;
        rc = db_p->GetBatchList(batch_list);
        if (rc) {
            std::cout << "Failed to get batch list from database" << std::endl;
            exit(1);
        }

        // it's first time start service, no batch vector file yet
        if (batch_list.size() == 0) {
            batch_cur = 0;
        } else {
            batch_cur = batch_list[0].batch;
        }

        if (batch_cur == 0 && batch_list.size() == 0) {
            // we can not start to service for query, there have no vector yet
            // we can only support vector add/delete
            // when we get query request, response with 405 error code
            // TODO: service1b must remove 
            // vector delete may be only need to support delete by batch index, not delete by single vector
        } else {
            char path_batch_precomputed_idx[1024];
            for (size_t i = 0; i < batch_list.size(); i++) {
                index->get_path_vector(sys_conf, i, path_batch_idx);
                index->get_path_precomputed_idx(sys_conf, i, path_batch_precomputed_idx);

                // TODO: when service start, shold not build precomputed index
                // it will cost too much time and CPU resource
                if (!exists(path_batch_precomputed_idx)) {
                    idx_cur = i;
                    break;
                }
            }
        }

        if (idx_cur == 0) {
            std::cout << "No precomputed index not found" << std::endl;
            // TODO: disable query function from service1b
        }
    }

    //=====================================
    // Construct IVF-HNSW + Grouping Index
    //=====================================
    size_t ver_idx;
    {
        size_t batch_start, batch_end;

        rc = db_p->GetLatestIndexInfo(ver_idx, batch_start, batch_end);
        if (rc) {
            std::cout << "Failed to get index info" << std::endl;
            exit(1);
        }
        if (ver_idx == 0) {
            std::cout << "No index info, cannot provide query service" << std::endl;
            // valid index version should start from 1
            // TODO: service need to disable query function
            // ie. response query request with 405 response code
        } else {
            // Load latest index
            char path_index[1024];

            index->get_path_index(sys_conf, ver_idx, path_index);
            if (exists(path_index)) {
                // Load Index
                std::cout << "Loading index from " << path_index << std::endl;
                index->read(path_index);
            } else {
                // TODO: in service1b, we should report a BUG, because we only AppendIndexInto to database after we success write index into file
            } 
        }
    }

    // if enable opq in PQ quantizer parameter
    // correct search using OPQ encoding rotate points in the coarse quantizer
    if (pq_conf.with_opq) {
        std::cout << "Rotating centroids" << std::endl;
        index->rotate_quantizer();
    }
    exit(1);
    //===================
    // Parse groundtruth
    //===================
    std::cout << "Parsing groundtruth" << std::endl;
    std::vector<std::priority_queue<std::pair<float, idx_t>>> answers;
    (std::vector<std::priority_queue<std::pair<float, idx_t>>>(opt.nq))
        .swap(answers);
    for (size_t i = 0; i < opt.nq; i++)
        answers[i].emplace(0.0f, massQA[opt.ngt * i]);

    //=======================
    // Set search parameters
    //=======================
    index->nprobe              = opt.nprobe;
    index->max_codes           = opt.max_codes;
    index->quantizer->efSearch = opt.efSearch;
    index->do_pruning          = opt.do_pruning;

    //========
    // Search
    //========
    size_t correct = 0;
    float  distances[opt.k];
    long   labels[opt.k];

    StopW stopw = StopW();
    for (size_t i = 0; i < opt.nq; i++) {
        index->search(opt.k, massQ.data() + i * opt.d, distances, labels);
        std::priority_queue<std::pair<float, idx_t>> gt(answers[i]);
        std::unordered_set<idx_t>                    g;

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        for (size_t j = 0; j < opt.k; j++)
            if (g.count(labels[j]) != 0) {
                correct++;
                break;
            }
    }
    //===================
    // Represent results
    //===================
    const float time_us_per_query = stopw.getElapsedTimeMicro() / opt.nq;
    std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.nq
              << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;

    delete index;
    return 0;
}
