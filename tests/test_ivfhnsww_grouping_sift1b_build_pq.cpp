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
    int   rc;
    Index_DB* db_p = nullptr;
    db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
    rc   = db_p->GetSysConfig(sys_conf);
    if (rc) {
        std::cout << "Failed to get configuration" << std::endl;
        exit(1);
    }
    {
        path_centroids = "${path_data}/centroids_sift1b.fvecs" 
        path_edges = "${path_model}/hnsw_M${M}_ef${efConstruction}.ivecs" 
        path_info = "${path_model}/hnsw_M${M}_ef${efConstruction}.bin" 
    }

    // Make sure 
    //==================
    // Initialize Index
    //==================
    IndexIVF_HNSW_Grouping* index = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc);
    index->build_quantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);
    index->do_opq = opt.do_opq;

    size_t code_size, nsubc;
    //==========
    // Train PQ
    //==========
    {
        int    rc;
        char   path_pq_base[1024], path_cur[512];
        char   path_learn[512], path_full[1024];
        size_t ver;
        bool   with_opq;

        // int IndexIVF_HNSW_Grouping::get_latest_pq_info(char *path, size_t &ver,
        // bool &with_opq, size_t &code_size, size_t &nsubc)
        // Here set a value for path_pq_base[0],
        // path_pq_base[0] = '.';
        if (index->get_latest_pq_info(path_pq_base, ver, with_opq, code_size, nsubc) {
            std::cout << "Error when get PQ info from database" << std::endl;
            exit(1);
        }
        if (path_pq_base[0] == '\0')
        {
            strcpy(path_learn, opt.path_learn);
            ver = 0;

            // all following is default value
            // TODO: defined default value as enume/const/macro later
            with_opq  = true;
            code_size = 16;
            nsubc     = 64;

            getwd(path_cur);
            sprintf(path_pq_base, "%s/models/SIFT1B/", path_cur);

            std::cout << "No PQ files found, start building" << std::endl;

            rc = index->build_pq_files(opt.path_learn, path_pq_base, true, 0, opt.code_size,
                                       0.00262144, opt.nsubc);
            if (rc) {
                std::cout << "Failed to build PQ files" << std::endl;
                exit(1);
            }
            std::cout << "Success build PQ files" << std::endl;
            rc = index->append_pq_info(opt.path_learn, 0, true, opt.code_size,
                                       opt.nsubc);
            if (rc) {
                std::cout << "Failed to add PQ files to database" << std::endl;
                exit(1);
            }
        } else {
            std::cout << "Load latest PQ files with version: " << ver << std::endl;

            // get PQ codebook path
            index->get_pq_path(path_base_model, ver, path_full);
            std::cout << "Loading Residual PQ codebook from " << path_full << std::endl;
            if (index->pq)
                delete index->pq;
            index->pq = faiss::read_ProductQuantizer(path_full);

            if (with_opq) {
                // get OPQ rotation matrix path
                index->get_opq_matrix_path(path_base_model, ver, path_full);
                std::cout << "Loading Residual OPQ rotation matrix from " << path_full << std::endl;
                index->opq_matrix = dynamic_cast<faiss::LinearTransform*>(faiss::read_VectorTransform(path_full));
            }

            // get norm PQ codebook path
            index->get_norm_pq_path(path_base_model, ver, path_full);
            std::cout << "Loading Norm PQ codebook from " << path_full << std::endl;
            if (index->norm_pq)
                delete index->norm_pq;
            index->norm_pq = faiss::read_ProductQuantizer(path_full);
        }

        // TODO: the fllowing example is how to cleanup old PQ file
        // if (index->action_on_pq(path_pq_base, 0, true, ACTION_PQ::PQ_CLEANUP) == false) {
        // }
    }
 

    //====================
    // Precompute indices
    //====================
    {
        char path_base_data[1024];
        char path_base_model[1024];
        Index_DB* db_p = nullptr;

        // Following code is 1st start point in service, get work path first

        db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
#    if 0 
        rc = db_p->GetPathInfo(path_base_data, path_base_model);
        if (rc) {
            std::cout << "Failed to get configuration" << std::endl;
            exit(1);
        }
#    endif

        // code for example, use following path which ivf-hnsw project used
        {
            getwd(path_cur);
            sprintf(path_base_data, "%s/data/SIFT1B/", path_cur);
            sprintf(path_base_model, "%s/models/SIFT1B/", path_cur);
        }

        // load batch of data
        std::vector<batch_info_t> batch_list
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

        // Example, not get those info from Postgres
        for (int i = 0; i < 100; i++) {
            batch_info_t batch_v;

            batch_v.batch = i;
            batch_v.valid = true;
            batch_list.emplace_back(bath_v);
        }

        if (batch_cur == 0 && batch_list.size() == 0) {
            // we can not start to service for query, there have no vector yet
            // we can only support vector add/delete
            // when we get query request, response with 405 error code
            // TODO:
        } else {
            for (size_t i = 0; i < batch_list.size(), i++) {
                index->get_vector_path(path_base_data, i, path_batch);
                index->get_vector_path(path_batch_precomputed_idx, i, path_batch);

                // build precomputed index
                // TODO: when service start, shold not build precomputed index
                // it will cost too much time and CPU resource
                if (!exists(path_batch_precomputed_idx)) {
                    rc = index->build_prcomputed_index(path_batch,
                                                       path_batch_precomputed_idx);
                    if (rc) {
                        std::cout << "Failed to build precomuted index for vector file: "
                                  << path_batch << std::endl;
                        exit(1);
                    }
                }
            }
        }
    }
    // path_index = "${path_model}/ivfhnsw_OPQ${code_size}_nsubc${nsubc}.index"
    //=====================================
    // Construct IVF-HNSW + Grouping Index
    //=====================================
    rc = GetLatestIndexInfo(ver, batch_start, batch_end);
    if (rc) {
        std::cout << "Failed to get index info" << std::endl;
        exit(1);
    }
    if (ver == 0) {
        std::cout << "No index info, cannot provide query service" << std::endl;
        // TODO: service need to disable query function
        // ie. response query request with 405 response code
    } else {
        // Load latest index
        char path_i5ndex[1024];

        index->get_index_path(path_base_model, ver, path_index);

        if (exists(path_index)) {
            // Load Index
            std::cout << "Loading index from " << path_index << std::endl;
            index->read(path_index);
        } else {
            // TODO: in service1b, we should report a BUG, because we only AppendIndexInto to database after we success write index into file
            // rc = index->build_index(path_base_data, batch_list[0], batch_list[batch_list.size() - 1], path_base_model, ver + 1);
            // if (rc) {
            //     // don't need to output message, already have in calling function
            //     exit(1);
            // }
        } 
    }


    // For correct search using OPQ encoding rotate points in the coarse
    // quantizer
    if (opt.do_opq) {
        std::cout << "Rotating centroids" << std::endl;
        index->rotate_quantizer();
    }
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
