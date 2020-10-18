#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>
#include <ivf-hnsw/hnswalg.h>

using namespace hnswlib;
using namespace ivfhnsw;


//===========================================
// IVF-HNSW + Grouping (+ Pruning) on DEEP1B
//===========================================
// Note: during construction process,
// we use <groups_per_iter> parameter.
// Set it based on the capacity of your RAM
//===========================================
int main(int argc, char **argv) {
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
    //==================
    // Initialize Index 
    //==================
    IndexIVF_HNSW_Grouping *index = new IndexIVF_HNSW_Grouping(opt.d, opt.nc, opt.code_size, 8, opt.nsubc);
    index->build_quantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);
    index->do_opq = opt.do_opq;

    //==========
    // Train PQ 
    //==========
    if (exists(opt.path_pq) && exists(opt.path_norm_pq)) {
        std::cout << "Loading Residual PQ codebook from " << opt.path_pq << std::endl;
        if (index->pq) delete index->pq;
        index->pq = faiss::read_ProductQuantizer(opt.path_pq);

        if (opt.do_opq){
            std::cout << "Loading Residual OPQ rotation matrix from " << opt.path_opq_matrix << std::endl;
            index->opq_matrix = dynamic_cast<faiss::LinearTransform *>(faiss::read_VectorTransform(opt.path_opq_matrix));
        }
        std::cout << "Loading Norm PQ codebook from " << opt.path_norm_pq << std::endl;
        if (index->norm_pq) delete index->norm_pq;
        index->norm_pq = faiss::read_ProductQuantizer(opt.path_norm_pq);
    }
    else {
        // Load learn set
        std::vector<float> trainvecs(opt.nt * opt.d);
        {
            std::ifstream learn_input(opt.path_learn, std::ios::binary);
            readXvecFvec<uint8_t>(learn_input, trainvecs.data(), opt.d, opt.nt);
        }
        // Set Random Subset of sub_nt trainvecs
        std::vector<float> trainvecs_rnd_subset(opt.nsubt * opt.d);
        random_subset(trainvecs.data(), trainvecs_rnd_subset.data(), opt.d, opt.nt, opt.nsubt);

        std::cout << "Training PQ codebooks" << std::endl;
        index->train_pq(opt.nsubt, trainvecs_rnd_subset.data());

        if (opt.do_opq){
            std::cout << "Saving Residual OPQ rotation matrix to " << opt.path_opq_matrix << std::endl;
            faiss::write_VectorTransform(index->opq_matrix, opt.path_opq_matrix);
        }
        std::cout << "Saving Residual PQ codebook to " << opt.path_pq << std::endl;
        faiss::write_ProductQuantizer(index->pq, opt.path_pq);

        std::cout << "Saving Norm PQ codebook to " << opt.path_norm_pq << std::endl;
        faiss::write_ProductQuantizer(index->norm_pq, opt.path_norm_pq);
    }

    //====================
    // Precompute indices 
    //====================
    if (!exists(opt.path_precomputed_idxs)){
        std::cout << "Precomputing indices" << std::endl;
        StopW stopw = StopW();

        std::ifstream input(opt.path_base, std::ios::binary);
        std::ofstream output(opt.path_precomputed_idxs, std::ios::binary);

        const uint32_t batch_size = 1000000;
        const size_t nbatches = opt.nb / batch_size;

        std::vector<float> batch(batch_size * opt.d);
        std::vector<idx_t> precomputed_idx(batch_size);

        index->quantizer->efSearch = 220;
        for (size_t i = 0; i < nbatches; i++) {
            if (i % 10 == 0) {
                std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                          << (100.*i) / nbatches << "%" << std::endl;
            }
            readXvecFvec<uint8_t>(input, batch.data(), opt.d, batch_size);
            index->assign(batch_size, batch.data(), precomputed_idx.data());

            output.write((char *) &batch_size, sizeof(uint32_t));
            output.write((char *) precomputed_idx.data(), batch_size * sizeof(idx_t));
        }
    }

    //=====================================
    // Construct IVF-HNSW + Grouping Index 
    //=====================================
    // Force reconstruct index for debug
    {
        rc = index->add_one_batch_to_index_v2("/mnt/hdd_strip/orcv_search/data/split_1000/bigann_base_000.bvecs", "/root/src/ivf-hnsw/precomputed_idxs_000.ivecs", 0);
        if (rc) {
            std::cout << "Failed to access file: "
                      << "/mnt/hdd_strip/orcv_search/data/split_1000/bigann_base_000.bvecs" << std::endl;
            exit(-1);
        }
        rc = index->add_one_batch_to_index_v2("/mnt/hdd_strip/orcv_search/data/split_1000/bigann_base_001.bvecs", "/root/src/ivf-hnsw/precomputed_idxs_001.ivecs", 1000000L);
        if (rc) {
            std::cout << "Failed to access file: "
                      << "/mnt/hdd_strip/orcv_search/data/split_1000/bigann_base_001.bvecs" << std::endl;
            exit(-1);
        }
        // Computing centroid norms and inter-centroid distances
        std::cout << "Computing centroid norms" << std::endl;
        index->compute_centroid_norms();
        std::cout << "Computing centroid dists" << std::endl;
        index->compute_inter_centroid_dists();

        // Save index, pq and norm_pq
        std::cout << "Saving index to " << opt.path_index << std::endl;
        index->write(opt.path_index);
    }
    // For correct search using OPQ encoding rotate points in the coarse quantizer
    if (opt.do_opq) {
        std::cout << "Rotating centroids"<< std::endl;
        index->rotate_quantizer();
    }
    //===================
    // Parse groundtruth
    //=================== 
    std::cout << "Parsing groundtruth" << std::endl;
    std::vector<std::priority_queue< std::pair<float, idx_t >>> answers;
    (std::vector<std::priority_queue< std::pair<float, idx_t >>>(opt.nq)).swap(answers);
    for (size_t i = 0; i < opt.nq; i++)
        answers[i].emplace(0.0f, massQA[opt.ngt*i]);

    //=======================
    // Set search parameters
    //=======================
    index->nprobe = opt.nprobe;
    index->max_codes = opt.max_codes;
    index->quantizer->efSearch = opt.efSearch;
    index->do_pruning = opt.do_pruning;

    //========
    // Search 
    //========
    size_t correct = 0;
    float distances[opt.k];
    long labels[opt.k];

    StopW stopw = StopW();
    for (size_t i = 0; i < opt.nq; i++) {
        index->search(opt.k, massQ.data() + i * opt.d, distances, labels);
        for (size_t j = 0; j < opt.k; j++) {
            if ((i + 1000000) == labels[j]) {
                correct++;
            }
        }
    }
    //===================
    // Represent results 
    //===================
    const float time_us_per_query = stopw.getElapsedTimeMicro() / opt.nq;
    std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.nq << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;

    delete index;
    return 0;
}
