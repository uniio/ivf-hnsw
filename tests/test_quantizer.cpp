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

int main(int argc, char **argv) {
    //===============
    // Parse Options
    //===============
    Parser opt = Parser(argc, argv);

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
    IndexIVF_HNSW_Grouping* index = new IndexIVF_HNSW_Grouping(opt.d, opt.nc, opt.code_size, 8, opt.nsubc);
    index->build_quantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);
    index->do_opq = opt.do_opq;

    //====================
    // Precompute indices
    //====================
    std::cout << "Precomputing indices" << std::endl;
    std::vector<idx_t> precomputed_idx(opt.nq);
    index->quantizer->efSearch = 220;
    index->assign(opt.nq, massQ.data(), precomputed_idx.data());

    //=======================
    // Set search parameters
    //=======================
    index->nprobe              = opt.nprobe;
    index->quantizer->efSearch = opt.efSearch;
    size_t correct             = 0;
    for (size_t idx = 0; idx < opt.nq; idx++) {
        auto query_start = massQ.data();
        auto query       = query_start + idx * opt.d;
        auto coarse      = index->quantizer->searchKnn(query, index->nprobe);
        for (int_fast32_t i = index->nprobe - 1; i >= 0; i--) {
            idx_t centroid_idx = coarse.top().second;
            coarse.pop();
            if (centroid_idx == precomputed_idx[idx]) {
                correct++;
                break;
            }
        }
    }

    //===================
    // Represent results
    //===================
    std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.nq << std::endl;

    delete index;
    return 0;
}
