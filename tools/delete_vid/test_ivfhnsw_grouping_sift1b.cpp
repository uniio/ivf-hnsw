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
int main(int argc, char **argv) 
{
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
        std::cout << "Failed to Load PQ codebook." << std::endl;
        return 1;
    }

    //=====================================
    // Construct IVF-HNSW + Grouping Index 
    //=====================================
    if (exists(opt.path_index)){
        // Load Index 
        std::cout << "Loading index from " << opt.path_index << std::endl;
        index->read(opt.path_index);
    } else {
        std::cout << "Failed to Load index: " << opt.path_index << std::endl;
    }
    // For correct search using OPQ encoding rotate points in the coarse quantizer
    if (opt.do_opq) {
        std::cout << "Rotating centroids"<< std::endl;
        index->rotate_quantizer();
    }
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

    size_t base_id = 1 * 100 * 10000;
    StopW stopw = StopW();
    for (size_t i = 0; i < opt.nq; i++) {
        index->search(opt.k, massQ.data() + i*opt.d, distances, labels);
        std::cout << "query vector id: " << i + base_id << std::endl;
        for(size_t j = 0; j < opt.k; j++) {
            std::cout << "check search result " << labels[j];
            if(i + base_id == (size_t)labels[j]) {
                correct++;
                std::cout << " match" << std::endl;
            } else {
                std::cout << " not match" << std::endl;
            }
        }
    }
    idx_t vid = 0 + base_id;
    idx_t batch_pos = vid - base_id;
    index->delete_vid(massQ.data() + batch_pos*opt.d, vid);
    for (size_t i = 0; i < opt.nq; i++) {
        index->search(opt.k, massQ.data() + i*opt.d, distances, labels);
        std::cout << "query vector id: " << i + base_id << std::endl;
        for(size_t j = 0; j < opt.k; j++) {
            std::cout << "check search result " << labels[j];
            if(i + base_id == (size_t)labels[j]) {
                correct++;
                std::cout << " match" << std::endl;
            } else {
                std::cout << " not match" << std::endl;
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
