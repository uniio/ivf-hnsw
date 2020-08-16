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
    // in this test, path_base only stand for directory name
    // and don't consider how to get precomputed index in runtime
    std::vector<std::string> base_files;
    std::vector<std::string> precomputed_idxs_files;
    get_files(opt.path_base, ".bvecs", base_files);
    check_files("bigann_base_", base_files);
    get_files(opt.path_base, ".ivecs", precomputed_idxs_files);
    check_files("precomputed_idxs_sift1b_", precomputed_idxs_files);

    if (base_files.size() != precomputed_idxs_files.size()) {
        std::cout << "base vector segments not match with index segments" << std::endl;
        assert(0);
    }
    size_t segments_num = base_files.size();
    size_t segments_idx = 0;
    char  index_nm[1024];

add_loop:
    if (segments_idx == segments_num) {
        std::cout << "add vector test finish" << std::endl;
        delete index;
        return 0;
    }
    get_index_name(opt.path_index, segments_idx, index_nm);

    //=====================================
    // Construct IVF-HNSW + Grouping Index
    //=====================================
    if (exists(index_nm)) {
        // Load Index
        std::cout << "Loading index from " << index_nm << std::endl;
        index->read(index_nm);
    } else {
        // Adding groups to index 
        std::cout << "Adding groups to index" << std::endl;
        StopW stopw = StopW();

        auto base_segment = std::string(opt.path_base) + "/" + base_files[segments_idx];
        auto precomputed_idx_segment = std::string(opt.path_base) + "/" + precomputed_idxs_files[segments_idx];
        std::cout << "Load base vector from file: " << base_segment << std::endl;
        std::cout << "Load precomputed index vector from file: " << precomputed_idx_segment << std::endl;

        size_t vec_count = base_vec_num(base_segment.c_str(), opt.d);
        std::cout << "vector count " << vec_count << " in the loop" << std::endl;

        const size_t batch_size = 1000000;
        const size_t nbatches = vec_count / batch_size;
        size_t groups_per_iter = 250000;

        std::vector<uint8_t> batch(batch_size * opt.d);
        std::vector<idx_t> idx_batch(batch_size);

        for (size_t ngroups_added = 0; ngroups_added < opt.nc; ngroups_added += groups_per_iter)
        {
            std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                      << ngroups_added << " / " << opt.nc << std::endl;

            std::vector<std::vector<uint8_t>> data(groups_per_iter);
            std::vector<std::vector<idx_t>> ids(groups_per_iter);

            // Iterate through the dataset extracting points from groups,
            // whose ids lie in [ngroups_added, ngroups_added + groups_per_iter)
            std::ifstream base_input(base_segment, std::ios::binary);
            std::ifstream idx_input(precomputed_idx_segment, std::ios::binary);

            for (size_t b = 0; b < nbatches; b++) {
                readXvec<uint8_t>(base_input, batch.data(), opt.d, batch_size);
                readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);

                for (size_t i = 0; i < batch_size; i++) {
                    // only process index which lies in
                    // [ngroups_added, ngroups_added + groups_per_iter)
                    if (idx_batch[i] < ngroups_added ||
                        idx_batch[i] >= ngroups_added + groups_per_iter)
                        continue;

                    idx_t idx = idx_batch[i] % groups_per_iter;
                    for (size_t j = 0; j < opt.d; j++)
                        data[idx].push_back(batch[i * opt.d + j]);
                    ids[idx].push_back(b * batch_size + i);
                }
            }
            base_input.close();
            idx_input.close();

            // If <opt.nc> is not a multiple of groups_per_iter, change <groups_per_iter> on the last iteration
            if (opt.nc - ngroups_added <= groups_per_iter)
                groups_per_iter = opt.nc - ngroups_added;

            size_t j = 0;
            #pragma omp parallel for
            for (size_t i = 0; i < groups_per_iter; i++) {
                #pragma omp critical
                {
                    if (j % 10000 == 0) {
                        std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                                  << (100. * (ngroups_added + j)) / opt.nc
                                  << "%" << std::endl;
                    }
                    j++;
                }
                const size_t group_size = ids[i].size();
                std::vector<float> group_data(group_size * opt.d);
                // Convert bytes to floats
                for (size_t k = 0; k < group_size * opt.d; k++)
                    group_data[k] = 1. *data[i][k];

                index->add_group(ngroups_added + i, group_size, group_data.data(), ids[i].data());
            }
        }
        // Computing centroid norms and inter-centroid distances
        std::cout << "Computing centroid norms"<< std::endl;
        index->compute_centroid_norms();
        std::cout << "Computing centroid dists"<< std::endl;
        index->compute_inter_centroid_dists();

        // Save index, pq and norm_pq 
        index->write(index_nm, true);

        std::cout << "Saving index to " << index_nm<< std::endl;
        if (index->prepare_db()) {
            std::cout << "Failed to prepare database" << std::endl;
            exit(-1);
        }
//        if (index->write_db_index(segments_idx)) {
//            std::cout << "Failed to write index table" << std::endl;
//            exit(-1);
//        }
        if (index->create_new_batch(segments_idx)) {
            std::cout << "Failed to write index table" << std::endl;
            exit(-1);
        }
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
        index->search(opt.k, massQ.data() + i*opt.d, distances, labels);
        std::priority_queue<std::pair<float, idx_t >> gt(answers[i]);
        std::unordered_set<idx_t> g;

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
    std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.nq << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;

    // try to add next vector segment, add run query again
    if (segments_idx != segments_num) {
        segments_idx++;
        goto add_loop;
    }

    delete index;
    return 0;
}
