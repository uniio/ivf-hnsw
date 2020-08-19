#include "IndexIVF_HNSW_Grouping.h"
#include <unistd.h>
#include <algorithm>

// #define TRACE_NEIGHBOUR

namespace ivfhnsw
{
    static FILE *fp_centriod = NULL;
    static const char *log_centriod = "centriod.log";

    int centriodTraceSetup()
    {
        fp_centriod = fopen(log_centriod, "w");
        if (fp_centriod == NULL)
            return -1;

        return 0;
    }

    void centriodTraceClose()
    {
        if (fp_centriod != NULL)
            fclose(fp_centriod);
    }


    //================================================
    // IVF_HNSW + grouping( + pruning) implementation
    //================================================
    IndexIVF_HNSW_Grouping::IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                                                   size_t nbits_per_idx, size_t nsubcentroids):
           IndexIVF_HNSW(dim, ncentroids, bytes_per_code, nbits_per_idx), nsubc(nsubcentroids)
    {
        alphas.resize(nc);
        nn_centroid_idxs.resize(nc);
        subgroup_sizes.resize(nc);

        query_centroid_dists.resize(nc);
        std::fill(query_centroid_dists.begin(), query_centroid_dists.end(), 0);
        inter_centroid_dists.resize(nc);
    }

    IndexIVF_HNSW_Grouping::IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                                                   size_t nbits_per_idx, size_t nsubcentroids, Index_DB *db_ref):
           IndexIVF_HNSW(dim, ncentroids, bytes_per_code, nbits_per_idx), nsubc(nsubcentroids)
    {
        alphas.resize(nc);
        nn_centroid_idxs.resize(nc);
        subgroup_sizes.resize(nc);

        query_centroid_dists.resize(nc);
        std::fill(query_centroid_dists.begin(), query_centroid_dists.end(), 0);
        inter_centroid_dists.resize(nc);

        db_p = db_ref;
    }

    IndexIVF_HNSW_Grouping::~IndexIVF_HNSW_Grouping()
    {
        if (db_free && db_p)
            delete db_p;
    }

    // try to connect database
    int IndexIVF_HNSW_Grouping::setup_db(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr)
    {
        int rc;

        db_p = new Index_DB(host, port, db_nm, db_usr, pwd_usr);
        rc = db_p->Connect();
        if (rc) {
            delete db_p;
            db_p = nullptr;
        } else {
            db_free =  true;
        }

        return rc;
    }

    void IndexIVF_HNSW_Grouping::add_group(size_t centroid_idx, size_t group_size,
                                           const float *data, const idx_t *idxs)
    {
        // Find NN centroids to source centroid 
        const float *centroid = quantizer->getDataByInternalId(centroid_idx);
        std::priority_queue<std::pair<float, idx_t>> nn_centroids_raw = quantizer->searchKnn(centroid, nsubc + 1);

        std::vector<float> centroid_vector_norms_L2sqr(nsubc);
        nn_centroid_idxs[centroid_idx].resize(nsubc);
        while (nn_centroids_raw.size() > 1) {
            centroid_vector_norms_L2sqr[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
            nn_centroid_idxs[centroid_idx][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;

            if (fp_centriod) {
                // if centriod info trace enabled, log it
                fprintf(fp_centriod, "centroid index:\t%u\tsub centroid distance:\t%f\n",
                        centroid_idx, centroid_vector_norms_L2sqr[nn_centroids_raw.size() - 2]);
            }

            nn_centroids_raw.pop();
        }
        if (group_size == 0)
            return;

        const float *centroid_vector_norms = centroid_vector_norms_L2sqr.data();
        const idx_t *nn_centroids = nn_centroid_idxs[centroid_idx].data();

        // Compute centroid-neighbor_centroid and centroid-group_point vectors
        std::vector<float> centroid_vectors(nsubc * d);
        for (size_t subc = 0; subc < nsubc; subc++) {
            const float *neighbor_centroid = quantizer->getDataByInternalId(nn_centroids[subc]);
            faiss::fvec_madd(d, neighbor_centroid, -1., centroid, centroid_vectors.data() + subc * d);
        }

        // Compute alpha for group vectors
        alphas[centroid_idx] = compute_alpha(centroid_vectors.data(), data, centroid,
                                             centroid_vector_norms, group_size);

        // Compute final subcentroids
        std::vector<float> subcentroids(nsubc * d);
        for (size_t subc = 0; subc < nsubc; subc++) {
            const float *centroid_vector = centroid_vectors.data() + subc * d;
            float *subcentroid = subcentroids.data() + subc * d;
            faiss::fvec_madd(d, centroid, alphas[centroid_idx], centroid_vector, subcentroid);
        }

        // Find subcentroid idx
        std::vector<idx_t> subcentroid_idxs(group_size);
        compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data, group_size);

        // Compute residuals
        std::vector<float> residuals(group_size * d);
        compute_residuals(group_size, data, residuals.data(), subcentroids.data(), subcentroid_idxs.data());

        // Rotate residuals
        if (do_opq){
            std::vector<float> copy_residuals(group_size * d);
            memcpy(copy_residuals.data(), residuals.data(), group_size * d * sizeof(float));
            opq_matrix->apply_noalloc(group_size, copy_residuals.data(), residuals.data());
        }

        // Compute codes
        std::vector<uint8_t> xcodes(group_size * code_size);
        pq->compute_codes(residuals.data(), xcodes.data(), group_size);

        // Decode codes
        std::vector<float> decoded_residuals(group_size * d);
        pq->decode(xcodes.data(), decoded_residuals.data(), group_size);

        // Reverse rotation
        if (do_opq){
            std::vector<float> copy_decoded_residuals(group_size * d);
            memcpy(copy_decoded_residuals.data(), decoded_residuals.data(), group_size * d * sizeof(float));
            opq_matrix->transform_transpose(group_size, copy_decoded_residuals.data(), decoded_residuals.data());
        }

        // Reconstruct data
        std::vector<float> reconstructed_x(group_size * d);
        reconstruct(group_size, reconstructed_x.data(), decoded_residuals.data(),
                    subcentroids.data(), subcentroid_idxs.data());

        // Compute norms 
        std::vector<float> norms(group_size);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, group_size);

        // Compute norm codes
        std::vector<uint8_t> xnorm_codes(group_size);
        norm_pq->compute_codes(norms.data(), xnorm_codes.data(), group_size);

        // Distribute codes
        std::vector<std::vector<idx_t> > construction_ids(nsubc);
        std::vector<std::vector<uint8_t> > construction_codes(nsubc);
        std::vector<std::vector<uint8_t> > construction_norm_codes(nsubc);
        for (size_t i = 0; i < group_size; i++) {
            idx_t idx = idxs[i];
            idx_t subcentroid_idx = subcentroid_idxs[i];

            construction_ids[subcentroid_idx].push_back(idx);
            construction_norm_codes[subcentroid_idx].push_back(xnorm_codes[i]);
            for (size_t j = 0; j < code_size; j++)
                construction_codes[subcentroid_idx].push_back(xcodes[i * code_size + j]);
        }
        // Add codes to the index
        for (size_t subc = 0; subc < nsubc; subc++) {
            idx_t subgroup_size = construction_norm_codes[subc].size();
            subgroup_sizes[centroid_idx].push_back(subgroup_size);

            for (size_t i = 0; i < subgroup_size; i++) {
                ids[centroid_idx].push_back(construction_ids[subc][i]);
                for (size_t j = 0; j < code_size; j++)
                    codes[centroid_idx].push_back(construction_codes[subc][i * code_size + j]);
                norm_codes[centroid_idx].push_back(construction_norm_codes[subc][i]);
            }
        }
    }

    /** Search procedure
      *
      * During the IVF-HNSW-PQ + Grouping search we compute
      *
      *  d = || x - y_S - y_R ||^2
      *
      * where x is the query vector, y_S the coarse sub-centroid, y_R the
      * refined PQ centroid. The expression can be decomposed as:
      *
      *  d = (1 - α) * (|| x - y_C ||^2 - || y_C ||^2) + α * (|| x - y_N ||^2 - || y_N ||^2) + || y_S + y_R ||^2 - 2 * (x|y_R)
      *      -----------------------------------------   -----------------------------------   -----------------   -----------
      *                         term 1                                 term 2                        term 3          term 4
      *
      * We use the following decomposition:
      * - term 1 is the distance to the coarse centroid, that is computed
      *   during the 1st stage search in the HNSW graph, minus the norm of the coarse centroid.
      * - term 2 is the distance to y_N one of the <subc> nearest centroids,
      *   that is used for the sub-centroid computation, minus the norm of this centroid.
      * - term 3 is the L2 norm of the reconstructed base point, that is computed at construction time, quantized
      *   using separately trained product quantizer for such norms and stored along with the residual PQ codes.
      * - term 4 is the classical non-residual distance table.
      *
      * Norms of centroids are precomputed and saved without compression, as their memory consumption is negligible.
      * If it is necessary, the norms can be added to the term 3 and compressed to byte together. We do not think that
      * it will lead to considerable decrease in accuracy.
      *
      * Since y_R defined by a product quantizer, it is split across
      * sub-vectors and stored separately for each sub-vector.
    */
    void IndexIVF_HNSW_Grouping::search(size_t k, const float *x, float *distances, long *labels)
    {
        // Distances to subcentroids. Used for pruning.
        std::vector<float> query_subcentroid_dists;

        // Indices of coarse centroids, which distances to the query are computed during the search time
        std::vector<idx_t> used_centroid_idxs;
        used_centroid_idxs.reserve(nsubc * nprobe);
        idx_t centroid_idxs[nprobe]; // Indices of the nearest coarse centroids

        // For correct search using OPQ rotate a query
        const float *query = (do_opq) ? opq_matrix->apply(1, x) : x;

#ifdef TRACE_CENTROIDS
        trace_query_centroid_dists.clear();
        trace_centroid_idxs.clear();
#endif

        // Find the nearest coarse centroids to the query
        auto coarse = quantizer->searchKnn(query, nprobe);
        for (int_fast32_t i = nprobe - 1; i >= 0; i--) {
            idx_t centroid_idx = coarse.top().second;
            centroid_idxs[i] = centroid_idx;
            query_centroid_dists[centroid_idx] = coarse.top().first;
            used_centroid_idxs.push_back(centroid_idx);

#ifdef TRACE_CENTROIDS
            trace_query_centroid_dists.push_back(coarse.top().first);
            trace_centroid_idxs.push_back(coarse.top().second);
#endif

            coarse.pop();
        }

        // Computing threshold for pruning
        float threshold = 0.0;
        if (do_pruning) {
            size_t ncode = 0;
            size_t nsubgroups = 0;

            query_subcentroid_dists.resize(nsubc * nprobe);
            float *qsd = query_subcentroid_dists.data();

            for (size_t i = 0; i < nprobe; i++) {
                const idx_t centroid_idx = centroid_idxs[i];
                const size_t group_size = norm_codes[centroid_idx].size();
                if (group_size == 0)
                    continue;

                const float alpha = alphas[centroid_idx];
                const float term1 = (1 - alpha) * query_centroid_dists[centroid_idx];

                for (size_t subc = 0; subc < nsubc; subc++) {
                    if (subgroup_sizes[centroid_idx][subc] == 0)
                        continue;

                    const idx_t nn_centroid_idx = nn_centroid_idxs[centroid_idx][subc];
                    // Compute the distance to the coarse centroid if it is not computed
                    if (query_centroid_dists[nn_centroid_idx] < EPS) {
                        const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idx);
                        query_centroid_dists[nn_centroid_idx] = fvec_L2sqr(query, nn_centroid, d);
                        used_centroid_idxs.push_back(nn_centroid_idx);
                    }
                    qsd[subc] = term1 - alpha * ((1 - alpha) * inter_centroid_dists[centroid_idx][subc]
                                                 - query_centroid_dists[nn_centroid_idx]);
                    threshold += qsd[subc];
                    nsubgroups++;
                }
                ncode += group_size;
                qsd += nsubc;
                if (ncode >= 2 * max_codes)
                    break;
            }
            threshold /= nsubgroups;
        }

        // Precompute table
        pq->compute_inner_prod_table(query, precomputed_table.data());

        // Prepare max heap with k answers
        faiss::maxheap_heapify(k, distances, labels);

        size_t ncode = 0;
        const float *qsd = query_subcentroid_dists.data();

#ifdef TRACE_NEIGHBOUR
        std::vector<float> query_vector_dists;
        char *neighbour_log = "neighbour_hit.log";
        std::ofstream log_trace;
        log_trace.open(neighbour_log, std::ofstream::out | std::ofstream::app);
        if (!log_trace.is_open()) {
            std::cout << "Failed to open log file for neighbour traceing" << std::endl;
        }
#endif

        for (size_t i = 0; i < nprobe; i++) {
            const idx_t centroid_idx = centroid_idxs[i];
            const size_t group_size = norm_codes[centroid_idx].size();
            if (group_size == 0)
                continue;

            const float alpha = alphas[centroid_idx];
            const float term1 = (1 - alpha) * (query_centroid_dists[centroid_idx] - centroid_norms[centroid_idx]);

            const uint8_t *code = codes[centroid_idx].data();
            const uint8_t *norm_code = norm_codes[centroid_idx].data();
            const idx_t *id = ids[centroid_idx].data();

#ifdef TRACE_NEIGHBOUR
            log_trace << "centroid " << centroid_idx << " with threshold: " << threshold;
            log_trace << " get neighbours distance:\n";
            query_vector_dists.clear();
#endif

            for (size_t subc = 0; subc < nsubc; subc++) {
                const size_t subgroup_size = subgroup_sizes[centroid_idx][subc];
                if (subgroup_size == 0)
                    continue;

                // Check pruning condition
                if (!do_pruning || qsd[subc] < threshold) {
                    const idx_t nn_centroid_idx = nn_centroid_idxs[centroid_idx][subc];

                    // Compute the distance to the coarse centroid if it is not computed
                    if (query_centroid_dists[nn_centroid_idx] < EPS) {
                        const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idx);
                        query_centroid_dists[nn_centroid_idx] = fvec_L2sqr(query, nn_centroid, d);
                        used_centroid_idxs.push_back(nn_centroid_idx);
                    }

                    const float term2 = alpha * (query_centroid_dists[nn_centroid_idx] - centroid_norms[nn_centroid_idx]);
                    norm_pq->decode(norm_code, norms.data(), subgroup_size);

                    for (size_t j = 0; j < subgroup_size; j++) {
                        const float term4 = 2 * pq_L2sqr(code + j * code_size);
                        const float dist = term1 + term2 + norms[j] - term4; //term3 = norms[j]
#ifdef TRACE_NEIGHBOUR
                        if (log_trace.is_open()) {
                            query_vector_dists.push_back(dist);
                        }
#endif
                        if (dist < distances[0]) {
                            faiss::maxheap_pop(k, distances, labels);
                            faiss::maxheap_push(k, distances, labels, dist, id[j]);
                        }
                    }
                    ncode += subgroup_size;
                }
                // Shift to the next group
                code += subgroup_size * code_size;
                norm_code += subgroup_size;
                id += subgroup_size;
            }
#ifdef TRACE_NEIGHBOUR
            if (log_trace.is_open()) {
                std::sort(query_vector_dists.begin(), query_vector_dists.end());
                for (auto it = query_vector_dists.begin(); it < query_vector_dists.end(); it++) {
                    log_trace << *it << "\n";
                }
            }
#endif
            if (ncode >= max_codes)
                break;
            if (do_pruning)
                qsd += nsubc;
        }
#ifdef TRACE_NEIGHBOUR
        if (log_trace.is_open()) log_trace.close();
#endif
        // Zero computed dists for later queries
        for (idx_t used_centroid_idx : used_centroid_idxs)
            query_centroid_dists[used_centroid_idx] = 0;

        if (do_opq)
            delete const_cast<float *>(query);
    }

    void IndexIVF_HNSW_Grouping::searchDisk(size_t k, const float *query, float *distances,
                                            long *labels, const char *path_base)
    {
        float distances_base[2*k];
        long labels_base[2*k];
        std::vector<SearchInfo_t> searchRet(2*k);

        // search by ANN first to get 2*k result
        search(k, query, distances_base, labels_base);

        /*
         *  get vector from disk according to result's lablel value (vector index in base vector file)
         *  and recalculate exactly distance between vector and query
         */
        SearchInfo_t sret;
        for (int di = 2*k - 1; di >= 0; di--) {
            sret.distance = getL2Distance(query, path_base, d,
                    labels_base[di], base_vec);;
            sret.label = labels_base[di];
            searchRet.push_back(sret);
        }

        // sort search result by real distance then label value
        std::sort(searchRet.begin(), searchRet.end(), cmp);

        // return top k value with minimize real distance
        for (int i = 0; i < k; i++) {
            distances[i] = searchRet[i].distance;
            labels[i] = searchRet[i].label;
        }
    }

    int IndexIVF_HNSW_Grouping::prepare_db()
    {
        int rc = 0;

        if (db_p == nullptr) {
            std::cout << "Connect to servicedb ..." << std::endl;
            rc = setup_db("localhost", 5432, "servicedb", "postgres", "postgres");
        }
        return rc;
    }

    int IndexIVF_HNSW_Grouping::create_new_batch(size_t batch_idx)
    {
        return db_p->CreateBatch(batch_idx);
    }

    int IndexIVF_HNSW_Grouping::write(const char *path_index, bool do_trunc)
    {
        std::ofstream output;
        int rc = 0;

        try {
            if (do_trunc)
                output.open(path_index, std::ios::binary | std::ios::trunc);
            else
                output.open(path_index, std::ios::binary);

            write_variable(output, d);
            write_variable(output, nc);
            write_variable(output, nsubc);

            // Save vector indices
            for (size_t i = 0; i < nc; i++)
                write_vector(output, ids[i]);

            // Save PQ codes
            for (size_t i = 0; i < nc; i++)
                write_vector(output, codes[i]);

            // Save norm PQ codes
            for (size_t i = 0; i < nc; i++)
                write_vector(output, norm_codes[i]);

            // Save NN centroid indices
            for (size_t i = 0; i < nc; i++)
                write_vector(output, nn_centroid_idxs[i]);

            // Write group sizes
            for (size_t i = 0; i < nc; i++)
                write_vector(output, subgroup_sizes[i]);

            // Save alphas
            write_vector(output, alphas);

            // Save centroid norms
            // centroid norms count is nc
            write_vector(output, centroid_norms);

            // Save inter centroid distances
            for (size_t i = 0; i < nc; i++)
                write_vector(output, inter_centroid_dists[i]);
        } catch (...) {
            std::cout << "Error when write index file: " << path_index << std::endl;
            rc = -1;
        }

        if (output.is_open()) output.close();
        if (rc) unlink(path_index);

        return rc;
    }

    int IndexIVF_HNSW_Grouping::write(const char *path_index)
    {
        return this->write(path_index, false);
    }

    int IndexIVF_HNSW_Grouping::read(const char *path_index)
    {
        int rc = 0;
        std::ifstream input;

        try {
            input.open(path_index, std::ios::binary);

            read_variable(input, d);
            read_variable(input, nc);
            read_variable(input, nsubc);

            // Read ids
            for (size_t i = 0; i < nc; i++)
                read_vector(input, ids[i]);

            // Read PQ codes
            for (size_t i = 0; i < nc; i++)
                read_vector(input, codes[i]);

            // Read norm PQ codes
            for (size_t i = 0; i < nc; i++)
                read_vector(input, norm_codes[i]);

            // Read NN centroid indices
            for (size_t i = 0; i < nc; i++)
                read_vector(input, nn_centroid_idxs[i]);

            // Read group sizes
            for (size_t i = 0; i < nc; i++)
                read_vector(input, subgroup_sizes[i]);

            // Read alphas
            read_vector(input, alphas);

            // Read centroid norms
            read_vector(input, centroid_norms);

            // Read inter centroid distances
            for (size_t i = 0; i < nc; i++)
                read_vector(input, inter_centroid_dists[i]);
        } catch (...) {
            std::cout << "Error when read index file: " << path_index << std::endl;
            rc = -1;
        }

        if (input.is_open()) input.close();

        return rc;
    }


    void IndexIVF_HNSW_Grouping::train_pq(size_t n, const float *x)
    {
        std::vector<float> train_subcentroids;
        std::vector<float> train_residuals;

        train_subcentroids.reserve(n*d);
        train_residuals.reserve(n*d);

        std::vector<idx_t> assigned(n);
        assign(n, x, assigned.data());

        std::unordered_map<idx_t, std::vector<float>> group_map;

        for (size_t i = 0; i < n; i++) {
            const idx_t key = assigned[i];
            for (size_t j = 0; j < d; j++)
                group_map[key].push_back(x[i*d + j]);
        }

        // Train Residual PQ
        std::cout << "Training Residual PQ codebook " << std::endl;
        for (auto group : group_map) {
            const idx_t centroid_idx = group.first;
            const float *centroid = quantizer->getDataByInternalId(centroid_idx);
            const std::vector<float> data = group.second;
            const int group_size = data.size() / d;

            std::vector<idx_t> nn_centroid_idxs(nsubc);
            std::vector<float> centroid_vector_norms(nsubc);
            auto nn_centroids_raw = quantizer->searchKnn(centroid, nsubc + 1);

            while (nn_centroids_raw.size() > 1) {
                centroid_vector_norms[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
                nn_centroid_idxs[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
                nn_centroids_raw.pop();
            }

            // Compute centroid-neighbor_centroid and centroid-group_point vectors
            std::vector<float> centroid_vectors(nsubc * d);
            for (size_t subc = 0; subc < nsubc; subc++) {
                const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idxs[subc]);
                faiss::fvec_madd(d, nn_centroid, -1., centroid, centroid_vectors.data() + subc * d);
            }

            // Find alphas for vectors
            const float alpha = compute_alpha(centroid_vectors.data(), data.data(), centroid,
                                              centroid_vector_norms.data(), group_size);

            // Compute final subcentroids 
            std::vector<float> subcentroids(nsubc * d);
            for (size_t subc = 0; subc < nsubc; subc++)
                faiss::fvec_madd(d, centroid, alpha, centroid_vectors.data() + subc*d, subcentroids.data() + subc*d);

            // Find subcentroid idx
            std::vector<idx_t> subcentroid_idxs(group_size);
            compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), group_size);

            // Compute Residuals
            std::vector<float> residuals(group_size * d);
            compute_residuals(group_size, data.data(), residuals.data(), subcentroids.data(), subcentroid_idxs.data());

            for (size_t i = 0; i < group_size; i++) {
                const idx_t subcentroid_idx = subcentroid_idxs[i];
                for (size_t j = 0; j < d; j++) {
                    train_subcentroids.push_back(subcentroids[subcentroid_idx*d + j]);
                    train_residuals.push_back(residuals[i*d + j]);
                }
            }
        }
        // Train OPQ rotation matrix and rotate residuals
        if (do_opq) {
            faiss::OPQMatrix *matrix = new faiss::OPQMatrix(d, pq->M);

            std::cout << "Training OPQ Matrix" << std::endl;
	    std::cout << "Max Training Points: " << n << std::endl;
	    std::cout << "d: " << d << " M: " << pq->M << std::endl;
            matrix->verbose = true;
            matrix->max_train_points = n;
            matrix->niter = 100;
            matrix->train(n, train_residuals.data());
            opq_matrix = matrix;
	    std::cout << "Training OPQ Matrix Done" << std::endl;

            std::vector<float> copy_train_residuals(n * d);
            memcpy(copy_train_residuals.data(), train_residuals.data(), n * d * sizeof(float));
            opq_matrix->apply_noalloc(n, copy_train_residuals.data(), train_residuals.data());
        }

        printf("Training %zdx%zd PQ on %ld vectors in %dD\n", pq->M, pq->ksub, train_residuals.size() / d, d);
        pq->verbose = true;
        pq->train(n, train_residuals.data());

        // Norm PQ
        std::cout << "Training Norm PQ codebook " << std::endl;
        std::vector<float> train_norms;
        const float *residuals = train_residuals.data();
        const float *subcentroids = train_subcentroids.data();

        for (auto p : group_map) {
            const std::vector<float> data = p.second;
            const size_t group_size = data.size() / d;

            // Compute Codes 
            std::vector<uint8_t> xcodes(group_size * code_size);
            pq->compute_codes(residuals, xcodes.data(), group_size);

            // Decode Codes 
            std::vector<float> decoded_residuals(group_size * d);
            pq->decode(xcodes.data(), decoded_residuals.data(), group_size);

            // Reverse rotation
            if (do_opq){
                std::vector<float> copy_decoded_residuals(group_size * d);
                memcpy(copy_decoded_residuals.data(), decoded_residuals.data(), group_size * d * sizeof(float));
                opq_matrix->transform_transpose(group_size, copy_decoded_residuals.data(), decoded_residuals.data());
            }

            // Reconstruct Data 
            std::vector<float> reconstructed_x(group_size * d);
            for (size_t i = 0; i < group_size; i++)
                faiss::fvec_madd(d, decoded_residuals.data() + i*d, 1., subcentroids+i*d, reconstructed_x.data() + i*d);

            // Compute norms 
            std::vector<float> group_norms(group_size);
            faiss::fvec_norms_L2sqr(group_norms.data(), reconstructed_x.data(), d, group_size);

            for (size_t i = 0; i < group_size; i++)
                train_norms.push_back(group_norms[i]);

            residuals += group_size * d;
            subcentroids += group_size * d;
        }
        printf("Training %zdx%zd PQ on %ld vectors in 1D\n", norm_pq->M, norm_pq->ksub, train_norms.size());
        norm_pq->verbose = true;
        norm_pq->train(n, train_norms.data());
    printf("done xxxxxxxxxxxxxxxxx\n");
    }

    void IndexIVF_HNSW_Grouping::compute_inter_centroid_dists()
    {
        for (size_t i = 0; i < nc; i++) {
            const float *centroid = quantizer->getDataByInternalId(i);
            inter_centroid_dists[i].resize(nsubc);
            for (size_t subc = 0; subc < nsubc; subc++) {
                const idx_t nn_centroid_idx = nn_centroid_idxs[i][subc];
                const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idx);
                inter_centroid_dists[i][subc] = fvec_L2sqr(nn_centroid, centroid, d);
            }
        }
    }

    void IndexIVF_HNSW_Grouping::dump_inter_centroid_dists(char *path)
    {
        FILE *fp;
        char buf[512];

        fp = fopen(path, "w");
        if (fp == NULL) {
            std::cout << "Failed to open file: " << path << std::endl;
            return;
        }

        for (size_t i = 0; i < nc; i++) {
            for (size_t subc = 0; subc < nsubc; subc++) {
                sprintf(buf, "distance of centriod %lu to centriod %lu is %f\n",
                        i, subc, inter_centroid_dists[i][subc]);
                fwrite(buf, 1, strlen(buf), fp);
            }
        }

        fclose(fp);
    }

    void IndexIVF_HNSW_Grouping::compute_residuals(size_t n, const float *x, float *residuals,
                                                   const float *subcentroids, const idx_t *keys)
    {
        for (size_t i = 0; i < n; i++) {
            const float *subcentroid = subcentroids + keys[i]*d;
            faiss::fvec_madd(d, x + i*d, -1., subcentroid, residuals + i*d);
        }
    }

    void IndexIVF_HNSW_Grouping::reconstruct(size_t n, float *x, const float *decoded_residuals,
                                             const float *subcentroids, const idx_t *keys)
    {
        for (size_t i = 0; i < n; i++) {
            const float *subcentroid = subcentroids + keys[i] * d;
            faiss::fvec_madd(d, decoded_residuals + i*d, 1., subcentroid, x + i*d);
        }
    }

    void IndexIVF_HNSW_Grouping::compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                                          const float *x, size_t group_size)
    {
        for (size_t i = 0; i < group_size; i++) {
            float min_dist = 0.0;
            idx_t min_idx = -1;
            for (size_t subc = 0; subc < nsubc; subc++) {
                const float *subcentroid = subcentroids + subc * d;
                float dist = fvec_L2sqr(subcentroid, x + i*d, d);
                if (min_idx == -1 || dist < min_dist){
                    min_dist = dist;
                    min_idx = subc;
                }
            }
            subcentroid_idxs[i] = min_idx;
        }
    }

    float IndexIVF_HNSW_Grouping::compute_alpha(const float *centroid_vectors, const float *points,
                                                const float *centroid, const float *centroid_vector_norms_L2sqr,
                                                size_t group_size)
    {
        float group_numerator = 0.0;
        float group_denominator = 0.0;

        std::vector<float> point_vectors(group_size * d);
        for (size_t i = 0; i < group_size; i++)
            faiss::fvec_madd(d, points + i*d , -1., centroid, point_vectors.data() + i*d);

        for (size_t i = 0; i < group_size; i++) {
            const float *point_vector = point_vectors.data() + i * d;
            const float *point = points + i * d;

            std::priority_queue<std::pair<float, std::pair<float, float>>> maxheap;

            for (size_t subc = 0; subc < nsubc; subc++) {
                const float *centroid_vector = centroid_vectors + subc * d;

                float numerator = faiss::fvec_inner_product(centroid_vector, point_vector, d);
                numerator = (numerator > 0) ? numerator : 0.0;

                const float denominator = centroid_vector_norms_L2sqr[subc];
                const float alpha = numerator / denominator;

                std::vector<float> subcentroid(d);
                faiss::fvec_madd(d, centroid, alpha, centroid_vector, subcentroid.data());

                const float dist = fvec_L2sqr(point, subcentroid.data(), d);
                maxheap.emplace(-dist, std::make_pair(numerator, denominator));
            }

            group_numerator += maxheap.top().second.first;
            group_denominator += maxheap.top().second.second;
        }
        return (group_denominator > 0) ? group_numerator / group_denominator : 0.0;
    }

    bool IndexIVF_HNSW_Grouping::action_on_pq(char *path_model, size_t pq_ver, bool with_opq, ACTION_PQ action)
    {
        char path_ver[1024], path_full[1024];
        sprintf(path_ver, "%s/%lu", path_model, pq_ver);

        sprintf(path_full, "%s/pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
        if (action == ACTION_PQ::PQ_CHECK) {
            if (!exists(path_full)) return false;
        } else {
            unlink(path_full);
        }

        sprintf(path_full, "%s/norm_pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
        if (action == ACTION_PQ::PQ_CHECK) {
            if (!exists(path_full)) return false;
        } else {
            unlink(path_full);
        }

        if (with_opq) {
            sprintf(path_full, "%s/matrix_pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
            if (action == ACTION_PQ::PQ_CHECK) {
                if (!exists(path_full)) return false;
            } else {
                unlink(path_full);
            }
        }

        return true;
    }

    int IndexIVF_HNSW_Grouping::build_pq_files(system_conf_t &sys_conf, pq_conf_t &pq_conf)
    {
        char path_learn[1024], path_model[1024];

        sprintf(path_learn, "%s/%s", sys_conf.path_base_data, "bigann_learn.bvecs");
        strcpy(path_model, sys_conf.path_base_model);

        return build_pq_files(path_learn, path_model, pq_conf.ver, pq_conf.with_opq, 
                              sys_conf.code_size, pq_conf.nt, pq_conf.nsubt, 
                              sys_conf.nsubc);
    }

    int IndexIVF_HNSW_Grouping::build_pq_files(const char *path_learn, const char *path_model,
                                               size_t pq_ver,
                                               bool with_opq, size_t code_size,
                                               size_t n_train, size_t n_sub_train, size_t nsubc)
    {
        int rc = 0;
        char path_full[1024];
        char path_ver[1024];
        uint32_t dim;
        size_t nvecs;
        std::vector<float> trainvecs;
        std::vector<float> trainvecs_rnd_subset;

        if (pq_ver == 0) {
            std::cout << "Invalid PQ version 0" << std::endl;
            return -1;
        }

        // Prepare output directory for store PQ files
        sprintf(path_ver, "%s/%lu", path_model, pq_ver);
        if (!exists(path_ver)) {
            if (mkdir_p(path_ver, 0755)) {
                std::cout << "Failed to create directory: " << path_ver << std::endl;
                return rc;
            }
        }

        StopW stopw = StopW();
        std::cout << "Build PQ files based on learning set file: " << path_learn << std::endl;
        try {
            rc = get_vec_attr(path_learn, dim, nvecs);
            if (rc) return rc;

            if (nvecs < n_train) {
                std::cout << "No enough train vector in file: " << path_learn << std::endl;
                return -1;
            } else {
                nvecs = n_train;
            }

            trainvecs.resize(nvecs * dim);
            {
                 std::ifstream learn_input(path_learn, std::ios::binary);
                 readXvecFvec<uint8_t>(learn_input, trainvecs.data(), dim, nvecs);
            }
            std::cout << "Get train vector count: " << nvecs << std::endl;
        } catch (...) {
            std::cout << "Failed to get train vector" << std::endl;
            goto err_handle;
        }


        try {
            // Set Random Subset of sub_nt trainvecs
            std::cout << "Get random subset with vector numnber of: " << n_sub_train << std::endl;
            trainvecs_rnd_subset.resize(n_sub_train * dim);
            random_subset(trainvecs.data(), trainvecs_rnd_subset.data(), dim, nvecs, n_sub_train);
            std::cout << "Done with " << (stopw.getElapsedTimeMicro() / 1000000.0) << "s" << std::endl;
        } catch (...) {
            std::cout << "Failed to get random subset from train vector" << std::endl;
            goto err_handle;
        }

        try {
            stopw.reset();
            std::cout << "Training PQ codebooks ..." << std::endl;
            train_pq(nvecs, trainvecs_rnd_subset.data());
            std::cout << "Done with " << (stopw.getElapsedTimeMicro() / 1000000.0) << "s" << std::endl;
        } catch (...) {
            std::cout << "Failed to training PQ codebooks" << std::endl;
            goto err_handle;
        }

        try {
            sprintf(path_full, "%s/pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
            std::cout << "Saving Residual PQ codebook to " << path_full << std::endl;
            faiss::write_ProductQuantizer(pq, path_full);

            sprintf(path_full, "%s/norm_pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
            std::cout << "Saving Norm PQ codebook to " << path_full<< std::endl;
            faiss::write_ProductQuantizer(norm_pq, path_full);

            if (with_opq) {
                sprintf(path_full, "%s/matrix_pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
                std::cout << "Saving Residual OPQ rotation matrix to " << path_full<< std::endl;
                faiss::write_VectorTransform(opq_matrix, path_full);
            }
        } catch (...) {
            std::cout << "Failed to write PQ codebooks" << std::endl;
            goto err_handle;
        }

        goto out;

err_handle:
        {
            rc = -1;
            std::cout << "Failed when build PQ files" << std::endl;

            sprintf(path_full, "%s/pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
            unlink(path_full);
            sprintf(path_full, "%s/norm_pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
            unlink(path_full);
            if (with_opq) {
                sprintf(path_full, "%s/matrix_pq%lu_nsubc%lu.opq", path_ver, code_size, nsubc);
                unlink(path_full);
            }
        }

out:
        return rc;
    }

    int IndexIVF_HNSW_Grouping::append_pq_info(system_conf_t &sys_conf, pq_conf_t &pq_conf)
    {
        int rc;

        rc = prepare_db();
        if (rc) {
            std::cout << "Failed to prepare database" << std::endl;
            return rc;
        }

        return db_p->AppendPQInfo(pq_conf.ver, pq_conf.with_opq, sys_conf.code_size, sys_conf.nsubc);
    }

    // TODO: I/O error handler when read PQ codebooks from file
    // or force core dump when failed to read pq codebooks
    // to my understanding core dump is ok
    int IndexIVF_HNSW_Grouping::load_pq_codebooks(system_conf_t &sys_conf, pq_conf_t &pq_conf)
    {
        char path_full[1024];

        std::cout << "Load latest PQ files with version: " << pq_conf.ver << std::endl;

        // get PQ codebook path
        get_path_pq(sys_conf, pq_conf.ver, path_full);
        std::cout << "Loading Residual PQ codebook from " << path_full << std::endl;
        try {
            if (pq)
                delete pq;
            pq = faiss::read_ProductQuantizer(path_full);
            std::cout << "Finish Load Residual PQ codebook: " << path_full << std::endl;
        } catch (...) {
            std::cout << "Failed to Load Residual PQ codebook: " << path_full << std::endl;
            return -1;
        }


        if (pq_conf.with_opq) {
            // get OPQ rotation matrix path
            get_path_opq_matrix(sys_conf, pq_conf.ver, path_full);
            std::cout << "Loading Residual OPQ rotation matrix from " << path_full << std::endl;
            try {
                opq_matrix = dynamic_cast<faiss::LinearTransform *>(faiss::read_VectorTransform(path_full));
                std::cout << "Finish Load Residual OPQ rotation matrix: " << path_full << std::endl;
            } catch (...) {
                std::cout << "Failed to Load Residual OPQ rotation matrix: " << path_full << std::endl;
                return -1;
            }
        }

        // get norm PQ codebook path
        get_path_norm_pq(sys_conf, pq_conf.ver, path_full);
        std::cout << "Loading Norm PQ codebook from " << path_full << std::endl;
        try {
            if (norm_pq)
                delete norm_pq;
            norm_pq = faiss::read_ProductQuantizer(path_full);
            std::cout << "Finsh Load Norm PQ codebook: " << path_full << std::endl;
        } catch (...) {
            std::cout << "Failed to Load Norm PQ codebook: " << path_full << std::endl;
            return -1;
        }

        return 0;
    }

    /*
     * load quantizer into index
     *
     *  @param  sys_conf   system_orca table's configuration
     *  @param  pq_conf    pq_conf table's record (a version of pq's config)
     *
     * two case:
     *
     * 1:  centroids file, info file, edges file exists
     *     Load file content, and build quantizer in memory
     *
     * 2: info file, edges file not exists
     *     build quantizer in memory with centroids file
     *     and generate info file and edges file
     *
    */
    int IndexIVF_HNSW_Grouping::load_quantizer(system_conf_t &sys_conf, pq_conf_t &pq_conf)
    {
        char path_centroids[1024], path_info[1024], path_edges[1024];

        get_path_centroids(sys_conf, path_centroids);
        get_path_info(sys_conf, pq_conf, path_info);
        get_path_edges(sys_conf, pq_conf, path_edges);

        // TODO: mocify following function, force core dump when failed to build quantizer
        build_quantizer(path_centroids, path_info, path_edges, pq_conf.M, pq_conf.efConstruction);
    }

    int IndexIVF_HNSW_Grouping::build_prcomputed_index(system_conf_t &sys_conf, size_t skip_batch)
    {
        std::vector<batch_info_t> batch_list;
        char path_vector[1024], path_precomputed_idx[1024];
        int rc;

        rc = db_p->GetBatchList(batch_list);
        size_t sz = batch_list.size();
        for (auto i = 0; i < sz; i++) {
            auto a_batch = batch_list[i];

            if (a_batch.batch == skip_batch)
                continue;

            if (a_batch.valid == false || a_batch.precomputed_idx == true)
                continue;

            get_path_vector(sys_conf, a_batch.batch, path_vector);
            get_path_precomputed_idx(sys_conf, a_batch.batch, path_precomputed_idx);
            rc = build_prcomputed_index(path_vector, path_precomputed_idx);
            if (rc) {
                std::cout << "Failed to build precomputing index for batch: " <<  a_batch.batch << std::endl;
                return rc;
            }
        }

        return 0;
    }

    int IndexIVF_HNSW_Grouping::build_prcomputed_index(const char *path_base, const char *path_prcomputed_index)
    {
        uint32_t dim;
        size_t nvecs;
        int rc;

        rc = get_vec_attr(path_base, dim, nvecs);
        if (rc) return rc;

        std::cout << "Build precomputing indices" << std::endl;
        StopW stopw = StopW();

        std::ifstream input(path_base, std::ios::binary);
        std::ofstream output(path_prcomputed_index, std::ios::binary);

        if (!input.is_open()) {
            std::cout << "Failed to open base vector file: " << path_base << std::endl;
            return -1;
        }

        if (!output.is_open()) {
            std::cout << "Failed to open prcomputed_index file: " << path_prcomputed_index << std::endl;
            return -1;
        }

        const uint32_t batch_size_max = 1000000;
        uint32_t batch_size = batch_size_max;
        if (nvecs < batch_size_max) batch_size = nvecs;

        const size_t nbatches = nvecs / batch_size;

        /*
         * get correct batch size, we cannot ensure vector number in
         * file always multiple of batch_size_max as in test code
         */
        std::vector<size_t> batch_size_list(nbatches);
        for (size_t i = 0; i < nbatches; i++) {
            if (i != (nbatches - 1)) {
                batch_size_list[i] = batch_size;
            } else {
                batch_size_list[i] = nvecs - i*batch_size;
            }
        }

        std::vector<float> batch(batch_size_max * dim);
        std::vector<idx_t> precomputed_idx(batch_size_max);

        // TODO: if we should let efSearch as a function parameter ?
        quantizer->efSearch = 220;
        for (size_t i = 0; i < nbatches; i++) {
            // get current batch size
            batch_size = batch_size_list[i];

            if (i % 10 == 0) {
                std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                          << (100.*i) / nbatches << "%" << std::endl;
            }

            try {
                readXvecFvec<uint8_t>(input, batch.data(), dim, batch_size);
                assign(batch_size, batch.data(), precomputed_idx.data());

                output.write((char *) &batch_size, sizeof(uint32_t));
                output.write((char *) precomputed_idx.data(), batch_size * sizeof(idx_t));
            } catch (...) {
                rc = -1;
                break;
            }
        }

        return rc;
    }

    int IndexIVF_HNSW_Grouping::add_one_batch_to_index(const char *path_vector,
                                                     const char *path_precomputed_idx)
    {
        const size_t batch_size_max = 1000000;
        size_t groups_per_iter = 250000;
        std::vector<uint8_t> batch(batch_size_max * d);
        std::vector<idx_t> idx_batch(batch_size_max);
        uint32_t batch_size, dim;
        size_t nbatches, nvecs;
        int rc;

        // get vector number in the batch of vector file
        rc = get_vec_attr(path_vector, dim, nvecs);
        if (rc) return rc;

        std::cout << "Build index for vector file: " << path_vector << std::endl;
        StopW stopw = StopW();

        batch_size = batch_size_max;
        if (nvecs < batch_size_max) batch_size = nvecs;
        nbatches = nvecs / batch_size;

        // fix behavior when total vector record number in file smaller than groups_per_iter
        if (nvecs < groups_per_iter) groups_per_iter = nvecs;

        /*
         * not all batch have same vector number
         * although we try to let every batch file has same vector number
         * but, when server crash, we will start with a new batch
         * which will lead previous last batch not have enough vector data
         * following code used to get correct batch size
         *
         */
        std::vector<size_t> batch_size_list(nbatches);
        for (size_t i = 0; i < nbatches; i++) {
            if (i != (nbatches - 1)) {
                batch_size_list[i] = batch_size;
            } else {
                batch_size_list[i] = nvecs - i*batch_size;
            }
        }

        for (size_t ngroups_added = 0; ngroups_added < nvecs; ngroups_added += groups_per_iter)
        {
            std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                      << ngroups_added << " / " << nvecs << std::endl;

            std::vector<std::vector<uint8_t>> data(groups_per_iter);
            std::vector<std::vector<idx_t>> ids(groups_per_iter);

            // Iterate through the dataset extracting points from groups,
            // whose ids lie in [ngroups_added, ngroups_added + groups_per_iter)
            std::ifstream base_input(path_vector, std::ios::binary);
            std::ifstream idx_input(path_precomputed_idx, std::ios::binary);

            for (size_t b = 0; b < nbatches; b++) {
                // get correct batch_size value, because
                batch_size = batch_size_list[b];

                readXvec<uint8_t>(base_input, batch.data(), d, batch_size);
                readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);

                for (size_t i = 0; i < batch_size; i++) {
                    if (idx_batch[i] < ngroups_added ||
                        idx_batch[i] >= ngroups_added + groups_per_iter)
                        continue;

                    idx_t idx = idx_batch[i] % groups_per_iter;
                    for (size_t j = 0; j < d; j++)
                        data[idx].push_back(batch[i * d + j]);
                    ids[idx].push_back(b * batch_size + i);
                }
            }

            /*
             * If <opt.nc> is not a multiple of groups_per_iter,
             * change <groups_per_iter> on the last iteration
             */
            if (nc - ngroups_added <= groups_per_iter)
                groups_per_iter = nc - ngroups_added;

            size_t j = 0;
            #pragma omp parallel for
            for (size_t i = 0; i < groups_per_iter; i++) {
                #pragma omp critical
                {
                    if (j % 10000 == 0) {
                        std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                                  << (100. * (ngroups_added + j)) / nc << "%" << std::endl;
                    }
                    j++;
                }
                const size_t group_size = ids[i].size();
                std::vector<float> group_data(group_size * d);
                // Convert bytes to floats
                for (size_t k = 0; k < group_size * d; k++)
                    group_data[k] = 1. *data[i][k];

                add_group(ngroups_added + i, group_size, group_data.data(), ids[i].data());
            }
        }
    }

    void IndexIVF_HNSW_Grouping::get_path_info(const system_conf_t sys_conf, const pq_conf_t pq_conf, char *path_index)
    {
        sprintf(path_index, "%s/hnsw_M%lu_ef%lu.bin", sys_conf.path_base_model, pq_conf.M, pq_conf.efConstruction);
    }
    void IndexIVF_HNSW_Grouping::get_path_edges(const system_conf_t sys_conf, const pq_conf_t pq_conf, char *path_index)
    {
        sprintf(path_index, "%s/hnsw_M%lu_ef%lu.ivecs", sys_conf.path_base_model, pq_conf.M, pq_conf.efConstruction);
    }

    void IndexIVF_HNSW_Grouping::get_path_centroids(const system_conf_t sys_conf, char *path_index) {
        sprintf(path_index, "%s/centroids_sift1b.fvecs", sys_conf.path_base_data);
    }

    void IndexIVF_HNSW_Grouping::get_path_index(const system_conf_t sys_conf, const size_t idx_ver, char* path_index)
    {
        sprintf(path_index, "%s/%lu/ivfhnsw_OPQ%lu_nsubc%lu_%lu.index",
                sys_conf.path_base_model, idx_ver, code_size, nsubc, idx_ver);
    }

    void IndexIVF_HNSW_Grouping::get_path_pq(const system_conf_t sys_conf, const size_t idx_ver, char *path_index) {
        sprintf(path_index, "%s/%lu/pq%lu_nsubc%lu.opq",
                sys_conf.path_base_model, idx_ver, code_size, nsubc);
    }

    void IndexIVF_HNSW_Grouping::get_path_opq_matrix(const system_conf_t sys_conf, const size_t idx_ver, char *path_index) {
        sprintf(path_index, "%s/%lu/matrix_pq%lu_nsubc%lu.opq",
                sys_conf.path_base_model, idx_ver, code_size, nsubc);
    }

    void IndexIVF_HNSW_Grouping::get_path_norm_pq(const system_conf_t sys_conf, const size_t idx_ver, char *path_index) {
        sprintf(path_index, "%s/%lu/norm_pq%lu_nsubc%lu.opq",
                sys_conf.path_base_model, idx_ver, code_size, nsubc);
    }

    void IndexIVF_HNSW_Grouping::get_path_vector(const system_conf_t sys_conf, const size_t batch_idx, char *path_vector) {
        // TODO: current code assume maximize batch index number is 100
        // change format 02lu to 03lu etc. when use bigger batch index number
        sprintf(path_vector, "%s/bigann_base_%02lu.bvecs",
                sys_conf.path_base_model, batch_idx);
    }

    void IndexIVF_HNSW_Grouping::get_path_precomputed_idx(const system_conf_t sys_conf, 
                                                          const size_t batch_idx,
                                                          char* path_precomputed_idx )
    {
        // TODO: current code assume maximize batch index number is 100
        // change format 02lu to 03lu etc. when use bigger batch index number
        sprintf(path_precomputed_idx,
                "%s/precomputed_idxs_sift1b_%02lu.ivecs",
                sys_conf.path_base_data,
                batch_idx);
    }

    int IndexIVF_HNSW_Grouping::save_index(const system_conf_t sys_conf, const size_t idx_ver)
    {
        char path_index[1024], path_index_dir[1024], *ptr;
        int rc;

        // Computing centroid norms and inter-centroid distances
        std::cout << "Computing centroid norms"<< std::endl;
        compute_centroid_norms();
        std::cout << "Computing centroid dists"<< std::endl;
        compute_inter_centroid_dists();

        // Save index, always truncate file
        get_path_index(sys_conf, idx_ver, path_index);
        strcpy(path_index_dir, path_index);
        ptr = rindex(path_index_dir, '/');
        *ptr = '\0';
        rc = mkdir_p(path_index_dir, 0755);
        if (rc) {
            std::cout << "Failed to create directory: " << path_index_dir << std::endl;
            return rc;
        }
        std::cout << "Saving index to " << path_index << std::endl;

        return write(path_index, true);
    }

    int IndexIVF_HNSW_Grouping::build_batchs_to_index(const system_conf_t &sys_conf, std::vector<batch_info_t> &batch_list)
    {
        std::vector<size_t> b_list(0);
        auto sz = b_list.size();
        for (size_t i = 0; i < sz; i++) {
            if (batch_list[i].precomputed_idx == false) {
                char path_vector[1024], path_precomputed_idx[1024];
                // batch vector has no precomputed index, build it first
                get_path_vector(sys_conf, batch_list[i].batch, path_vector);
                get_path_precomputed_idx(sys_conf, batch_list[i].batch, path_vector);
                std::cout << "Build precomputed index for vector of batch: " << batch_list[i].batch << std::endl;
                int rc = build_prcomputed_index(path_vector, path_precomputed_idx);
            }
            b_list.push_back(batch_list[i].batch);
        }

        return build_batchs_to_index(sys_conf, b_list);
    }

    int IndexIVF_HNSW_Grouping::build_batchs_to_index(const system_conf_t &sys_conf, std::vector<size_t> &batch_list)
    {
        char path_vector[1024];
        char path_precomputed_idx[1024];
        int  rc;

        auto sz = batch_list.size();
        for (size_t i = i; i < sz; i++) {
            // get vector file path and precomputed index file path
            sprintf(path_vector, "%s/bigann_base_%lu.bvecs", sys_conf.path_base_data, batch_list[i]);
            sprintf(path_precomputed_idx, "%s/precomputed_idxs_%lu.ivecs", sys_conf.path_base_data, batch_list[i]);

            rc = add_one_batch_to_index(path_vector, path_precomputed_idx);
            if (rc) {
                std::cout << "Failed to add vector from file: " << path_vector << " to index" << std::endl;
                return -1;
            }
        }

        return 0;
    }

    int IndexIVF_HNSW_Grouping::build_batchs_to_index(const system_conf_t &sys_conf,
                                            const size_t batch_begin,
                                            const size_t batch_end)
    {
        std::cout << "Build index for vector from batch: " << batch_begin << " to batch: " << batch_end << std::endl;
        std::vector<size_t> batch_list(0);
        for (size_t i = batch_begin; i <= batch_end; i++)
            batch_list.push_back(i);

        return build_batchs_to_index(sys_conf, batch_list);
    }

    int IndexIVF_HNSW_Grouping::build_index(const system_conf_t sys_conf,
                                            const size_t batch_begin,
                                            const size_t batch_end,
                                            const size_t index_ver)
    {
        int rc;
        rc = build_batchs_to_index(sys_conf, batch_begin, batch_end);
        if (rc) {
            // don't need to output message, already have in calling function
            return rc;
        }
        rc = save_index(sys_conf, index_ver);
        if (rc) {
            // don't need to output message, already have in calling function
            return rc;
        }
        rc = db_p->AppendIndexInfo(index_ver, batch_begin, batch_end);
        if (!rc) {
            std::cout << "Success build index for batch from " << batch_begin << " to " << batch_end << std::endl;
        } else {
            std::cout << "Failed to build index for batch from " << batch_begin << " to " << batch_end << std::endl;
        }

        return rc;
    }
}
