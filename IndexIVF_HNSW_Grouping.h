#ifndef IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
#define IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H

#include "IndexIVF_HNSW.h"

namespace ivfhnsw {
// used for info tracing
#define TRACE_CENTROIDS

    // util function for centriod trace
    extern int centriodTraceSetup();
    extern void centriodTraceClose();

    enum class ACTION_PQ {PQ_CHECK, PQ_CLEANUP};

    //=======================================
    // IVF_HNSW + Grouping( + Pruning) index
    //=======================================
    struct IndexIVF_HNSW_Grouping: IndexIVF_HNSW
    {
        size_t nsubc;         ///< Number of sub-centroids per group
        bool do_pruning;      ///< Turn on/off pruning

        std::vector<std::vector<idx_t> > nn_centroid_idxs;    ///< Indices of the <nsubc> nearest centroids for each centroid
        std::vector<std::vector<idx_t> > subgroup_sizes;      ///< Sizes of sub-groups for each group
        std::vector<float> alphas;    ///< Coefficients that determine the location of sub-centroids

    private:
        Index_DB *db_p = nullptr;
        bool db_free = false;

        system_conf_t sys_conf;
        std::vector<batch_info_t> batch_list;
    public:
        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids);

        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids, Index_DB *db_ref);

        ~IndexIVF_HNSW_Grouping();

        int load_sys_conf();

        /** Add <group_size> vectors of dimension <d> from the <group_idx>-th group to the index.
          *
          * @param group_idx         index of the group
          * @param group_size        number of base vectors in the group
          * @param x                 base vectors to add (size: group_size * d)
          * @param ids               ids to store for the vectors (size: groups_size)
        */
        void add_group(size_t group_idx, size_t group_size, const float *x, const idx_t *ids);

        /*
         * @param  k             number of the closest vertices to search, ie. how many result need to return
         * @param  query         query vector used to search
         * @param  distances     distances between query vector and points (vectors in search result)
         * @param  labels        order no when add vector into index, begin from 0
         * @param  path_base     path of base vector data file
         *
         */
        void search(size_t k, const float *query, float *distances, long *labels);

        int search(size_t k, const float* query, std::vector<size_t> &id_vectors);
        int search(size_t k, const uint8_t* query, std::vector<size_t>& id_vectors);

        // apply disk search based on ANN search result
        void searchDisk(size_t k, const float *query, float *distances, long *labels, const char *path_base);

        int write(const char *path_index);
        int read(const char *path_index);

        // setup database related things
        int setup_db(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr);

        int prepare_db();

        // allocate a new batch to store new vector
        int create_new_batch(size_t batch_idx, size_t vector_id);

        // commit batch index status
        int commit_batch(size_t batch_idx);

        void train_pq(size_t n, const float *x);

        /// Compute distances between the group centroid and its <subc> nearest neighbors in the HNSW graph
        void compute_inter_centroid_dists();

        /// Write distance between centroids to file given by path
        void dump_inter_centroid_dists(char *path);

        /*
         * Build PQ files
         *
         * @param path_learn    learn vector file full path
         * @param path_out      directory to store PQ files generated
         * @param pq_ver        version of PQ
         * @param with_opq      enable opq or not
         * @param code_size     code size per vector in bytes
         * @param n_train       number of learn vectors
         * @param n_sub_train   number of learn vectors to train (random subset of the learn set)
         * @param nsubc         number of subcentroids per group
         *
         */
        int build_pq_files(const char *path_learn, const char *path_out,
                           size_t pq_ver, bool with_opq, size_t code_size,
                           size_t n_train, size_t n_sub_train, size_t nsubc);

        /*
         * build PQ codebooks
         *
         *  @param  sys_conf   system_orca table's configuration
         *  @param  pq_conf    pq_conf table's record (a version of pq's config)
         *
         */
        int build_pq_files(system_conf_t &sys_conf, pq_conf_t &pq_conf);

        int append_pq_info(system_conf_t &sys_conf, pq_conf_t &pq_conf);

        bool action_on_pq(char *path_out, size_t pq_ver, bool with_opq, ACTION_PQ action);

        /*
         * load PQ codebooks into index
         *
         *  @param  sys_conf   system_orca table's configuration
         *  @param  pq_conf    pq_conf table's record (a version of pq's config)
         *
        */
        int load_pq_codebooks(system_conf_t &sys_conf, pq_conf_t &pq_conf);

        /*
         * load quantizer into index
         *
         *  @param  sys_conf   system_orca table's configuration
         *  @param  pq_conf    pq_conf table's record (a version of pq's config)
         *
        */
        int load_quantizer(system_conf_t &sys_conf, pq_conf_t &pq_conf);

        /*
         * Build Precomputed Index file
         *
         * @param path_base base vector file full path
         * @param path_prcomputed_index  prcomputed index file full path
         *
         * return value:
         *  0  success build precomputed index file
         * -1  error happend in progress of build precomputed index file
         *
         * TODO: notice
         * current code has limition on size of vector file
         * may be we cannot process vector file which has vector number more than 100W
         * caller must ensure not exceed this limit
         */
        int build_precomputed_index(const char *path_base, const char *path_prcomputed_index);

        /*
         * Build Precomputed Index files
         *
         * build precomputed index for batch, which has no precomputed index
         *
         * @param  sys_conf      system_orca table configure
         *
         *
         */
        int build_precomputed_index(system_conf_t &sys_conf);

        int build_one_precomputed_index(system_conf_t &sys_conf, size_t batch_idx);

        int build_precomputed_index_ex(const char* path_base, const char* path_prcomputed_index);

        int build_precomputed_index_ex(system_conf_t& sys_conf);

        int build_one_precomputed_index_ex(system_conf_t& sys_conf, size_t batch_idx);

        /*
         * Build Index file with given batchs of data
         *
         * @param path_base  base vector file full path
         * @param batch_begin  first batch number to process
         * @param batch_end  last batch number to process
         *
         * return value:
         *  0  success build index file
         * -1  error happend in progress of build index file
         *
         */
        int build_batchs_to_index(const system_conf_t &sys_conf, const size_t batch_begin, const size_t batch_end);

        /*
         * Build Index file with given batchs of data
         *
         * @param sys_conf  system conf
         * @param batch_list  list of batch number
         *
         * return value:
         *  0  success build index file
         * -1  error happend in progress of build index file
         *
         */
        int build_batchs_to_index(const system_conf_t &sys_conf, std::vector<size_t> &batch_list);

        /*
         * Build Index file with given batchs of data
         *
         * @param sys_conf  system conf
         * @param batch_list  list of valid batch info
         *
         * return value:
         *  0  success build index file
         * -1  error happend in progress of build index file
         *
         */
        int build_batchs_to_index(const system_conf_t &sys_conf, std::vector<batch_info_t> &batch_list);
        int add_one_batch_to_index(const system_conf_t &sys_conf, size_t batch_idx, bool final_add);

        int build_batchs_to_index_ex(const system_conf_t& sys_conf, const size_t batch_begin, const size_t batch_end);
        int build_batchs_to_index_ex(const system_conf_t& sys_conf, std::vector<size_t>& batch_list);
        int build_batchs_to_index_ex(const system_conf_t& sys_conf, std::vector<batch_info_t>& batch_list);
        int add_one_batch_to_index_ex(const system_conf_t& sys_conf, size_t batch_idx, bool final_add);
        int add_one_batch_to_index_ex(const char *path_vector, const char *path_precomputed_idx);
        int add_one_batch_to_index_ex(const system_conf_t &sys_conf, size_t batch_idx);

        void get_path_index(const system_conf_t &sys_conf, const size_t idx_ver, char *path_index);
        int save_index(const system_conf_t &sys_conf, const size_t idx_ver);

        void get_path_pq(const system_conf_t &sys_conf, const size_t idx_ver, char *path_index);
        void get_path_opq_matrix(const system_conf_t &sys_conf, const size_t idx_ver, char *path_index);
        void get_path_norm_pq(const system_conf_t &sys_conf, const size_t idx_ver, char *path_index);

        void get_path_vector(const system_conf_t &sys_conf, const size_t batch_idx, char *path_vector);
        void get_path_precomputed_idx(const system_conf_t &sys_conf, const size_t batch_idx, char *path_precomputed_idx);

        int build_index(const system_conf_t &sys_conf, const size_t batch_begin, const size_t batch_end, const size_t index_ver);
        int rebuild_index(system_conf_t &sys_conf, size_t &batch_start, size_t &batch_end);

        int build_index_ex(const system_conf_t& sys_conf, const size_t batch_begin, const size_t batch_end, const size_t index_ver);
        int rebuild_index_ex(system_conf_t& sys_conf, size_t& batch_start, size_t& batch_end);

        int load_index(const system_conf_t &sys_conf, const size_t idx_ver);

        int get_batchs_attr();
        int get_batchs_attr_ex();

        int getBatchByLabel(long label, size_t &vec_no);

        int deleteBatchByTime(time_t time_del);
        int deleteBatchFile(system_conf_t &sys_conf, size_t batch_idx);
        int deleteBatchFiles(system_conf_t &sys_conf, std::vector<batch_info_t> &batch_list);


      protected:
        /// Distances to the coarse centroids. Used for distance computation between a query and base points
        std::vector<float> query_centroid_dists;

        /// Distances between coarse centroids and their sub-centroids
        std::vector<std::vector<float>> inter_centroid_dists;

    private:
        void compute_residuals(size_t n, const float *x, float *residuals,
                               const float *subcentroids, const idx_t *keys);

        void reconstruct(size_t n, float *x, const float *decoded_residuals,
                         const float *subcentroids, const idx_t *keys);

        void compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                      const float *points, size_t group_size);

        float compute_alpha(const float *centroid_vectors, const float *points,
                            const float *centroid, const float *centroid_vector_norms_L2sqr, size_t group_size);

        size_t getBatchByLabel(long label);

        int get_vec_id(const char* vec_path, size_t vec_no, uint32_t& vec_id);
        int get_vec_id_ex(const char* vec_path, size_t vec_no, uint32_t& vec_id);

        /*
         * Add vectors in a batch file into index
         *
         * @param path_base  base vector file full path
         * @param path_precomputed_idx  precomputed index file file full path
         *
         * return value:
         *  0  success to add the batch vector
         * -1  failed to add the batch vector
         *
         */
        int add_one_batch_to_index(const char* path_base, const char* path_precomputed_idx);
        int add_one_batch_to_index(const system_conf_t& sys_conf, size_t batch_idx);
    };
}
#endif //IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
