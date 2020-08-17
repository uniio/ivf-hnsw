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

        Index_DB *db_p = nullptr;
        bool db_free = false;

    public:
        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids);

        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids, Index_DB *db_ref);

        ~IndexIVF_HNSW_Grouping();

        /** Add <group_size> vectors of dimension <d> from the <group_idx>-th group to the index.
          *
          * @param group_idx         index of the group
          * @param group_size        number of base vectors in the group
          * @param x                 base vectors to add (size: group_size * d)
          * @param ids               ids to store for the vectors (size: groups_size)
        */
        void add_group(size_t group_idx, size_t group_size, const float *x, const idx_t *ids);

        void search(size_t k, const float *x, float *distances, long *labels);

        // apply disk search based on ANN search result
        void searchDisk(size_t k, const float *query, float *distances, long *labels, const char *path_base);

        int write(const char *path_index);
        int read(const char *path_index);

        // similar as write function, except can truncate file before write
        int write(const char *path_index, bool do_trunc);

        // setup database related things
        int setup_db(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr);

        int prepare_db();

        // commit batch index status
        int create_new_batch(size_t batch_idx);

        void train_pq(size_t n, const float *x);

        /// Compute distances between the group centroid and its <subc> nearest neighbors in the HNSW graph
        void compute_inter_centroid_dists();

        /// Write distance between centroids to file given by path
        void dump_inter_centroid_dists(char *path);

        /*
         * Build PQ files
         *
         * @param path_learn  learn vector file full path
         * @param path_out    directory to store PQ files generated
         * @param pq_ver    version of PQ
         * @param with_opq    enable opq or not
         * @param code_size   Code size per vector in bytes
         * @param rsubt       ratio of vectors in learn vector file to train
         * @param nsubc       number of subcentroids per group
         *
         */
        int build_pq_files(const char *path_learn, const char *path_out,
        		size_t pq_ver,
                bool with_opq, size_t code_size, double rsubt, size_t nsubc);

        int append_pq_info(size_t ver, bool with_opq, size_t code_size, size_t nsubc);
        int get_latest_pq_info(size_t &ver, bool &with_opq, size_t &code_size, size_t &nsubc);

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
        int build_prcomputed_index(const char *path_base, const char *path_prcomputed_index);

        /*
         * Build Precomputed Index files
         *
         * build precomputed index for batch, which has no precomputed index
         *
         * @param  sys_conf      system_orca table configure
         * @param  skip_batch    don't process given batch
         *
         * skip_batch used to skip given batch, that is batch which service1b current used
         *
         */
        int build_prcomputed_index(system_conf_t &sys_conf, size_t skip_batch);


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
        int build_batchs_to_index(const char *path_base, const size_t batch_begin, const size_t batch_end);

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
        int add_one_batch_vector(const char *path_base, const char *path_precomputed_idx);
        void get_path_index(const system_conf_t sys_conf, const size_t idx_ver, char *path_index);
        int save_index(const system_conf_t sys_conf, const size_t idx_ver);

        void get_path_centroids(const system_conf_t sys_conf, char *path_index);

        void get_path_info(const system_conf_t sys_conf, const pq_conf_t pq_conf, char *path_index);
        void get_path_edges(const system_conf_t sys_conf, const pq_conf_t pq_conf, char *path_index);

        void get_path_pq(const system_conf_t sys_conf, const size_t idx_ver, char *path_index);
        void get_path_opq_matrix(const system_conf_t sys_conf, const size_t idx_ver, char *path_index);
        void get_path_norm_pq(const system_conf_t sys_conf, const size_t idx_ver, char *path_index);

        void get_path_vector(const system_conf_t sys_conf, const size_t batch_idx, char *path_vector);
        void get_path_precomputed_idx(const system_conf_t sys_conf, const size_t batch_idx, char *path_precomputed_idx);

        int build_index(const system_conf_t sys_conf, const size_t batch_begin, const size_t batch_end, const size_t index_ver);

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
    };
}
#endif //IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
