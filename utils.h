#ifndef IVF_HNSW_LIB_UTILS_H
#define IVF_HNSW_LIB_UTILS_H

#include <queue>
#include <limits>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <assert.h>

#include <faiss/utils.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

#define EPS 0.00001

namespace ivfhnsw {
    /// Clock class
    class StopW {
        std::chrono::steady_clock::time_point time_begin;
    public:
        StopW() {
            time_begin = std::chrono::steady_clock::now();
        }

        float getElapsedTimeMicro() {
            std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
            return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
        }

        void reset() {
            time_begin = std::chrono::steady_clock::now();
        }
    };

    /// Read variable of the arbitrary type
    template<typename T>
    void read_variable(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    /// Read std::vector of the arbitrary type
    template<typename T>
    void read_vector(std::istream &in, std::vector<T> &vec)
    {
        uint32_t size;
        in.read((char *) &size, sizeof(uint32_t));
        vec.resize(size);
        in.read((char *) vec.data(), size * sizeof(T));
    }

    /// Write variable of the arbitrary type
    template<typename T>
    void write_variable(std::ostream &out, const T &val) {
        out.write((char *) &val, sizeof(T));
    }

    /// Write std::vector in the fvec/ivec/bvec format
    template<typename T>
    void write_vector(std::ostream &out, std::vector<T> &vec)
    {
        const uint32_t size = vec.size();
        out.write((char *) &size, sizeof(uint32_t));
        out.write((char *) vec.data(), size * sizeof(T));
    }


    /// Read fvec/ivec/bvec format vectors
    template<typename T>
    void readXvec(std::ifstream &in, T *data, const size_t d, const size_t n = 1)
    {
        uint32_t dim = d;
        for (size_t i = 0; i < n; i++) {
            in.read((char *) &dim, sizeof(uint32_t));
            if (dim != d) {
                std::cout << "file error\n";
                exit(1);
            }
            in.read((char *) (data + i * dim), dim * sizeof(T));
        }
    }

    /// Write fvec/ivec/bvec format vectors
    template<typename T>
    void writeXvec(std::ofstream &out, T *data, const size_t d, const size_t n = 1)
    {
        const uint32_t dim = d;
        for (size_t i = 0; i < n; i++) {
            out.write((char *) &dim, sizeof(uint32_t));
            out.write((char *) (data + i * dim), dim * sizeof(T));
        }
    }

    /// Read fvec/ivec/bvec format vectors and convert them to the float array
    template<typename T>
    void readXvecFvec(std::ifstream &in, float *data, const size_t d, const size_t n = 1)
    {
        uint32_t dim = d;
        T mass[d];

        for (size_t i = 0; i < n; i++) {
            in.read((char *) &dim, sizeof(uint32_t));
            if (dim != d) {
                std::cout << "file error\n";
                exit(1);
            }
            in.read((char *) mass, dim * sizeof(T));
            for (size_t j = 0; j < d; j++)
                data[i * dim + j] = 1. * mass[j];
        }
    }

    /// Check if file exists
    inline bool exists(const char *path) {
        std::ifstream f(path);
        return f.good();
    }

    enum vec_t {
        base_vec = 0,
        centroid_vec = 1
    };

    typedef struct SearchInfo {
        float distance;
        long label;
    } SearchInfo_t;

    /// Get a random subset of <sub_nx> elements from a set of <nx> elements
    void random_subset(const float *x, float *x_out, size_t d, size_t nx, size_t sub_nx);

    /// Main fast distance computation function
    float fvec_L2sqr(const float *x, const float *y, size_t d);

    // calculate distance between query vector and vector in base vector file indicated by vector id
    float getL2Distance(const float *query, const char *path_base, const size_t dim,
                        const long vec_id, vec_t type_v);

    bool cmp(SearchInfo_t a, SearchInfo_t b);

    size_t base_vec_num(const char *path_base, size_t vec_dim);

    void get_files(const char *path_dir, const char *file_ext, std::vector<std::string> &file_list);
    void check_files(const char *file_prefix, std::vector<std::string> &file_list);
    void get_index_name(const char *path_idx, size_t idx, char *idx_name);
    int mkdir_p(const char *dir, const mode_t mode);
    int get_vec_attr(const char *path_vec, uint32_t &dim, size_t &nvecs);

    /// clone file_src in new file named file_dst
    int copy_file(const char *file_src, const char *file_dst);
}
#endif //IVF_HNSW_LIB_UTILS_H
