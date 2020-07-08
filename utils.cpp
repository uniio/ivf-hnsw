
#include "utils.h"

namespace ivfhnsw {

    void random_subset(const float *x, float *x_out, size_t d, size_t nx, size_t sub_nx) {
        long seed = 1234;
        std::vector<int> perm(nx);
        faiss::rand_perm(perm.data(), nx, seed);

        for (size_t i = 0; i < sub_nx; i++)
            memcpy(x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
    }


    float fvec_L2sqr(const float *x, const float *y, size_t d) {
        float PORTABLE_ALIGN32 TmpRes[8];
        #ifdef USE_AVX
        size_t qty16 = d >> 4;

        const float *pEnd1 = x + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (x < pEnd1) {
            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return (res);
        #else
        size_t qty16 = d >> 4;

        const float *pEnd1 = x + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (x < pEnd1) {
            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return (res);
        #endif
    }

    // help function for getL2Distance, to get original vector from base vector file by
    // vector id, and store result in data
    int readBaseVec(const char *path_base, const size_t dim, const size_t vec_id, float *data) {
        auto vec_off = vec_id * (sizeof(uint32_t) + dim);
        std::ifstream fs_input(path_base, std::ios::binary);
        fs_input.seekg(vec_off, fs_input.beg);
        readXvecFvec<uint8_t>(fs_input, data, dim, 1);
        fs_input.close();

        return 0;
    }

    // calculate distance between query vector and vector in base vector file indicated by vector id
    float getL2Distance(const float *query, const char *path_base, const size_t dim, const long vec_id) {
        std::vector<float> baseVec(dim);

        readBaseVec(path_base, dim, vec_id, baseVec.data());
        return fvec_L2sqr(query, baseVec.data(), dim);
    }
}
