/*
 * orcv.hpp
 *
 */

#ifndef ORCV_H_
#define ORCV_H_

#include <stdint.h>

typedef struct orcvhdr {
    uint32_t n;               // number of vectors (changes)
    uint32_t nc;              // number of centroids (fixed)
    uint32_t code_size;        // code size in PQ format
    uint32_t code_bytes;       // code size in bytes
    uint32_t d;               // vector dimensions
    uint32_t M;               // seach index internal
    uint32_t efConstruction;    // search index internal
    float dmatch;          // distance for vector match
    float dnear;           // distance for vector near match
    uint8_t do_opq;           // if vector rotation matrix is used
} orcvhdr_t;



#endif /* ORCV_H_ */
