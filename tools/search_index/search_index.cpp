/*
 * load_index.cpp
 *
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <ivf-hnsw/IndexIVF_DB.h>
#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>
#include <ivf-hnsw/hnswalg.h>
#include <ivf-hnsw/utils.h>

using std::cout;
using std::endl;

using namespace hnswlib;
using namespace ivfhnsw;

const size_t max_file_size =  4096 * 1024 * 34; // 136Mb

char  *mmap_batchfile();
void  unmmap_batchfile(char *p, size_t size);
int   search_index(IndexIVF_HNSW_Grouping* index, int vec_id);
int   search_rand_index(IndexIVF_HNSW_Grouping* index, int max_search);
char  *parse_batchfile(int vector_num, uint8_t* query);
void  print_binary_vector(int len, uint8_t *vector);

int main(int argc, char** argv) {
    system_conf_t           sys_conf;
    pq_conf_t               pq_conf;
    Index_DB*               db_p  = nullptr;
    IndexIVF_HNSW_Grouping* index = nullptr;
    int                     rc;
    size_t                  ver, batch_start, batch_end;

    int vec_id;
    int max_search;
    if(argc != 2) {
        std::cout << "usage: ./search_index [max_search_times]" << std::endl;
        goto out;
    }
    vec_id = atoi(argv[1]);
    max_search = vec_id;
    std::cout << "[Result] Exec: ./search_index " << vec_id  << std::endl;
    

    // Initialize database interface
    // Following code is 1st start point in service, get work path first
    db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
    rc = db_p->Connect();
    if (rc) {
        std::cout << "Failed to connect to Database server" << std::endl;
        goto out;
    }

    rc = db_p->GetSysConfig(sys_conf);
    if (rc) {
        std::cout << "Failed to get system configuration" << std::endl;
        goto out;
    }

    rc = db_p->GetLatestPQConf(pq_conf);
    if (rc) {
        std::cout << "Failed to get PQ configuration" << std::endl;
        goto out;
    }

    //==================
    // Initialize Index
    //==================
    index = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);
    index->do_opq  = pq_conf.with_opq;
    // set nprobe
    index->nprobe = sys_conf.nprobe;
    // set other sys_conf
    //index->set_sysconf(sys_conf);
    index->load_sys_conf();
    // get batch list size
    rc = index->get_batchs_attr();
    //rc = index->get_batchlist();
    if (rc) {
        std::cout << "Failed to Get Batch List File Size from DB" << std::endl;
        goto out;
    }

    //==========
    // Load PQ
    //==========
    rc = index->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ CodeBooks" << std::endl;
        goto out;
    }
    std::cout << "Success to load PQ CodeBooks" << std::endl;

    rc = db_p->GetLatestIndexInfo(ver, batch_start, batch_end);
    if (rc) {
        std::cout << "Failed to get vector batch info" << std::endl;
        goto out;
    }
    if (ver == 0) {
        std::cout << "No batch info" << std::endl;
        // valid version always start from 1
        rc  = 1;
        goto out;
    }

    rc = index->load_quantizer(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load quantizer" << std::endl;
        goto out;
    }
    /*
    index->load_pq_codebooks(sys_conf, pq_conf);
    if (rc) {
        std::cout << "Failed to load PQ codebooks" << std::endl;
        goto out;
    }
    */

    rc = index->load_index(sys_conf, ver);
    if (rc) {
        std::cout << "Failed to load index" << std::endl;
        goto out;
    }
    std::cout << "Success load index" << std::endl;
#if 0
    rc = search_index(index, vec_id);
#else
    rc = search_rand_index(index, max_search);
#endif
    if (rc) {
        std::cout << "Failed to search index" << std::endl;
        goto out;
    }

out:
    if (db_p != nullptr) delete db_p;
    if (index != nullptr) delete index;

    return rc;
}
int search_rand_index(IndexIVF_HNSW_Grouping* index, int max_search)
{
    int i;
    int rc = -1;
    int rand_vec;
    int success_count = max_search;
    int searched = 0;
    srand((int)time(NULL));
    for(i = 0; i < max_search; i++) {
        rand_vec = rand()%1000001;
        std::cout << "[Result] search_time="<< i << " vec_id=" << rand_vec << std::endl;
        rc = search_index(index, rand_vec);
        if (rc < 0) {
            std::cout << "Failed to search index, search_time="<< i << " vec_id=" << rand_vec << std::endl;
            success_count--;
            continue;
        } else if(rc == 1) {
            searched++;
            std::cout << "[Result] find vector " << rand_vec << std::endl;
        } else {
            std::cout << "[Result] not find vector " << rand_vec << std::endl;
        }
    }
    std::cout << "[Result] max_search_="<< max_search << " success=" << success_count << " searched=" << searched << std::endl;
    return 0;
}

int search_index(IndexIVF_HNSW_Grouping* index, int vec_id)
{
    int rc = -1;
    bool searched = false;
    int i;
    int sz;
    size_t k = 10;
    char *p_file;
    uint8_t query[128] = "";
    std::vector<size_t> id_vectors;

    //TODO
    p_file = parse_batchfile(vec_id, query);
    if(p_file == MAP_FAILED || query == nullptr) {
        std::cout << "Failed to parse batch to get query vector" << std::endl;
        goto out;
    }
    std::cout << "Successed to parse batch to get query vector" << std::endl;

    print_binary_vector(128, query);
    //int search(size_t k, const uint8_t* query, std::vector<size_t>& id_vectors)
    rc = index->search(k, query, id_vectors);
    if(rc) {
        std::cout << "Failed to search one index" << std::endl;
        goto out;
    }
    std::cout << "Successed to search index, Result is {";
    sz = id_vectors.size();
    for(i = 0; i < sz; i++) {
        std::cout << id_vectors[i] << " ,";
        if(id_vectors[i] == (size_t)vec_id) {
            searched = true;
        }
    }
    std::cout << "}" << std::endl;
    if(searched)
        rc = 1; // search success and get the vector we want to search
    else
        rc = 0; // search success but not get the vector we want to search
out:
    unmmap_batchfile(p_file, max_file_size);
    return rc;
}

void print_binary_vector(int len, uint8_t *vector)
{
    printf("[print binary vec] : ");
    for(int i=0; i<len; i++) {
        printf("%02x ",(unsigned char)vector[i]);
    }   
    printf("\n");
}

char* parse_batchfile(int vector_num, uint8_t* query)
{
    char *p;
    char *p_head = NULL;
    uint32_t vid;
    uint32_t dim;
    p_head = mmap_batchfile();
    if(p_head == MAP_FAILED) {
        goto out;
    }
    p = p_head;
    p += (vector_num * 136);
    memcpy(&vid, p, sizeof(uint32_t));
    p += sizeof(uint32_t);
    memcpy(&dim, p, sizeof(uint32_t));
    p += sizeof(uint32_t);
    memcpy(query, p, sizeof(uint8_t) * 128);
    printf("vid=%d, dim=%d\n", vid, dim);
out:
    return p_head;
}

char *mmap_batchfile()
{
	char *p_file = (char *)MAP_FAILED;
    char *path = "./bigann_base_000.bvecs";
    int fd = open(path, O_RDONLY, 0666);
    if(fd < 0) {
        printf("open %s failed\n", path);
        goto out;
    }
	p_file = (char*)mmap(NULL, max_file_size, PROT_READ, MAP_SHARED, fd, 0);	
    if(p_file == (char *)MAP_FAILED) {
        printf("mmap %s failed\n", path);
        goto out;
    }
out:
    if(fd >= 0) {
        close(fd);
    }
    return p_file;
     
}

void unmmap_batchfile(char *p, size_t size)
{
    if(p != MAP_FAILED && p != NULL) {
        munmap(p, size);
    }
}
