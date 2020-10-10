/*
 * build_index.cpp
 *
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_DB.h>
#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>
#include <ivf-hnsw/hnswalg.h>
#include <ivf-hnsw/utils.h>
#include <time.h>
using std::cout;
using std::endl;

using namespace hnswlib;
using namespace ivfhnsw;

#define INDEX_COUNT 1   //索引个数
#define BATCH_COUNT 1  //BATCH 文件个数
#define INDEX_BATCH 1   //每个索引负责BATCH　文件个数 




int main(int argc, char** argv) {
    system_conf_t           sys_conf;
    pq_conf_t               pq_conf;
    Index_DB*               db_p  = nullptr;
    int                     rc;
    size_t                  ver, batch_start, batch_end;
    std::vector<size_t>     batchs_to_index_test[INDEX_COUNT];
    time_t t;
    char ch[64] = {0};
    pid_t pid = getpid();
    std::string cmd;

    int j=0;
    for(int i =0 ;i < INDEX_COUNT;i++) 
    {
        std::cout << "------" << i << "------"  << std::endl;
        int n = 0;
        std::vector<size_t>     batchs_to_index;
        for(;j <= BATCH_COUNT;j++)
        {
                                                                            
            if(n < INDEX_BATCH)
            {   
                batchs_to_index_test[i].push_back(j);
                n++;    
            }
            else
            {
                n = 0;
                for(size_t bs=0;bs<batchs_to_index_test[i].size();bs++)
                {
                    std::cout << "-----------------" << batchs_to_index_test[i].at(bs) << "------------------"  << std::endl;
                
                }
                std::cout << "-----------------------------------------------"  << std::endl;
                break;
            }
            
           
        }
    }

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
    // Load PQ
    //==========
    rc = db_p->GetLatestIndexInfo(ver, batch_start, batch_end);
    if (rc) {
        std::cout << "Failed to get vector batch info" << std::endl;
        goto out;
    }
    if (ver == 0) {
        std::cout << "No batch info" << std::endl;
        // valid version always start from 1
    }

    //==================
    // Initialize Index
    //==================

    IndexIVF_HNSW_Grouping *indexGroup[INDEX_COUNT];
    cmd += "cat /proc/";
    cmd += std::to_string(pid);
    cmd += "/status | grep VmRSS";
    for(size_t i=0;i<INDEX_COUNT;i++)
    {
        std::cout << "----index " << i << " begin memory " << std::endl;
        system(cmd.c_str());

        indexGroup[i] = new IndexIVF_HNSW_Grouping(sys_conf.dim, sys_conf.nc, sys_conf.code_size, 8, sys_conf.nsubc, db_p);

        t = time(NULL);
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t));
        std::cout << "----index " << i << " begin time " <<  ch <<  std::endl;

        rc = indexGroup[i]->load_quantizer(sys_conf, pq_conf);
        if (rc) {
            std::cout << "Failed to load quantizer" << std::endl;
            goto out;
        }

        t = time(NULL);
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t));
        std::cout << "----load quantizer: " << i << " end  time " <<  ch <<  std::endl;
        indexGroup[i]->load_pq_codebooks(sys_conf, pq_conf);
        if (rc) {
            std::cout << "Failed to load PQ codebooks" << std::endl;
            goto out;
        }
        t = time(NULL);
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t));
        std::cout << "----load pq_codebooks: " << i << " end  time " <<  ch <<  std::endl;

        std::cout << "----index " << i << " load codebook memory " << std::endl;
        system(cmd.c_str());

        t = time(NULL);
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t));
        std::cout << "----build batch: " << i << " begin time " <<  ch <<  std::endl;

        rc = indexGroup[i]->build_batchs_to_index_ex(sys_conf, batchs_to_index_test[i]);
        if (rc) {
            std::cout << "Failed to rebuild index " << i << std::endl;
            goto out;
        }

        t = time(NULL);
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t));
        std::cout << "----build batch: " << i << " end time " <<  ch <<  std::endl;


        t = time(NULL);
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t));
        std::cout << "----index " << i << " end time " <<  ch <<  std::endl;

        std::cout << "----index " << i << " end memory " << std::endl;
        system(cmd.c_str());

	rc = indexGroup[i]->save_index(sys_conf, ver);
        if (rc) {
            std::cout << "Failed to save index" << std::endl;
            goto out;
        }
        std::cout << "Success to save index" << std::endl;

/*        rc = db_p->AppendIndexInfo(ver, batch_start, batch_end);
        if (rc) {
            std::cout << "Failed to add index info to database" << std::endl;
            goto out;
        }
        std::cout << "Success add rebuild index info to database" << std::endl;
*/
        }
    
    //	std::cout << "Please check memory cost of two index" << std::endl;
    //	while (1) {
    //   	    sleep(1);
    //	}

out:
    if (db_p != nullptr) delete db_p;
    for(size_t i=0;i<INDEX_COUNT;i++)
    {
       if(indexGroup[i]!= nullptr)
       {
            delete indexGroup[i];
            indexGroup[i] = nullptr;
       }
    }

    return rc;
}
