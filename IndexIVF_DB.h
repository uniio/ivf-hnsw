/*
 * IndexIVF_DB.h
 *
 */
#ifndef INDEXIVF_DB_H_
#define INDEXIVF_DB_H_

#include <vector>
#include <string.h>
#include <arpa/inet.h>
#include <postgresql/libpq-fe.h>
#include <ivf-hnsw/utils.h>

namespace ivfhnsw {

typedef struct {
    size_t batch;
    size_t start_id;
    bool valid;
    int ts;
    bool no_precomputed_idx;
    size_t batch_size;
} batch_info_t;

class Index_DB {
  private:
    typedef uint32_t idx_t;
    char conninfo[512];
    PGconn *conn = nullptr;

  public:
    explicit Index_DB(const char *host, uint32_t port, const char *db_nm, const char *db_usr, const char *pwd_usr);
    virtual ~Index_DB();

    int Connect();

    /*
     *  AllocateBatch used to get a new batch number used for new data
     *
     *  @param  batch       batch number
     *  @param  start_id    first vector id in the batch
     *
     */
    int AllocateBatch(size_t batch, size_t start_id);

    /*
     *  ActiveBatch used to mark the batch, it means the batch vector file not write anymore
     *  it can be used to build precomputed index
     *
     *  @param  batch   batch number
     *
     */
    int ActiveBatch(size_t batch);

    /*
     *  ActivePrecomputedIndex used to mark the batch, it means precomputed index for the batch is created
     *
     *  @param  batch   batch number
     *
     */
    int ActivePrecomputedIndex(size_t batch);
    int AppendPQInfo(size_t ver, bool with_opq, size_t nsubc);
    int GetLatestPQInfo(size_t &ver, bool &with_opq, size_t &code_size, size_t &nsubc);
    int AppendPQConf(pq_conf_t &pq_conf);
    int GetLatestPQConf(pq_conf_t &pq_conf);
    int GetBatchList(std::vector<batch_info_t> &batch_list);

    int AppendIndexInfo(size_t idx_ver, size_t batch_start, size_t batch_end);
    int GetLatestIndexInfo(size_t &ver, size_t &batch_start, size_t &batch_end);

    int SetSysConfig(system_conf_t &sys_conf);
    int GetSysConfig(system_conf_t &sys_conf);

    int GetLatestBatch(int &batch);

  private:
    int DropTable(char *tbl_nm);
    int CmdWithTrans(char *sql_str);
};
}

#endif /* INDEXIVF_DB_H_ */
