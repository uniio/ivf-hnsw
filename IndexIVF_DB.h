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

typedef struct batch_info {
    size_t batch;
    bool valid;
} batch_info_t;

class Index_DB {
  private:
    typedef uint32_t idx_t;
    char conninfo[512];
    PGconn *conn = nullptr;

  public:
    explicit Index_DB(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr);
    virtual ~Index_DB();

    int Connect();
    int CreateBatch(size_t batch);
    int AppendPQInfo(size_t ver, bool with_opq, size_t code_size, size_t nsubc);
    int GetLatestPQInfo(size_t &ver, bool &with_opq, size_t &code_size, size_t &nsubc);
    int AppendPQConf(pq_conf_t &pq_conf);
    int GetLatestPQConf(pq_conf_t &pq_conf);
    int GetBatchList(std::vector<batch_info_t> &batch_list);

    int AppendIndexInfo(size_t idx_ver, size_t batch_start, size_t batch_end);
    int GetLatestIndexInfo(size_t &ver, size_t &batch_start, size_t &batch_end);

    int SetSysConfig(system_conf_t &sys_conf);
    int GetSysConfig(system_conf_t &sys_conf);

  private:
    int GetLatestBatch(size_t &batch);
    int DropTable(char *tbl_nm);
    int CmdWithTrans(char *sql_str);
};
}

#endif /* INDEXIVF_DB_H_ */
