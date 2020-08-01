/*
 * IndexIVF_DB.h
 *
 */

#ifndef INDEXIVF_DB_H_
#define INDEXIVF_DB_H_

#include <vector>
#include <postgresql/libpq-fe.h>
#if 0
class Index_DB {
private:
    typedef uint32_t idx_t;
	char conninfo[512];
    PGconn *conn = nullptr;
public:
    explicit Index_DB(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr);
	virtual ~Index_DB();

	int Connect();
	int CreateIndexTables(size_t batch_idx);
	void DropIndexTables(size_t batch_idx);
	int CreateServiceTable();
	int DropServiceTables();
	int SaveIndexMeta(size_t dim, size_t nc, size_t nsub);
	template<typename T>
	int SaveVector(char *table_nm, std::vector<T> &id);
private:
	int CreateTable(char *cmd_str);
	int UpdateIndex(size_t batch_idx);
	int UpdateMeta(size_t batch_idx);
	int DropTable(char *tbl_nm);
};
#endif

#endif /* INDEXIVF_DB_H_ */
