/*
 * IndexIVF_DB.cpp
 *
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <arpa/inet.h>
#include <postgresql/libpq-fe.h>
#include "IndexIVF_DB.h"

#if 0
using std::cout;
using std::endl;

Index_DB::Index_DB(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr) {
    sprintf(conninfo, "hostaddr=%s port=%u port=%s user=%s user=%s");
}

Index_DB::~Index_DB() {
	if (conn) PQfinish(conn);
}

int Index_DB::Connect() {
	conn = PQconnectdb(conninfo);
	if (PQstatus(conn) != CONNECTION_OK) {
		cout << "connect failed. PQstatus : " << PQstatus(conn)  << endl;
		cout << PQerrorMessage(conn) << endl;

		return -1;
	}

	return 0;
}

int Index_DB::CreateTable(char *cmd_str) {
	PGresult   *res;
	int rc = 0;

    res = PQexecParams(conn,
                       cmd_str,
                       0,
                       NULL,
                       NULL,
                       NULL,
                       NULL,
                       1);

    if (PQresultStatus(res) != PGRES_TUPLES_OK)
    {
    	cout << "Failed to create table: " << PQerrorMessage(conn) << endl;
    	rc = -1;
    }
    PQclear(res);

    return rc;
}

int Index_DB::CreateIndexTables(size_t batch_idx) {
	PGresult *res;
	char sql_str[1024], tbl_nm[128];
	int rc = -1;

    // create vector index table
	sprintf(tbl_nm, "index_vector_%lu", batch_idx);
	sprintf(sql_str,
			"CREATE TABLE %s (dim INTEGER, id bytea)",
			tbl_nm);
	rc = CreateTable(sql_str);
	if (rc) return rc;

    // create PQ codec table
	sprintf(tbl_nm, "pq_codec_%lu", batch_idx);
	sprintf(sql_str,
			"CREATE TABLE %s (dim INTEGER, codes bytea)",
			tbl_nm);
	rc = CreateTable(sql_str);
	if (rc) return rc;

    // create norm PQ codec table
	sprintf(tbl_nm, "norm_codec_%lu", batch_idx);
	sprintf(sql_str,
			"CREATE TABLE %s (dim INTEGER, norm_codes bytea)",
			tbl_nm);
	rc = CreateTable(sql_str);
	if (rc) return rc;

    // create NN centriods index table
	sprintf(tbl_nm, "nn_centroid_idxs_%lu", batch_idx);
	sprintf(sql_str,
			"CREATE TABLE %s (dim INTEGER, nn_centroid_idxs bytea)",
			tbl_nm);
	rc = CreateTable(sql_str);
	if (rc) return rc;

    // create group size table
	sprintf(tbl_nm, "subgroup_sizes_%lu", batch_idx);
	sprintf(sql_str,
			"CREATE TABLE %s (dim INTEGER, subgroup_sizes bytea)",
			tbl_nm);
	rc = CreateTable(sql_str);
	if (rc) return rc;

    // create inter centroid distances table
	sprintf(tbl_nm, "inter_centroid_dists_%lu", batch_idx);
	sprintf(sql_str,
			"CREATE TABLE %s (dim INTEGER, inter_centroid_dists bytea)",
			tbl_nm);
	rc = CreateTable(sql_str);
	if (rc) return rc;

    // create msic table for alphas and centroid_norms
	sprintf(tbl_nm, "misc_%lu", batch_idx);
	sprintf(sql_str,
			"CREATE TABLE %s (size INTEGER, misc_data bytea)",
			tbl_nm);
	rc = CreateTable(sql_str);
	if (rc) return rc;
}

int Index_DB::DropTable(char *tbl_nm) {
	char sql_str[1024];
	int rc = 0;

	// drop index meta table
	sprintf(sql_str, "DROP TABLE IF EXISTS %s", tbl_nm);
    PGresult *res = PQexec(conn, sql_str);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
    	cout << "Failed to drop table: " << tbl_nm << endl;
    	rc = -1;
    }
    PQclear(res);

    return rc;
}

int Index_DB::DropServiceTables() {
	int rc;

	rc = DropTable("system");
	if (rc) return rc;

	rc = DropTable("index_meta");
	if (rc) return rc;
}

void Index_DB::DropIndexTables(size_t batch_idx) {
	char tbl_nm[128];
	int rc = -1;

    // drop PQ codec table
    sprintf(tbl_nm, "pq_codec_%lu", batch_idx);
	DropTable(tbl_nm);

    // drop norm PQ codec table
    sprintf(tbl_nm, "norm_codec_%lu", batch_idx);
	DropTable(tbl_nm);

    // drop NN centriods index table
    sprintf(tbl_nm, "nn_centroid_idxs_%lu", batch_idx);
	DropTable(tbl_nm);

    // drop NN group size table
    sprintf(tbl_nm, "subgroup_sizes_%lu", batch_idx);
	DropTable(tbl_nm);

    // drop inter centroid distances table
    sprintf(tbl_nm, "inter_centroid_dists_%lu", batch_idx);
	DropTable(tbl_nm);

    // drop msic table for alphas and centroid_norms
    sprintf(tbl_nm, "misc_%lu", batch_idx);
	DropTable(tbl_nm);
}


/*
 * system (batch_idx INTEGER)
 * index_meta (dim INTEGER, nc INTEGER, nsubc INTEGER)
*/
int Index_DB::CreateServiceTable() {
	int rc = 0;
	PGresult *res;

	rc = CreateTable("CREATE TABLE IF NOT EXISTS system (batch_idx INTEGER NOT NULL)");
	if (rc) goto out;

	rc = CreateTable("CREATE TABLE IF NOT EXISTS index_meta (dim INTEGER, nc INTEGER, nsubc INTEGER)");
out:
	return rc;
}

template<typename T>
int Index_DB::SaveVector(char *table_nm, std::vector<T> &id) {
	int rc = 0;
	size_t sz = id.size();
	size_t dsize = sz * sizeof(T);

    PGresult* res;
    const uint32_t sz_big_endian = htonl((uint32_t)sz);
    const char* const paramValues[] = {&sz_big_endian, id.data()};
    const int paramLenghts[] = {sizeof(sz_big_endian), dsize};
    const int paramFormats[] = {1, 1}; /* binary */
    char sql_str[512];
    sprintf(sql_str,
    		"INSERT INTO %s (dim, id) VALUES ($1::integer, $2::bytea)",
			table_nm);

    res = PQexecParams(
      conn,
	  sql_str,
      2,
      NULL, /* Types of parameters, unused as casts will define types */
      paramValues,
      paramLenghts,
      paramFormats,
      1 // binary results
    );

    if (PQresultStatus(res) != PGRES_TUPLES_OK)
    {
    	cout << "Failed to insert data to table: " << table_nm << endl;
    	rc = -1;
    }
    PQclear(res);

    return rc;
}

int Index_DB::SaveIndexMeta(size_t dim, size_t nc, size_t nsub) {
	int rc = -1;
	PGresult *res;
	char sql_str[512];

    sprintf(sql_str, "INSERT index_meta SET dim=%u nc=%lu nsubc=%lu",
    		dim, nc, nsub);
    res = PQexec(conn, sql_str);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
    	cout << "INSERT command failed" << endl;
        goto out;
    }
    rc = 0;

out:
    PQclear(res);
    return rc;
}

/*
 * CREATE TABLE system (batch_idx INTEGER PRIMARY KEY);
*/
int Index_DB::UpdateIndex(size_t batch_idx) {
	int rc = -1;
	PGresult *res;
	char sql_str[512];

    res = PQexec(conn, "BEGIN");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
    	cout << "BEGIN command failed" << endl;
        PQclear(res);
        return rc;
    }
    PQclear(res);

    sprintf(sql_str, "INSERT system SET batch_idx=%u", batch_idx);
    res = PQexec(conn, sql_str);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
    	cout << "INSERT command failed" << endl;
        PQclear(res);
        return rc;
    }
    PQclear(res);

    sprintf(sql_str, "DELETE FROM system WHERE batch_idx<%u", batch_idx);
    res = PQexec(conn, sql_str);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
    	cout << "INSERT command failed" << endl;
        PQclear(res);
        return rc;
    }
    PQclear(res);

    res = PQexec(conn, "COMMIT");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
    	cout << "COMMIT command failed" << endl;
        PQclear(res);
        return rc;
    }
    PQclear(res);

    rc = 0;
    return rc;
}
#endif



