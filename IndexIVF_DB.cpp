/*
 * IndexIVF_DB.cpp
 *
 */

#include "IndexIVF_DB.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <postgresql/libpq-fe.h>

using std::cout;
using std::endl;

namespace ivfhnsw {

Index_DB::Index_DB(const char *host, uint32_t port, const char *db_nm, const char *db_usr, const char *pwd_usr) {
    sprintf(conninfo, "host=%s port=%u dbname=%s user=%s password=%s",
            host, port, db_nm, db_usr, pwd_usr);
}

Index_DB::~Index_DB() {
    if (conn)
        PQfinish(conn);
}

int Index_DB::Connect() {
    int rc = 0;
    conn = PQconnectdb(conninfo);
    if (PQstatus(conn) != CONNECTION_OK) {
        cout << "connect failed. PQstatus : " << PQstatus(conn) << endl;
        cout << PQerrorMessage(conn) << endl;
        PQfinish(conn);
        conn = nullptr;
        rc = -1;
    }

    return rc;
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

int Index_DB::GetBatchList(std::vector<batch_info_t> &batch_list) {
    PGresult *res;

    res = PQexec(conn, "SELECT * FROM batch_info ORDER BY batch ASC");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cout << "Failed to get data from batch_info table: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return -1;
    }

    int nRows = PQntuples(res);
    for (int i = 0; i < nRows; i++) {
        batch_info_t batch_cur;
        char* sb = PQgetvalue(res, i, PQfnumber(res, "ts"));
        if (!sb) {
            std::cout << "Invalid record in batch_info for batch: " << i << std::endl;
            return -1;
        }
        struct tm tm_time;
        strptime(sb, "%Y-%m-%d %H:%M:%S", &tm_time);
        batch_cur.ts = mktime(&tm_time);

        sb = PQgetvalue(res, i, PQfnumber(res, "no_precomputed_idx"));
        if (sb[0] == 'f' && sb[1] == '\0')
            batch_cur.no_precomputed_idx = false;
        else
            batch_cur.no_precomputed_idx = true;

        sb = PQgetvalue(res, i, PQfnumber(res, "valid"));
        if (sb[0] == 'f' && sb[1] == '\0')
            batch_cur.valid = false;
        else
            batch_cur.valid = true;

        batch_cur.batch = (size_t)atoi(PQgetvalue(res, i, PQfnumber(res, "batch")));
        batch_cur.start_id = (size_t)atoi(PQgetvalue(res, i, PQfnumber(res, "start_id")));
        batch_list.push_back(batch_cur);
    }
    PQclear(res);

    return 0;
}

int Index_DB::GetLatestBatch(int &batch) {
    std::vector<batch_info_t> batch_list;
    int rc;

    rc = GetBatchList(batch_list);
    if (rc)
        return rc;

    if (batch_list.size() != 0)
        batch = batch_list[batch_list.size() - 1].batch;
    else
        batch = -1;

    return 0;
}

int Index_DB::SetSysConfig(system_conf_t &sys_conf) {
    PGresult *res;
    char sql_str[1024];

    sprintf(sql_str, "INSERT INTO system_orca(path_base_data, path_base_model, batch_max, \
            dim, nc, nsubc, code_size, nprobe, max_codes, efSearch, do_pruning) \
            VALUES(%s, %s, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu)",
            sys_conf.path_base_data, sys_conf.path_base_model, sys_conf.batch_max,
            sys_conf.dim, sys_conf.nc, sys_conf.nsubc, sys_conf.code_size,
            sys_conf.nprobe, sys_conf.max_codes, sys_conf.efSearch, sys_conf.do_pruning);
    std::cout << "Setup path info with SQL: " << sql_str << std::endl;
    return CmdWithTrans(sql_str);
}

int Index_DB::GetSysConfig(system_conf_t &sys_conf) {
    PGresult *res;
    int rc = -1;
    int nRows;

    res = PQexec(conn, "SELECT * FROM system_orca");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cout << "Failed to retrieve data from system_orca table: " << PQerrorMessage(conn) << std::endl;
        goto out;
    }

    nRows = PQntuples(res);
    if (nRows == 0) {
        std::cout << "system_orca table is empty, BUG" << std::endl;
        goto out;
    }

    strcpy(sys_conf.path_base_data, PQgetvalue(res, 0, PQfnumber(res, "path_base_data")));
    strcpy(sys_conf.path_base_model, PQgetvalue(res, 0, PQfnumber(res, "path_base_model")));
    sys_conf.batch_max = atoi(PQgetvalue(res, 0, PQfnumber(res, "batch_max")));
    sys_conf.dim = atoi(PQgetvalue(res, 0, PQfnumber(res, "dim")));
    sys_conf.nc = atoi(PQgetvalue(res, 0, PQfnumber(res, "nc")));
    sys_conf.nsubc = atoi(PQgetvalue(res, 0, PQfnumber(res, "nsubc")));
    sys_conf.code_size = atoi(PQgetvalue(res, 0, PQfnumber(res, "code_size")));
    sys_conf.nprobe    = atoi(PQgetvalue(res, 0, PQfnumber(res, "nprobe")));
    sys_conf.max_codes = atoi(PQgetvalue(res, 0, PQfnumber(res, "max_codes")));
    sys_conf.efSearch  = atoi(PQgetvalue(res, 0, PQfnumber(res, "efSearch")));
    sys_conf.do_pruning = atoi(PQgetvalue(res, 0, PQfnumber(res, "do_pruning")));

out:
    PQclear(res);

    return 0;
}


int Index_DB::CmdWithTrans(char *sql_str) {
    int rc = -1;
    PGresult *res;

    res = PQexec(conn, "BEGIN");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        cout << "BEGIN command failed" << endl;
        PQclear(res);
        return rc;
    }
    PQclear(res);

    res = PQexec(conn, sql_str);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        cout << "Failed to execute command: " << sql_str << endl;
        std::cout << "Error of: " << PQerrorMessage(conn) << std::endl;
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

int Index_DB::AllocateBatch(size_t batch, size_t start_id) {
    char sql_str[512];

    sprintf(sql_str, "INSERT INTO batch_info(batch, start_id, ts, valid, no_precomputed_idx) \
            VALUES(%lu, %lu, NOW()::TIMESTAMP, FALSE, TRUE)",
            batch, start_id);
    return CmdWithTrans(sql_str);
}

int Index_DB::ActiveBatch(size_t batch) {
    char sql_str[512];

    sprintf(sql_str, "UPDATE batch_info SET valid = TRUE WHERE batch = %lu", batch);
    return CmdWithTrans(sql_str);
}

int Index_DB::ActivePrecomputedIndex(size_t batch) {
    char sql_str[512];

    sprintf(sql_str, "UPDATE batch_info SET no_precomputed_idx = FALSE WHERE batch = %lu", batch);
    return CmdWithTrans(sql_str);
}

int Index_DB::AppendIndexInfo(size_t idx_ver, size_t batch_start, size_t batch_end) {
    int rc = -1;
    PGresult *res;
    char sql_str[1024];
    char bool_var[8];

    sprintf(sql_str, "INSERT INTO index_info(ver, batch_start, batch_end) VALUES(%lu, %lu, %lu)",
            idx_ver, batch_start, batch_end);
    std::cout << "Append Index Info with SQL: " << sql_str << std::endl;
    return CmdWithTrans(sql_str);
}

int Index_DB::GetLatestIndexInfo(size_t &ver, size_t &batch_start, size_t &batch_end) {
    PGresult *res = PQexec(conn, "SELECT * FROM index_info ORDER BY ver DESC LIMIT 1");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        cout << "Failed to retrieve data from index_info table" << endl;
        PQclear(res);
        return -1;
    }

    int rows = PQntuples(res);
    /*
     * No records in index_info table, it means there have no index yet
     * service can not provide query service
     */
    if (rows == 0) {
        ver = 0;
        return 0;
    } else {
        ver = atoi(PQgetvalue(res, 0, PQfnumber(res, "ver")));
        batch_start = atoi(PQgetvalue(res, 0, PQfnumber(res, "batch_start")));
        batch_end = atoi(PQgetvalue(res, 0, PQfnumber(res, "batch_end")));
    }
    PQclear(res);

    return 0;
}

int Index_DB::AppendPQInfo(size_t ver, bool with_opq, size_t nsubc) {
    int rc = -1;
    PGresult *res;
    char sql_str[1024];
    char bool_var[8];

    if (with_opq) {
        strcpy(bool_var, "TRUE");
    } else {
        strcpy(bool_var, "FALSE");
    }

    sprintf(sql_str, "INSERT INTO pq_info(ver, with_opq, nsubc) VALUES(%lu, %s, %lu)",
            ver, bool_var, nsubc);
    std::cout << "Append PQ Info with SQL: " << sql_str << std::endl;
    return CmdWithTrans(sql_str);
}

int Index_DB::GetLatestPQInfo(size_t &ver, bool &with_opq, size_t &code_size, size_t &nsubc) {
    PGresult *res = PQexec(conn, "SELECT * FROM pq_info ORDER BY ver DESC LIMIT 1");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        cout << "Failed to retrieve data from vector_id_base table" << endl;
        PQclear(res);
        return -1;
    }

    int rows = PQntuples(res);
    /*
     * No records in pq_info table, it means it's new setup to run service
     */
    if (rows == 0) {
        code_size = 0;
        return 0;
    } else {
        ver = atoi(PQgetvalue(res, 0, PQfnumber(res, "ver")));

        // TODO: need to verify if correct to process booleand type like this
        char *sb = PQgetvalue(res, 0, PQfnumber(res, "with_opq"));
        if (sb[0] == 'f' && sb[1] == '\0')
            with_opq = false;
        else
            with_opq = true;

        code_size = atoi(PQgetvalue(res, 0, PQfnumber(res, "code_size")));
        nsubc = atoi(PQgetvalue(res, 0, PQfnumber(res, "nsubc")));
    }
    PQclear(res);

    return 0;
}

int Index_DB::AppendPQConf(pq_conf_t &pq_conf) {
    int rc = -1;
    PGresult *res;
    char sql_str[1024];

    sprintf(sql_str, "INSERT INTO pq_conf(ver, with_opq, M, efConstruction) VALUES(%lu, %s, %lu, %lu)",
            pq_conf.ver, pq_conf.with_opq ? "TRUE" : "FALSE", pq_conf.M, pq_conf.efConstruction);
    std::cout << "Append PQ Info with SQL: " << sql_str << std::endl;
    return CmdWithTrans(sql_str);
}

int Index_DB::GetLatestPQConf(pq_conf_t &pq_conf) {
    PGresult *res = PQexec(conn, "SELECT * FROM pq_conf ORDER BY ver DESC LIMIT 1");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        cout << "Failed to retrieve data from vector_id_base table" << endl;
        PQclear(res);
        return -1;
    }

    int rows = PQntuples(res);
    /*
     * No records in pq_info table, it means it's new setup to run service
     */
    if (rows == 0) {
        pq_conf.M = 0;
        return 0;
    } else {
        pq_conf.ver = atoi(PQgetvalue(res, 0, PQfnumber(res, "ver")));

        // TODO: need to verify if correct to process booleand type like this
        char *sb = PQgetvalue(res, 0, PQfnumber(res, "with_opq"));
        if (sb[0] == 'f' && sb[1] == '\0')
            pq_conf.with_opq = false;
        else
            pq_conf.with_opq = true;

        pq_conf.M = atoi(PQgetvalue(res, 0, PQfnumber(res, "M")));
        pq_conf.efConstruction = atoi(PQgetvalue(res, 0, PQfnumber(res, "efConstruction")));
        pq_conf.nt = atoi(PQgetvalue(res, 0, PQfnumber(res, "nt")));
        pq_conf.nsubt = atoi(PQgetvalue(res, 0, PQfnumber(res, "nsubt")));
    }
    PQclear(res);

    return 0;
}

} // namespace ivfhnsw
