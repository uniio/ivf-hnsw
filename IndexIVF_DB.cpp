/*
 * IndexIVF_DB.cpp
 *
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <postgresql/libpq-fe.h>
#include "IndexIVF_DB.h"

using std::cout;
using std::endl;

Index_DB::Index_DB(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr) {
    sprintf(conninfo, "host=%s port=%u dbname=%s user=%s password=%s",
            host, port, db_nm, db_usr, pwd_usr);
}

Index_DB::~Index_DB() {
    if (conn) PQfinish(conn);
}

int Index_DB::Connect() {
    int rc = 0;
    conn = PQconnectdb(conninfo);
    if (PQstatus(conn) != CONNECTION_OK) {
        cout << "connect failed. PQstatus : " << PQstatus(conn)  << endl;
        cout << PQerrorMessage(conn) << endl;
        PQfinish(conn);
        conn = nullptr;
        rc = -1;
    }

    return rc;
}



int Index_DB::CreateTable(const char *cmd_str, const char *table_nm) {
    PGresult   *res;
    int rc = 0;

    cout << "Execute command: " << cmd_str << endl;
    res = PQexecParams(conn,
                       cmd_str,
                       0,
                       NULL,
                       NULL,
                       NULL,
                       NULL,
                       1);

    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        cout << "Failed to create table: " << table_nm << endl;
        cout << PQerrorMessage(conn) << endl;
        rc = -1;
    }
    PQclear(res);

    return rc;
}

int Index_DB::CreateBaseTables(size_t batch_idx) {
    PGresult *res;
    char sql_str[1024], tbl_nm[128];
    int rc = -1;

    // create vector index table
    sprintf(tbl_nm, "bigann_base_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (vec_id INTEGER, dim INTEGER, vec bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;
}

int Index_DB::CreatePrecomputedIndexTables(size_t batch_idx) {
    PGresult *res;
    char sql_str[1024], tbl_nm[128];
    int rc = -1;

    // create vector index table
    sprintf(tbl_nm, "precomputed_idxs_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (batch_size INTEGER, idxs bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;
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
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;

    // create PQ codec table
    sprintf(tbl_nm, "pq_codec_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (dim INTEGER, codes bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;

    // create norm PQ codec table
    sprintf(tbl_nm, "norm_codec_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (dim INTEGER, norm_codes bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;

    // create NN centriods index table
    sprintf(tbl_nm, "nn_centroid_idxs_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (dim INTEGER, nn_centroid_idxs bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;

    // create group size table
    sprintf(tbl_nm, "subgroup_sizes_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (dim INTEGER, subgroup_sizes bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;

    // create inter centroid distances table
    sprintf(tbl_nm, "inter_centroid_dists_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (dim INTEGER, inter_centroid_dists bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
    if (rc) return rc;

    // create msic table for alphas and centroid_norms
    sprintf(tbl_nm, "misc_%lu", batch_idx);
    sprintf(sql_str,
            "CREATE TABLE %s (size INTEGER, misc_data bytea)",
            tbl_nm);
    rc = CreateTable(sql_str, tbl_nm);
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

int Index_DB::DropBaseTable(size_t batch_idx, bool drop_older) {
    char tbl_nm[128];
    int rc;

    // drop base table of given batch
    sprintf(tbl_nm, "bigann_base_%lu", batch_idx);
    rc = DropTable(tbl_nm);
    if (rc) return rc;

    // drop previous base table, when we disuse old vecotr data anymore
    if (drop_older) {
        for (size_t i = 0; i < batch_idx; i++) {
            sprintf(tbl_nm, "bigann_base_%lu", i);
            rc = DropTable(tbl_nm);
            if (rc) return rc;
        }
    }

    return rc;
}

int Index_DB::DropPrecomputedIndexTable(size_t batch_idx, bool drop_older) {
    char tbl_nm[128];
    int rc;

    // drop precomputed index table of given batch
    sprintf(tbl_nm, "precomputed_idxs_%lu", batch_idx);
    rc = DropTable(tbl_nm);
    if (rc) return rc;

    // drop previous precomputed index table, when we disuse old vecotr data anymore
    if (drop_older) {
        for (size_t i = 0; i < batch_idx; i++) {
            sprintf(tbl_nm, "precomputed_idxs_%lu", i);
            rc = DropTable(tbl_nm);
            if (rc) return rc;
        }
    }

    return rc;
}

void Index_DB::DropIndexTables(size_t batch_idx) {
    char tbl_nm[128];
    int rc = -1;

    // drop index vector table
    sprintf(tbl_nm, "index_vector_%lu", batch_idx);
    DropTable(tbl_nm);

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



int Index_DB::WriteIndexMeta(size_t dim, size_t nc, size_t nsubc) {
    int rc = -1;
    PGresult *res;
    char sql_str[512];

    sprintf(sql_str, "INSERT index_meta SET dim=%u nc=%lu nsubc=%lu",
            dim, nc, nsubc);
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

int Index_DB::LoadIndexMeta(size_t &dim, size_t &nc, size_t &nsubc) {
    PGresult* res;

    res = PQexec(conn, "DECLARE mycursor CURSOR FOR select * from index_meta");
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        std::cout << "DECLARE CURSOR failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return -1;
    }
    PQclear(res);

    // get all result from database
    res = PQexec(conn, "FETCH ALL in mycursor");
    if (PQresultStatus(res) != PGRES_TUPLES_OK)
    {
        std::cout << "FETCH ALL failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return -1;
    }

    int nFields = PQnfields(res);
    int nRows = PQntuples(res);
    if (nRows != 1) {
        std::cout << "Not expected records count in table: index_meta" << std::endl;
        PQclear(res);
        return -1;
    }
    if (nFields != 3) {
        std::cout << "Not expected fields number in table: index_meta" << std::endl;
        PQclear(res);
        return -1;
    }

    dim = atoi(PQgetvalue(res, 0, PQfnumber(res, "dim")));
    nc = atoi(PQgetvalue(res, 0, PQfnumber(res, "nc")));
    nsubc = atoi(PQgetvalue(res, 0, PQfnumber(res, "nsubc")));

    PQclear(res);

    return 0;
}

int Index_DB::UpdateBaseId(size_t id_base, bool init_stage) {
    int rc = 0;
    char sql_str[512];

    PGresult *res = PQexec(conn, "BEGIN");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) rc = -1;
    PQclear(res);
    if (rc) {
        cout << "BEGIN command failed" << endl;
        goto out;
    }

    if (init_stage) {
        sprintf(sql_str, "INSERT INTO vector_id_base VALUES(%lu)",
                id_base);
    } else {
        sprintf(sql_str, "UPDATE vector_id_base SET id_base=%lu WHERE id_base=%lu",
                id_base + 1, id_base);
    }

    if (PQresultStatus(res) != PGRES_COMMAND_OK) rc = -1;
    PQclear(res);
    if (rc) {
        if (init_stage) {
            cout << "UPDATE command failed" << endl;
        } else {
            cout << "UPDATE command failed" << endl;
        }

        goto out;
    }

    res = PQexec(conn, "COMMIT");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) rc = -1;
    PQclear(res);
    if (rc) {
        cout << "COMMIT command failed" << endl;
    }

out:
    return rc;
}

int Index_DB::GetBaseId(size_t &id_base) {
    PGresult *res = PQexec(conn, "SELECT * FROM vector_id_base LIMIT 1");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        cout << "Failed to retrieve data from vector_id_base table" << endl;
        PQclear(res);
        return -1;
    }

    int rows = PQntuples(res);
    /*
     * No records in vector_id_base table, it means it's new setup to run service
     * Insert record with id_base == 0
     * */
    if (rows == 0) {
        id_base = 0;
    } else {
        auto sval = PQgetvalue(res, 0, 0);
        id_base = atoi(sval);
    }
    PQclear(res);

    return 0;
}

int Index_DB::GetLatestBatch(size_t &batch_idx) {
    PGresult* res;

    res = PQexec(conn, "DECLARE mycursor CURSOR FOR select * from system");
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
        std::cout << "DECLARE CURSOR failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return -1;
    }
    PQclear(res);

    // get all result from database
    res = PQexec(conn, "FETCH ALL in mycursor");
    if (PQresultStatus(res) != PGRES_TUPLES_OK)
    {
        std::cout << "FETCH ALL failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return -1;
    }

    int nFields = PQnfields(res);
    int nRows = PQntuples(res);
    if (nRows != 1) {
        std::cout << "Not expected records count in table: system" << std::endl;
        PQclear(res);
        return -1;
    }
    if (nFields != 1) {
        std::cout << "Not expected fields number in table: system" << std::endl;
        PQclear(res);
        return -1;
    }

    batch_idx = atoi(PQgetvalue(res, 0, PQfnumber(res, "batch_idx")));
    PQclear(res);

    return 0;
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

    sprintf(sql_str, "INSERT INTO system(batch_idx) VALUES(%lu)", batch_idx);
    res = PQexec(conn, sql_str);
    cout << "INSERT result: " << PQresultStatus(res) << endl;
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        cout << "INSERT command failed" << endl;
        std::cout << "Error of: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return rc;
    }
    PQclear(res);

    sprintf(sql_str, "DELETE FROM system WHERE batch_idx<%u", batch_idx);
    res = PQexec(conn, sql_str);
    cout << "DELETE result: " << PQresultStatus(res) << endl;
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        cout << "DELETE command failed" << endl;
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



