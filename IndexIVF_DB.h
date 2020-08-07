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

class Index_DB {
private:
    typedef uint32_t idx_t;
    char conninfo[512];
    PGconn *conn = nullptr;
public:
    explicit Index_DB(char *host, uint32_t port, char *db_nm, char *db_usr, char *pwd_usr);
    virtual ~Index_DB();

    int Connect();
    int CreateIndexTables(size_t batch);
    void DropIndexTables(size_t batch);
    int CreateBaseTable(size_t batch);
    int DropBaseTables(std::vector<size_t> batchs);
    int DropBaseTable(size_t batch, bool drop_older = false);
    int CreatePrecomputedIndexTables(size_t batch);
    int DropPrecomputedIndexTable(size_t batch, bool drop_older = false);
    int WriteIndexMeta(size_t dim, size_t nc, size_t nsubc);
    int LoadIndexMeta(size_t &dim, size_t &nc, size_t &nsubc);
    int LoadIndex(size_t batch);
    template<typename T>
    int ReadVectors(char *table_nm, size_t num_centroids, std::vector<std::vector<idx_t>> &dvec) {
        PGresult* res;
        char sql_str[512];

        sprintf(sql_str, "DECLARE mycursor CURSOR FOR select * from %s", table_nm);
        res = PQexec(conn, sql_str);
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
        if ((size_t)nRows != num_centroids) {
            std::cout << "Not expected records count: " << num_centroids << " in table: " << table_nm << std::endl;
            PQclear(res);
            return -1;
        }

        dvec.resize(nRows);
        for (int i = 0; i < nRows; i++)
        {
            auto ivec = dvec[i];
            auto vsz = atoi(PQgetvalue(res, i, 0));
            ivec.resize(vsz);

            /*
             * The binary representation of BYTEA is a bunch of bytes, which could
             * include embedded nulls so we have to pay attention to field length.
             */
            auto bptr = PQgetvalue(res, i, 1);
            auto blen = PQgetlength(res, i, 1);
            memcpy(ivec.data(), bptr, blen);
        }

        PQclear(res);

        return 0;
    }

    template<typename T>
    int WriteVector(char *table_nm, char *col0, char *col1, std::vector<T> &ivec) {
        int rc = 0;
        size_t sz = ivec.size();
        size_t dsize = sz * sizeof(T);

        PGresult* res;
        const uint32_t sz_big_endian = htonl((uint32_t)sz);
        const char* const paramValues[] = {
                                               reinterpret_cast<const char* const>(&sz_big_endian),
                                               reinterpret_cast<const char* const>(ivec.data())
                                          };
        const int paramLenghts[] = {sizeof(sz_big_endian), dsize};
        const int paramFormats[] = {1, 1}; /* binary */
        char sql_str[512];
        sprintf(sql_str,
                "INSERT INTO %s (%s, %s) VALUES ($1::integer, $2::bytea)",
                table_nm, col0, col1);

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

        if (PQresultStatus(res) != PGRES_COMMAND_OK)
        {
            std::cout << "Failed to insert data to table: " << table_nm << std::endl;
            std::cout << "Error of: " << PQerrorMessage(conn) << std::endl;
            rc = -1;
        }
        PQclear(res);

        return rc;
    }
    template<typename T>
    int WriteBaseVector(char *table_nm, size_t vec_id, std::vector<T> &ivec) {
        int rc = 0;
        size_t sz = ivec.size();
        size_t dsize = sz * sizeof(T);

        PGresult* res;
        const uint32_t eid_big_endian = htonl((uint32_t)vec_id);
        const uint32_t sz_big_endian = htonl((uint32_t)sz);
        const char* const paramValues[] = {
                reinterpret_cast<const char* const>(&eid_big_endian),
                                               reinterpret_cast<const char* const>(&sz_big_endian),
                                               reinterpret_cast<const char* const>(ivec.data())
                                          };
        const int paramLenghts[] = {sizeof(eid_big_endian), sizeof(sz_big_endian), dsize};
        const int paramFormats[] = {1, 1, 1}; /* binary */
        char sql_str[512];
        sprintf(sql_str,
                "INSERT INTO %s (vec_id, dim, vec) VALUES ($1::integer, $2::integer, $3::bytea)",
                table_nm);

        res = PQexecParams(
          conn,
          sql_str,
          3,
          NULL, /* Types of parameters, unused as casts will define types */
          paramValues,
          paramLenghts,
          paramFormats,
          1 // binary results
        );

        if (PQresultStatus(res) != PGRES_COMMAND_OK)
        {
            std::cout << "Failed to insert data to table: " << table_nm << std::endl;
            std::cout << "Error of: " << PQerrorMessage(conn) << std::endl;
            rc = -1;
        }
        PQclear(res);

        return rc;
    }
    int UpdateIndex(size_t batch);
    int GetBaseId(size_t &id_base);
    int UpdateBaseId(size_t id_base, bool init_stage);
    int AppendPQInfo(const char *path, size_t ver, bool with_opq, size_t code_size, size_t nsubc);
    int GetLatestPQInfo(char *path, size_t &ver, bool &with_opq, size_t &code_size, size_t &nsubc);
private:
    int GetLatestBatch(size_t &batch);
    int CreateTable(const char *cmd_str, const char *table_nm);
    int UpdateMeta(size_t batch);
    int DropTable(char *tbl_nm);
    int CmdWithTrans(char *sql_str);
};

#endif /* INDEXIVF_DB_H_ */
