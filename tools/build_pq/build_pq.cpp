/*
 * build_pq.cpp
 *
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>
#include <ivf-hnsw/hnswalg.h>
#include <ivf-hnsw/utils.h>

#include "nlohmann/json.hpp"

using nlohmann::json;
using std::cout;
using std::endl;

using namespace hnswlib;
using namespace ivfhnsw;

typedef struct build_param {
    char path_centroids[1024];
    char path_model[512];
    char path_info[1024];
    char path_edges[1024];
    char path_learn[1024];
    size_t M;
    size_t efConstruction;
    size_t dim;
    size_t nc;
    size_t nsubc;
    size_t code_size;
    size_t ver;
    bool do_opq;
    double ratio_train;
} build_param_t;

static int
read_conf(char *conf_file, json &json_conf) {
    int rc = -1;
    std::ifstream in_conf(conf_file);

    if (!in_conf.is_open()) {
        cout << "Failed to open configure file: " << conf_file << endl;
        goto out;
    }

    // parse conf file
    try {
        json_conf = json::parse(in_conf);
        rc = 0;
    } catch(const json::exception & e) {
        cout << "Failed to parse configure file: " << conf_file << endl;
        cout << e.what() << endl;
    }
    in_conf.close();

out:
    return rc;
}

static void
usage_help(char *cmd)
{
    cout << cmd << " " << "[conf file(json format) path]" << endl;
    exit (1);
}

static build_param_t pq_build_param;

static void
dump_conf()
{
    cout << endl;
    cout << "Show PQ build parameters **************" << endl << endl;
    cout << "path_centroids : " << pq_build_param.path_centroids << endl;
    cout << "path_model : " << pq_build_param.path_model << endl;
    cout << "path_info : " << pq_build_param.path_info << endl;
    cout << "path_edges : " << pq_build_param.path_edges << endl;
    cout << "path_learn : " << pq_build_param.path_learn << endl;

    cout << "efConstruction : " << pq_build_param.efConstruction << endl;
    cout << "code_size : " << pq_build_param.code_size << endl;
    cout << "ver : " << pq_build_param.ver << endl;
    cout << "nc : " << pq_build_param.nc << endl;
    cout << "nsubc : " << pq_build_param.nsubc << endl;
    cout << "M : " << pq_build_param.M << endl;
    cout << "dim : " << pq_build_param.dim << endl;

    cout << "ratio_train : " << pq_build_param.ratio_train << endl;

    if (pq_build_param.do_opq)
        cout << "do_opq : true" << endl;
    else
        cout << "do_opq : false" << endl;

    cout << endl;
    cout << "Done **************" << endl;
}

static int
get_conf(json &json_conf)
{
    std::string str_v;

    if (json_conf["path_centroids"].is_null()) {
        cout << "path_centroids section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("path_centroids", "");
    strcpy(pq_build_param.path_centroids, str_v.c_str());

    if (json_conf["path_learn"].is_null()) {
        cout << "path_learn section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("path_learn", "");
    strcpy(pq_build_param.path_learn, str_v.c_str());

    if (json_conf["path_model"].is_null()) {
        cout << "path_model section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("path_model", "");
    strcpy(pq_build_param.path_model, str_v.c_str());

    if (json_conf["do_opq"].is_null()) {
        cout << "do_opq section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("do_opq", "");
    if (strcasecmp(str_v.c_str(), "false") == 0)
        pq_build_param.do_opq = false;
    else if (strcasecmp(str_v.c_str(), "true") == 0)
        pq_build_param.do_opq = true;
    else {
        cout << "invalid do_opq section value " << str_v << " in conf file" << endl;
        return -1;
    }

    if (json_conf["efConstruction"].is_null()) {
        cout << "efConstruction section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("efConstruction", "");
    pq_build_param.efConstruction = atoi(str_v.c_str());

    if (json_conf["code_size"].is_null()) {
        cout << "code_size section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("code_size", "");
    pq_build_param.code_size = atoi(str_v.c_str());

    if (json_conf["ver"].is_null()) {
        cout << "ver section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("ver", "");
    pq_build_param.ver = atoi(str_v.c_str());

    if (json_conf["nc"].is_null()) {
        cout << "nc section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("nc", "");
    pq_build_param.nc = atoi(str_v.c_str());

    if (json_conf["nsubc"].is_null()) {
        cout << "nsubc section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("nsubc", "");
    pq_build_param.nsubc = atoi(str_v.c_str());

    if (json_conf["M"].is_null()) {
        cout << "M section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("M", "");
    pq_build_param.M = atoi(str_v.c_str());

    if (json_conf["dim"].is_null()) {
        cout << "dim section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("dim", "");
    pq_build_param.dim = atoi(str_v.c_str());

    if (json_conf["ratio_train"].is_null()) {
        cout << "ratio_train section not found in conf file" << endl;
        return -1;
    }
    str_v = json_conf.value("ratio_train", "");
    pq_build_param.ratio_train = atof(str_v.c_str());

    sprintf(pq_build_param.path_edges, "%s/%lu/hnsw_M%lu_ef%lu.ivecs",
            pq_build_param.path_model,
            pq_build_param.ver,
            pq_build_param.M,
            pq_build_param.efConstruction);

    sprintf(pq_build_param.path_info, "%s/%lu/hnsw_M%lu_ef%lu.bin",
            pq_build_param.path_model,
            pq_build_param.ver,
            pq_build_param.M,
            pq_build_param.efConstruction);

    return 0;
}

static int
path_quantizer_check()
{
    if (!exists(pq_build_param.path_centroids)) {
        cout << "Can not find path: " << pq_build_param.path_centroids << endl;
        return -1;
    }

    if (exists(pq_build_param.path_edges)) {
        cout << "File " << pq_build_param.path_edges << "already exist" << endl;
        return -1;
    }

    if (exists(pq_build_param.path_info)) {
        cout << "File " << pq_build_param.path_info << "already exist" << endl;
        return -1;
    }

    return 0;
}


static int
path_pq_check()
{
    char path_model_ver[1024];

    if (!exists(pq_build_param.path_model)) {
        cout << "Can not find path: " << pq_build_param.path_model << endl;
        return -1;
    }

    sprintf(path_model_ver, "%s/%lu", pq_build_param.path_model, pq_build_param.ver);
    if (!exists(path_model_ver)) {
        int rc = mkdir_p(path_model_ver, 0755);
        if (rc) {
            cout << "Failed to create directory: " << path_model_ver << endl;
            return -1;
        }
    }

    if (exists(pq_build_param.path_edges)) {
        cout << "File " << pq_build_param.path_edges << "already exist" << endl;
        return -1;
    }

    if (exists(pq_build_param.path_info)) {
        cout << "File " << pq_build_param.path_info << "already exist" << endl;
        return -1;
    }

    return 0;
}

static int
path_check()
{
    int rc;

    rc = path_quantizer_check();
    if (rc) goto out;

    rc = path_pq_check();
    if (rc) goto out;

out:
    return rc;
}

int main(int argc, char **argv) {
    int rc;

    if (argc != 2) usage_help (argv[0]);

    json json_conf;
    rc = read_conf(argv[1], json_conf);
    if (rc) return rc;

    rc = get_conf(json_conf);
    if (rc) return rc;

    rc = path_check();
    if (rc) return rc;

    dump_conf();
    // exit(1);

    //==================
    // Initialize Index
    //==================
    IndexIVF_HNSW_Grouping *index = new IndexIVF_HNSW_Grouping(pq_build_param.dim,
                                                               pq_build_param.nc,
                                                               pq_build_param.code_size,
                                                               8,
                                                               pq_build_param.nsubc);

    index->build_quantizer(pq_build_param.path_centroids,
                           pq_build_param.path_info,
                           pq_build_param.path_edges,
                           pq_build_param.M,
                           pq_build_param.efConstruction);

    index->do_opq = pq_build_param.do_opq;

    //==========
    // Train PQ
    //==========
    index->build_pq_files(pq_build_param.path_learn,
                          pq_build_param.path_model,
                          pq_build_param.ver,
                          pq_build_param.do_opq,
                          pq_build_param.code_size,
                          pq_build_param.ratio_train,
                          pq_build_param.nsubc);

    index->append_pq_info(pq_build_param.ver,
                          pq_build_param.do_opq,
                          pq_build_param.code_size,
                          pq_build_param.nsubc);

    delete index;

    return 0;
}



