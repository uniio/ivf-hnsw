/*
  path_data_base is base directory to store 
  ts is timestamp when insert current record
  valid indicate the batch data is valid or invalid
*/

/*
 * TEXT type is best here, if use VARCHAR, we cant copy PQgetvalue result to destination
 * otherwise, it will trigger error: "*** buffer overflow detected ***: terminated"
*/
CREATE TABLE IF NOT EXISTS system_orca (
    path_base_data TEXT NOT NULL,
    path_base_model TEXT NOT NULL,
    batch_max INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    nc INTEGER NOT NULL,
    nsubc INTEGER NOT NULL,
    code_size INTEGER NOT NULL,
    nprobe INTEGER NOT NULL,
    max_codes INTEGER NOT NULL,
    efSearch INTEGER NOT NULL,
    do_pruning INTEGER NOT NULL
);

/*
  batch is batch number
  start_id is first vector id in the batch vector file
  ts is timestamp when insert current record
  valid indicate the batch data is valid or invalid
*/
CREATE TABLE IF NOT EXISTS batch_info (
    batch INTEGER NOT NULL,
    start_id INTEGER NOT NULL,
    ts TIMESTAMP NOT NULL,
    valid BOOLEAN NOT NULL,
    no_precomputed_idx BOOLEAN NOT NULL
);

/*
  ver           version of index
  batch_start   start of batch to build index
  batch_end     end of batch to build index
*/
CREATE TABLE IF NOT EXISTS index_info (
    ver INTEGER NOT NULL,
    batch_start INTEGER NOT NULL,
    batch_end INTEGER NOT NULL
);

/*
  ver is version of the PQ files
  with_opq is enable/disable opq encoding when build PQ files
  nsubc is number of subcentroids per group
*/
CREATE TABLE IF NOT EXISTS pq_info (
    ver INTEGER NOT NULL,
    with_opq BOOLEAN NOT NULL,
    nsubc INTEGER NOT NULL
);

/*
  ver is version of build PQ configuration
  with_opq is enable/disable opq encoding when build PQ files
  M is Min number of edges per point when build PQ
  nt is number of training vector
  nsubt is number of learn vectors to train (random subset of the learn set)
*/
CREATE TABLE IF NOT EXISTS pq_conf (
    ver INTEGER NOT NULL,
    with_opq BOOLEAN NOT NULL,
    M INTEGER NOT NULL,
    efConstruction INTEGER NOT NULL,
    nt INTEGER NOT NULL,
    nsubt INTEGER NOT NULL
);

/*
  Setup system_orca table
*/
INSERT INTO system_orca(path_base_data, path_base_model, batch_max, dim, nc, nsubc, code_size, nprobe, max_codes, efSearch, do_pruning) VALUES('/mnt/hdd_strip/orcv_search/data', '/mnt/hdd_strip/orcv_search/models', 1000, 128, 993127, 64, 16, 32, 10000, 80, 1);

/*
  Setup pq_conf table
*/
INSERT INTO pq_conf(ver, with_opq, M, efConstruction, nt, nsubt) VALUES(1, TRUE, 16, 210, 10000000, 262144);

/*
  Setup batch_info table, which will be used by precomputed index build
*/
/*
INSERT INTO batch_info("batch, start_id, ts, valid, no_precomputed_idx) VALUES(0, 0, current_timestamp, TRUE, TRUE);
INSERT INTO batch_info("batch, start_id, ts, valid, no_precomputed_idx) VALUES(1, 0, current_timestamp, TRUE, TRUE);
INSERT INTO batch_info("batch, start_id, ts, valid, no_precomputed_idx) VALUES(2, 0, current_timestamp, TRUE, TRUE);
INSERT INTO batch_info("batch, start_id, ts, valid, no_precomputed_idx) VALUES(3, 0, current_timestamp, TRUE, TRUE);
INSERT INTO batch_info("batch, start_id, ts, valid, no_precomputed_idx) VALUES(4, 0, current_timestamp, TRUE, TRUE);
INSERT INTO batch_info("batch, start_id, ts, valid, no_precomputed_idx) VALUES(5, 0, current_timestamp, TRUE, TRUE);
*/
