/*
  path_data_base is base directory to store 
  ts is timestamp when insert current record
  valid indicate the batch data is valid or invalid
*/

CREATE TABLE IF NOT EXISTS system (
    path_base_data CHARACTER(1024) NOT NULL;
    path_base_model CHARACTER(1024) NOT NULL;
    batch_max INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    nc INTEGER NOT NULL,
    nsubc INTEGER NOT NULL,
    code_size INTEGER NOT NULL
);

/*
  batch is batch number
  ts is timestamp when insert current record
  valid indicate the batch data is valid or invalid
*/
CREATE TABLE IF NOT EXISTS batch_info (
    batch INTEGER NOT NULL,
    ts TIMESTAMP NOT NULL,
    valid BOOLEAN NOT NULL
);

/*
  batch is batch number
  ts is timestamp when insert current record
*/
CREATE TABLE IF NOT EXISTS index_meta (
    dim INTEGER NOT NULL,
    nc INTEGER NOT NULL,
    nsubc INTEGER NOT NULL
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
  path is path to store PQ files
  ver is version of the PQ files
  with_opq is enable/disable opq encoding when build PQ files
  code_size is code size per vector in bytes
  nsubc is number of subcentroids per group
*/
CREATE TABLE IF NOT EXISTS pq_info (
    path VARCHAR(1024) NOT NULL,
    ver INTEGER NOT NULL,
    with_opq BOOLEAN NOT NULL,
    code_size INTEGER NOT NULL,
    nsubc INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pq_conf (
    path VARCHAR(1024) NOT NULL,
    ver INTEGER NOT NULL,
    with_opq BOOLEAN NOT NULL,
    code_size INTEGER NOT NULL,
    nsubc INTEGER NOT NULL
);
