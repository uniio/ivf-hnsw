CREATE TABLE IF NOT EXISTS system (
    batch INTEGER NOT NULL,
    ts TIMESTAMP NOT NULL
);

/*
  batch is batch number
  ts is timestamp when insert current record
*/

CREATE TABLE IF NOT EXISTS index_meta (
    dim INTEGER,
    nc INTEGER,
    nsubc INTEGER
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
