#include <iostream>
#include <vector>
#include <unistd.h>
#include <ivf-hnsw/IndexIVF_DB.h>

using namespace ivfhnsw;

using std::cout;
using std::endl;
using std::vector;

const size_t batch_size = 100000;
const size_t batch_max  = 10;
const size_t vec_id_min = 0;

int main() {
    Index_DB*            db_p = nullptr;
    int                  rc;
    size_t               vec_id, batch_idx;
    vector<batch_info_t> batch_list;

    db_p = new Index_DB("localhost", 5432, "servicedb", "postgres", "postgres");
    rc   = db_p->Connect();
    if (rc) {
        cout << "Failed to connect to Database server" << endl;
        goto out;
    }

    // make batch_info table empty
    rc = db_p->CleanupBatch();
    if (rc) {
        cout << "Failed to cleanup system batch info" << endl;
        goto out;
    }

    // allocate batch vector file and write it
    vec_id = vec_id_min;
    for (batch_idx = 0; batch_idx < batch_max; batch_idx++) {
        cout << "Allocate batch " << batch_idx << endl;
        rc = db_p->AllocateBatch(batch_idx, vec_id);
        if (rc) {
            cout << "Failed to allocate batch " << batch_idx << " for vectors write" << endl;
            goto out;
        }

        // write vector file behavior
        cout << "Write vectors to batch " << batch_idx << endl;
        sleep(1);

        db_p->ActiveBatch(batch_idx);
        if (rc) {
            cout << "Failed to commit batch " << batch_idx << endl;
            goto out;
        }
        cout << "Active batch " << batch_idx << endl;
        vec_id += batch_size;
    }

    // check batch records in test
    rc = db_p->GetBatchList(batch_list);
    if (rc) {
        cout << "Failed to get system batch info" << endl;
        goto out;
    }

    // batch records number is not we expected
    if (batch_list.size() != batch_max) {
        cout << "BUG! records not expected in batch_info table" << endl;
        goto out;
    }

    for (size_t i = 0; i < batch_list.size(); i++) {
        cout << "batch " << batch_list[i].batch << " with start vector id " << batch_list[i].start_id << endl;
    }

out:
    delete db_p;
    return rc;
}
