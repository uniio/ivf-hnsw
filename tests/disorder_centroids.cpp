#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/utils.h>

using namespace ivfhnsw;
using std::ios;
using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::vector;
using std::string;

int dumpCentroid(const vector<float> &centroids, uint32_t n_dim, char *file_nm) {
    int rc = -1;
    auto centroids_left = centroids.size()/n_dim;
    auto centroids_p = centroids.data();
    ofstream of_s(file_nm, std::ios::binary);

    if (!of_s.is_open()) {
        cout << "Failed to open file: " << file_nm << endl;
        goto out;
    }

    while (centroids_left > 0) {
        try {
            of_s.write((char *) &n_dim, sizeof(uint32_t));
            of_s.write((char *) (centroids_p), n_dim*sizeof(float));
            centroids_p += n_dim;
            centroids_left--;
        } catch (const std::exception&) {
            cout << "Failed to write file: " << file_nm << endl;
            goto out;
        }
    }
    rc = 0;
    cout << "Success to generate centriods file: " << file_nm << endl;

out:
    if (!of_s.is_open()) {
        of_s.close();
    }

    if (rc) {
        // dont check return value, because file may does not exist
        unlink(file_nm);
    }

    return rc;
}

int main(int argc, char **argv) {
    char *path_centroids_in = "/mnt/nfs_shared/vector1b/SIFT1B/centroids_sift1b.fvecs";
    char *path_centroids_out = "./centroids_sift1b.fvecs";
    uint32_t nc = 993127;
    uint32_t dim = 128;
    vector<float> centroids_in(nc * dim);
    vector<float> centroids_out(nc * dim);

    cout << "Loading centroids from " << path_centroids_in << endl;
    ifstream fs_input(path_centroids_in, std::ios::binary);
    readXvec<float>(fs_input, centroids_in.data(), dim, nc);

    cout << "disorder centroids" << endl;
    random_subset(centroids_in.data(), centroids_out.data(), dim, nc, nc);

    cout << "Saving new centroids to " << path_centroids_out << endl;
    dumpCentroid(centroids_out, dim, path_centroids_out);

    return 0;
}

