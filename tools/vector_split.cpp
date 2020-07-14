/*
 * vector_split.cpp
 *
 * Used to split base vector file into 10 parts
 *
 * current code only support SIFT1B data
 *
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

using std::cout;
using std::endl;
using std::string;

size_t split_count;
const size_t vector_dim = 128;
const size_t batch_num = 1000000;

static void
usage_help(char *cmd)
{
    cout << cmd << " " << "[base path]" << " " << "[split path]" << " " <<  "[splits_num(10/20/50)]" << endl;
    exit (1);
}

static int
cleanup_split(char *path)
{
    std::vector<string> file_list;
    DIR *dir;
    struct dirent *entry;

    // get file list in directory
    if ((dir = opendir(path)) == NULL) {
        cout << "Failed to open dir: " << path << endl;
        return -1;
    } else {
        while ((entry = readdir(dir)) != NULL)
            file_list.push_back(entry->d_name);
        closedir(dir);
    }

    // remove files in list
    for (auto it = file_list.begin(); it < file_list.end(); it++) {
        auto f_name = *it;
        auto f_path = path + f_name;
        int rc = unlink(f_path.c_str());
        if (!rc) {
            cout << "Failed to delete file: " << f_path << endl;
            return -1;
        }
    }

    // remove dir
    int rc = rmdir(path);
    if (rc) {
        cout << "Failed to delete dir: " << path << endl;
        return -1;
    }

    return 0;
}

static size_t
file_size(char *path_full)
{
    struct stat st;

    int rc = stat(path_full, &st);
    if (rc) {
        cout << "Failed to get attribute from file: " << path_full << endl;
        exit (-1);
    }

    return st.st_size;
}

static void
get_basename(char *file_nm, char *file_base)
{
    char tmp_nm[512];
    strcpy(tmp_nm, file_nm);
    char *ptr = strstr(tmp_nm, ".");
    if (ptr == NULL) return;

    *ptr = '\0';
    strcpy(file_base, tmp_nm);
}

static int
split_base(char *path_in, char *path_out)
{
    char *base_file = (char *)"bigann_base.bvecs";
    char base_nm[512];
    char full_path[1024];

    base_nm[0] = '\0';
    get_basename(base_file, base_nm);
    if (base_nm[0] == '\0') return -1;

    sprintf(full_path, "%s/%s", path_in, base_file);
    size_t sz_base = file_size(full_path);
    size_t sz_split = sz_base / split_count;

    FILE *fp_in, *fp_out;
    fp_in = fopen(full_path, "r+");
    if (fp_in == NULL) {
        cout << "Failed to open file: " << full_path << endl;
        return -1;
    }

    size_t rec_size = sizeof(uint32_t) + (vector_dim * sizeof(uint8_t));
    char *buf_in = (char *)malloc(batch_num * rec_size);
    if (buf_in == NULL) {
        cout << "Failed to allocate memory for read buffer" << endl;
        fclose(fp_in);
        return -1;
    }

    for (size_t i = 0; i < split_count; i++) {
        size_t d_read;
        size_t d_left = sz_split;

        sprintf(full_path, "%s/%s_%02lu%s", path_out, base_nm, i, ".bvecs");
        fp_out = fopen(full_path, "w+");
        if (fp_out == NULL) {
            cout << "Failed to open file: " << full_path << endl;
            fclose(fp_in);
            return -1;
        }
        while (!feof(fp_in) && !ferror(fp_in)) {
            d_read = fread(buf_in, 1, rec_size * batch_num, fp_in);
            if (d_read == 0 || (d_read % rec_size) != 0) {
                cout << "Failed to read file: " << base_file << endl;
                fclose(fp_in);
                return -1;
            }
            fwrite(buf_in, rec_size, d_read/rec_size, fp_out);
            d_left -= d_read;
            if (d_left == 0) {
                fclose(fp_out);
                cout << "Split segment " << i << " done" << endl;
                // start next loop to create next split segment file
                break;
            }
        }

    }
    fclose(fp_in);
    free(buf_in);

    return 0;
}

static int
split_precomputed_idxs(char *path_in, char *path_out)
{
    char *base_file = (char *)"precomputed_idxs_sift1b.ivecs";
    char base_nm[512];
    char full_path[1024];

    base_nm[0] = '\0';
    get_basename(base_file, base_nm);
    if (base_nm[0] == '\0') return -1;

    sprintf(full_path, "%s/%s", path_in, base_file);
    size_t sz_base = file_size(full_path);
    size_t sz_split = sz_base / split_count;
    const size_t batch_size = 1000000;

    FILE *fp_in, *fp_out;
    fp_in = fopen(full_path, "r+");
    if (fp_in == NULL) {
        cout << "Failed to open file: " << full_path << endl;
        return -1;
    }

    size_t batch_dsize = sizeof(uint32_t) + (batch_size * sizeof(uint32_t));
    char *buf_in = (char *)malloc(batch_dsize);
    if (buf_in == NULL) {
        cout << "Failed to allocate memory for read buffer" << endl;
        fclose(fp_in);
        return -1;
    }

    for (size_t i = 0; i < split_count; i++) {
        size_t d_read;
        size_t d_left = sz_split;

        sprintf(full_path, "%s/%s_%02lu%s", path_out, base_nm, i, ".ivecs");
        fp_out = fopen(full_path, "w+");
        if (fp_out == NULL) {
            cout << "Failed to open file: " << full_path << endl;
            fclose(fp_in);
            return -1;
        }
        while (!feof(fp_in) && !ferror(fp_in)) {
            d_read = fread(buf_in, 1, batch_dsize, fp_in);
            if (d_read == 0 ) {
                cout << "Failed to read file: " << base_file << endl;
                fclose(fp_in);
                return -1;
            }
            fwrite(buf_in, batch_dsize, 1, fp_out);
            d_left -= d_read;
            if (d_left == 0) {
                fclose(fp_out);
                cout << "Split segment " << i << " done" << endl;
                // start next loop to create next split segment file
                break;
            }
        }

    }
    fclose(fp_in);
    free(buf_in);

    return 0;
}

static int
create_dir(char *path)
{
    if (!access(path, F_OK)) return 0;

    char run_cmd[1024];
    pid_t result;

    sprintf(run_cmd, "mkdir -p %s", path);
    result = system(run_cmd);

    if (-1 == result || !WIFEXITED(result) || 0 != WEXITSTATUS(result)){
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    char path_split[1024];
    int rc;

    if (argc != 4) usage_help (argv[0]);

    char *path_base = argv[1];
    char *path_split_base = argv[2];

    split_count = atoi(argv[3]);
    if (split_count != 10 && split_count != 20 && split_count != 50) {
        // only allow 10/20/50 split count
        // no reason, only want to use those number in vector add/remove test
        usage_help (argv[0]);
    }
    sprintf(path_split, "%s_%lu", path_split_base, split_count);
    rc = create_dir(path_split);
    if (rc) {
        cout << "Failed to create dir: " << path_split << endl;
        exit (1);
    }

    rc = split_base(path_base, path_split);
    if (rc) {
        cleanup_split(path_split);
        cout << "Failed to split base vector file" << endl;
        exit (1);
    }

    rc = split_precomputed_idxs(path_base, path_split);
    if (rc) {
        cleanup_split(path_split);
        cout << "Failed to split precomputed idxs file" << endl;
        exit (1);
    }

    return 0;
}


