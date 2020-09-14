#include <ivf-hnsw/IndexIVF_HNSW.h>

using namespace ivfhnsw;

int main()
{
        struct IndexIVF_HNSW *index = nullptr;

        index = new IndexIVF_HNSW(128, 993127, 16, 8, 65536);

        delete index;

        return 0;
}
