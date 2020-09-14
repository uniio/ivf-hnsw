#include <faiss/ProductQuantizer.h>

int main()
{
        faiss::ProductQuantizer *pq = nullptr;
        faiss::ProductQuantizer *norm_pq = nullptr;

        pq = new faiss::ProductQuantizer(128, 16, 8);
        norm_pq = new faiss::ProductQuantizer(1, 1, 8);

        delete pq;
        delete norm_pq;

        return 0;
}
