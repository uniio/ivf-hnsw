# faiss
# https://github.com/orcadt/faiss.git 

build with Release mode, and change default -O3 to -O2
-O3 will segment fault in faiss, even cannot pass ivf-hnsw's test
