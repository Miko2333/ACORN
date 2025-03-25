#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>


#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


// added these
#include <faiss/Index.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <iostream>
#include <sstream>      // for ostringstream
#include <fstream>  
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <assert.h>     /* assert */
#include <thread>
#include <set>
#include <math.h>  
#include <numeric> // for std::accumulate
#include <cmath>   // for std::mean and std::stdev
#include <nlohmann/json.hpp>
#include "utils.cpp"




// create indices for debugging, write indices to file, and get recall stats for all queries
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "====================\nSTART: running TEST_ACORN for hnsw, sift data --" << nthreads << "cores\n" << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES for hnsw --...\n");
    double t0 = elapsed();
    
    int efc = 40; // default is 40
    int efs = 16; //  default is 16
    int k = 10; // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten by the dimension of the dataset
    int M; // HSNW param M TODO change M back
    int M_beta; // param for compression
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int gamma;
    int n_centroids;
    // int filter = 0;
    std::string dataset; // must be sift1B or sift1M or tripclick
    int test_partitions = 0;
    int step = 10; //2
    
    std::string assignment_type = "rand";
    int alpha = 0;

    srand(0); // seed for random number generator
    int num_trials = 60;


    size_t N = 0; // N will be how many we truncate nb from sift1M to

    int opt;
    {// parse arguments

        if (argc < 6 || argc > 8) {
            fprintf(stderr, "Syntax: %s <number vecs> <gamma> [<assignment_type>] [<alpha>] <dataset> <M> <M_beta>\n", argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);
     

        gamma = atoi(argv[2]);
        printf("gamma: %d\n", gamma);


        
        
        dataset = argv[3];
        printf("dataset: %s\n", dataset.c_str());
        if (dataset != "sift1M" && dataset != "sift1M_test" && dataset != "sift1B" && dataset != "tripclick" && dataset != "paper" && dataset != "paper_rand2m") {
            printf("got dataset: %s\n", dataset.c_str());
            fprintf(stderr, "Invalid <dataset>; must be a value in [sift1M, sift1B]\n");
            exit(1);
        }

        M = atoi(argv[4]);
        printf("M: %d\n", M);

        M_beta = atoi(argv[5]);
        printf("M_beta: %d\n", M_beta);

    }

    std::vector<int> metadata;
    for(int i = 0; i < N; i++)
        metadata.push_back(1);
    printf("Start building ACORN\n");
    // ACORN-gamma
    faiss::IndexACORNFlat acorn_gamma(d, M, gamma, metadata, M_beta);

    // ACORN-1
    // faiss::IndexACORNFlat acorn_1(d, M, 1, M*2);
    printf("Reading base\n");
    size_t nb, d2;
    float* xb = fvecs_read("./testing_data/sift1M/sift_base.fvecs", &d2, &nb);
    nb = 1000000;
    d = d2;
    std::cout << nb << '\n';
    printf("Adding base\n");
    acorn_gamma.add(nb, xb);
    printf("Reading queries\n");
    size_t nq;
    float* xq = fvecs_read("./testing_data/sift1M/sift_query.fvecs", &d2, &nq);
    nq = 100;
    std::cout << nq << '\n';
    
    std::vector<faiss::idx_t> nns2(k * nq);
    std::vector<float> dis2(k * nq);
    std::vector<char> filter_ids_map(nq * N);
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < N; j++) {
            filter_ids_map[i * N + j] = true;
        }
    }
    acorn_gamma.search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data());

    const faiss::ACORNStats& stats = faiss::acorn_stats;

    std::cout << "============= ACORN QUERY PROFILING STATS =============" << std::endl;
    printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
            elapsed() - t0,
            k,
            nq);
    std::cout << "n1: " << stats.n1 << std::endl;
    std::cout << "n2: " << stats.n2 << std::endl;
    std::cout << "n3 (number distance comps at level 0): " << stats.n3 << std::endl;
    std::cout << "ndis: " << stats.ndis << std::endl;
    std::cout << "nreorder: " << stats.nreorder << std::endl;
    printf("average distance computations per query: %f\n", (float)stats.n3 / stats.n1);
    
    return 0;
}