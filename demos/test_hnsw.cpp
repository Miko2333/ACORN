#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>


#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
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
    std::cout << "====================\nSTART: running TEST_HNSW for hnsw, sift data --" << nthreads << "cores\n" << std::endl;
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

    bool meta_flag = false;

    int meta_num = 0;

    {// parse arguments
        if (argc < 6 || argc > 8) {
            fprintf(stderr, "Syntax: %s <number vecs> <gamma> <dataset> <M> <M_beta> [<with_metadata>]\n", argv[0]);
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

        if (argc >= 7) {
            meta_flag = true;
            meta_num = atoi(argv[6]);
            printf("meta_num: %d\n", meta_num);
        }

        if (argc == 8) {
            efs = atoi(argv[7]);
            printf("efs: %d\n", efs);
        }
    }

    std::stringstream output_file;
    output_file << "./results/HNSW_res_" << dataset << "_" << meta_num << ".txt";
    freopen(output_file.str().c_str(), "w", stdout);
    printf("Reading base vectors\n");
    size_t nb = 0, db = 0;
    float* xb = NULL;
    if (dataset == "sift1M")
        xb = fvecs_read("./testing_data/sift1M/sift_base.fvecs", &db, &nb);
    else if (dataset == "paper")
        xb = fvecs_read("./testing_data/paper/paper_base.fvecs", &db, &nb);
    d = db;
    // nb = 1000000;
    std::cout << nb << ' ' << db << '\n';

    printf("Reading base metadata\n");
    std::vector<int> metadata_base(N, 0);
    if (meta_flag) {
        std::ifstream meta_in;
        std::stringstream filename;
        if (dataset == "sift1M") 
            filename << "./testing_data/sift1M/metadata_base_" << meta_num << ".txt";
        else if (dataset == "paper")
            filename << "./testing_data/paper/metadata_base_" << meta_num << ".txt";
        meta_in.open(filename.str());
        if (!meta_in) {
            std::cout << "Failed to open base metadata file\n";
            return 0;
        }
        for(int i = 0; i < N; i++)
            meta_in >> metadata_base[i];
        meta_in.close();
    }
    else {
        for(int i = 0; i < N; i++)
            metadata_base[i] = 1;
    }

    printf("Reading query vectors\n");
    size_t nq = 0, dq = 0;
    float* xq = NULL;
    if (dataset == "sift1M")
        xq = fvecs_read("./testing_data/sift1M/sift_query.fvecs", &dq, &nq);
    else if (dataset == "paper")
        xq = fvecs_read("./testing_data/paper/paper_query.fvecs", &dq, &nq);
    // nq = 1000;
    std::cout << nq << '\n';

    printf("Reading query metadata\n");
    std::vector<int> metadata_query(nq, 0);
    if (meta_flag) {
        std::ifstream meta_in;
        std::stringstream filename;
        if (dataset == "sift1M") 
            filename << "./testing_data/sift1M/metadata_query_" << meta_num << ".txt";
        else if (dataset == "paper")
            filename << "./testing_data/paper/metadata_query_" << meta_num << ".txt";
        meta_in.open(filename.str());
        if (!meta_in) {
            std::cout << "Failed to open query metadata file\n";
            return 0;
        }
        for(int i = 0; i < nq; i++)
            meta_in >> metadata_query[i];
        meta_in.close();
    }
    else {
        for(int i = 0; i < nq; i++)
            metadata_query[i] = 1;
    }
    
    printf("Reading groundtruth\n");
    size_t ng = 0, kg = 0;
    int* gt = NULL;
    if (meta_flag) {
        // experiment
        std::ifstream gt_file;
        std::stringstream filename;
        if (dataset == "sift1M")
            filename << "./testing_data/sift1M/groundtruth_" << meta_num << ".txt";
        else if (dataset == "paper")
            filename << "./testing_data/paper/groundtruth_" << meta_num << ".txt";
        gt_file.open(filename.str());
        if (!gt_file) {
            printf("Failed to open groundtruth file\n");
            return 0;
        }
        gt_file >> ng >> kg;
        gt = new int[ng * kg];
        for (int i = 0; i < ng * kg; i++) {
            gt_file >> gt[i];
        }
        gt_file.close();
    }
    else {
        if (dataset == "sift1M")
            gt = ivecs_read("./testing_data/sift1M/sift_groundtruth.ivecs", &kg, &ng);
        else if (dataset == "paper")
            gt = ivecs_read("./testing_data/paper/paper_groundtruth.ivecs", &kg, &ng);
    }
    std::cout << ng << ' ' << kg << '\n';

    printf("Start building HNSW\n");
    double t1 = elapsed();
    // ACORN-gamma
    faiss::IndexHNSWFlat hnsw(d, M);
    // ACORN-1
    // faiss::IndexACORNFlat acorn_1(d, M, 1, metadata, M);
    hnsw.hnsw.efSearch = efs;
    hnsw.add(nb, xb);
    printf("Base added: [%.3f s]\n", elapsed() - t1);

    int cand = k * gamma;
    std::vector<faiss::idx_t> nns(cand * nq);
    std::vector<faiss::idx_t> nns2(k * nq);
    std::vector<float> dis(cand * nq);
    std::vector<float> dis2(k * nq);
    std::vector<char> filter_ids_map(nq * N);
    printf("Predicate filtering\n");
    double t2 = elapsed();
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < N; j++) {
            // filter_ids_map[i * N + j] = true;
            if (metadata_query[i] == metadata_base[j])
                filter_ids_map[i * N + j] = true;
            else
                filter_ids_map[i * N + j] = false;
        }
    }
    printf("Predicate filtering done: [%.3f s]\n", elapsed() - t2);

    printf("Searching\n");
    double t3 = elapsed();
    // HNSW post-filtering
    
    hnsw.search(nq, xq, cand, dis.data(), nns.data());
    for (int i = 0; i < nq; i++) {
        int tot = 0;
        for (int j = 0; j < cand; j++) {
            if (filter_ids_map[i * N + nns[j + i * cand]]) {
                nns2[tot + i * k] = nns[j + i * cand];
                dis2[tot + i * k] = dis[j + i * cand];
                tot++;
            }
            if (tot == k)
                break;
        }
        for (int j = tot; j < k; j++) {
            nns2[j + i * k] = -1;
            dis2[j + i * k] = -1;
        }
    }
    double query_time = elapsed() - t3;
    printf("Search done: [%.3f s]\n", query_time);

    double ans = 0;
    for (int i = 0; i < nq; i++) {
        double res = 0;
        for (int j = 0; j < k; j++) {
            // if (i < 20)
            //     printf("%7ld %7d\n", nns2[j + i * k], gt[j + i * kg]);
            for (int l = 0; l < k; l++) {
                if (nns2[j + i * k] == gt[l + i * kg]) {
                    res = res + 1;
                    break;
                }
            }
        }
        ans += res / k;
    }
    ans /= nq;
    printf("Recall@%d = %.3f\n", k, ans);
    printf("QPS = %.3f\n", nq / query_time);
    const faiss::HNSWStats& stats = faiss::hnsw_stats;

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