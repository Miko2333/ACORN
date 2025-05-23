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
    std::string dataset; // must be sift1B or sift or tripclick
    int test_partitions = 0;
    int step = 10; //2
    
    std::string assignment_type = "rand";
    int alpha = 0;

    srand(0); // seed for random number generator
    int num_trials = 60;


    size_t N = 0; // N will be how many we truncate nb from sift to

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
        if (dataset != "sift" && dataset != "paper" && dataset != "gist") {
            printf("got dataset: %s\n", dataset.c_str());
            fprintf(stderr, "Invalid <dataset>; must be a value in [sift, sift1B]\n");
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
    if (dataset == "sift")
        xb = fvecs_read("./testing_data/sift/sift_base.fvecs", &db, &nb);
    else if (dataset == "paper")
        xb = fvecs_read("./testing_data/paper/paper_base.fvecs", &db, &nb);
    else if (dataset == "gist")
        xb = fvecs_read("./testing_data/gist/gist_base.fvecs", &db, &nb);
    d = db;
    // nb = 1000000;
    std::cout << nb << ' ' << db << '\n';

    printf("Reading base metadata\n");
    std::vector<int> metadata_base(N, 0);
    if (meta_flag) {
        std::ifstream meta_in;
        std::stringstream filename;
        if (dataset == "sift") 
            filename << "./testing_data/sift/metadata_base_" << meta_num << ".txt";
        else if (dataset == "paper")
            filename << "./testing_data/paper/metadata_base_" << meta_num << ".txt";
        else if (dataset == "gist")
            filename << "./testing_data/gist/metadata_base_" << meta_num << ".txt";
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
    if (dataset == "sift")
        xq = fvecs_read("./testing_data/sift/sift_query.fvecs", &dq, &nq);
    else if (dataset == "paper")
        xq = fvecs_read("./testing_data/paper/paper_query.fvecs", &dq, &nq);
    else if (dataset == "gist")
        xq = fvecs_read("./testing_data/gist/gist_query.fvecs", &dq, &nq);
    // nq = 1000;
    std::cout << nq << '\n';

    printf("Reading query metadata\n");
    std::vector<std::vector<int> > metadata_query(nq, std::vector<int> (2, 0));
    if (meta_flag) {
        std::ifstream meta_in;
        std::stringstream filename;
        if (dataset == "sift") 
            filename << "./testing_data/sift/metadata_query_" << meta_num << ".txt";
        else if (dataset == "paper")
            filename << "./testing_data/paper/metadata_query_" << meta_num << ".txt";
        else if (dataset == "gist")
            filename << "./testing_data/gist/metadata_query_" << meta_num << ".txt";
        meta_in.open(filename.str());
        if (!meta_in) {
            std::cout << "Failed to open query metadata file\n";
            return 0;
        }
        for(int i = 0; i < nq; i++)
            meta_in >> metadata_query[i][0] >> metadata_query[i][1];
        meta_in.close();
    }
    else {
        puts("Error");
        return 0;
    }
    
    printf("Reading groundtruth\n");
    size_t ng = 0, kg = 0;
    int* gt = NULL;
    if (meta_flag) {
        // experiment
        std::ifstream gt_file;
        std::stringstream filename;
        if (dataset == "sift")
            filename << "./testing_data/sift/groundtruth_" << meta_num << ".txt";
        else if (dataset == "paper")
            filename << "./testing_data/paper/groundtruth_" << meta_num << ".txt";
        else if (dataset == "gist")
            filename << "./testing_data/gist/groundtruth_" << meta_num << ".txt";
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
        if (dataset == "sift")
            gt = ivecs_read("./testing_data/sift/sift_groundtruth.ivecs", &kg, &ng);
        else if (dataset == "paper")
            gt = ivecs_read("./testing_data/paper/paper_groundtruth.ivecs", &kg, &ng);
        else if (dataset == "gist")
            gt = ivecs_read("./testing_data/gist/gist_groundtruth.ivecs", &kg, &ng);
    }
    std::cout << ng << ' ' << kg << '\n';

    printf("Start building HNSW\n");
    double t1 = elapsed();
    faiss::IndexHNSWFlat* hnsw = new faiss::IndexHNSWFlat(d, M);

    bool index_flag = false;
    std::stringstream index_file;
    index_file << "./indexes/HNSW_" << dataset << "_high.fvecs";
    if (meta_flag && fileExistsAndNotEmpty(index_file.str())) {
        index_flag = true;
    }
    if (index_flag) {
        printf("Loading HNSW index from file: %s\n", index_file.str().c_str());
        delete hnsw;
        faiss::Index* tmp = faiss::read_index(index_file.str().c_str());
        hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(tmp);
        if (!hnsw) {
            printf("Failed to load HNSW index from file: %s\n", index_file.str().c_str());
            return 0;
        }
    }
    else {
        printf("Creating HNSW index\n");
        hnsw->hnsw.efConstruction = efc;
        hnsw->add(nb, xb);
        faiss::write_index(hnsw, index_file.str().c_str());
        printf("Base added: [%.3f s]\n", elapsed() - t1);
    }

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
            if (metadata_query[i][0] <= metadata_base[j] && metadata_query[i][1] >= metadata_base[j])
                filter_ids_map[i * N + j] = true;
            else
                filter_ids_map[i * N + j] = false;
        }
    }
    printf("Predicate filtering done: [%.3f s]\n", elapsed() - t2);

    printf("Searching\n");
    double t3 = elapsed();
    // HNSW post-filtering
    hnsw->hnsw.efSearch = efs;
    hnsw->search(nq, xq, cand, dis.data(), nns.data());
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

    std::cout << "============= HNSW QUERY PROFILING STATS =============" << std::endl;
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