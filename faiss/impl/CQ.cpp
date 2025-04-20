/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/ACORN.h>
#include <faiss/impl/CQ.h>


#include <string>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>  
#include <unordered_map>
#include <iostream>
#include <fstream>

/*******************************************************
 * Added for debugging
 *******************************************************/

namespace {
    const int debugFlag = 0;

    // const int debugSearchFlag =  std::atoi(std::getenv(debugSearchFlag));
    const char* debugSearchFlagEnv = std::getenv("debugSearchFlag");
    int debugSearchFlag = debugSearchFlagEnv ? std::atoi(debugSearchFlagEnv) : 0;
    
    void debugTime() {
        if (debugFlag || debugSearchFlag) {
            struct timeval tval;
            gettimeofday(&tval, NULL);
            struct tm *tm_info = localtime(&tval.tv_sec);
            char timeBuff[25] = "";
            strftime(timeBuff, 25, "%H:%M:%S", tm_info);
            char timeBuffWithMilli[50] = "";
            sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
            std::string timestamp(timeBuffWithMilli);
            std::cout << timestamp << std::flush;
        }
    }

    double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }
}

#ifndef debug
#define debug(fmt, ...) \
    do { \
        if (debugFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__);\
        } \
        if (debugFlag == 2) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)
#endif

// same as debug but for search debugging only
#ifndef debug_search
#define debug_search(fmt, ...) \
    do { \
        if (debugSearchFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__);\
        } \
        if (debugSearchFlag == 2) { \
            fprintf(stdout, "%d:%s(): " fmt, __LINE__, __func__, __VA_ARGS__); \
        } \
        if (debugSearchFlag == 3) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)
#endif

namespace faiss {

CQ::CQ(int M, int gamma, std::vector<int>& metadata, int M_beta) :
    ACORN(M, gamma, metadata, M_beta) {}

namespace {

    using storage_idx_t = ACORN::storage_idx_t;
    using NodeDistCloser = ACORN::NodeDistCloser;
    using NodeDistFarther = ACORN::NodeDistFarther;
    
    /**************************************************************
     * Addition subroutines
     **************************************************************/
    
    /// remove neighbors from the list to make it smaller than max_size
    void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistCloser>& resultSet1,
            int max_size, int gamma, storage_idx_t q_id, int q_attr, ACORN& hnsw) {
        debug("shrink_neighbor_list from size %ld, to max size %d\n", resultSet1.size(), max_size);
        // if (resultSet1.size() < max_size) {
        //     return;
        // }
        std::priority_queue<NodeDistFarther> resultSet;
        std::vector<NodeDistFarther> returnlist;
    
        while (resultSet1.size() > 0) {
            resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
            resultSet1.pop();
        }
    
        // ACORN::shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma, q_id, q_attr);
        hnsw.shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma, q_id, q_attr);
    
    
        for (NodeDistFarther curen2 : returnlist) {
            resultSet1.emplace(curen2.d, curen2.id);
        }
    }
    
    // modified from normal hnsw
    /// add a link between two elements, possibly shrinking the list
    /// of links to make room for it.
    void add_link(
            ACORN& hnsw,
            DistanceComputer& qdis,
            storage_idx_t src,
            storage_idx_t dest,
            int level) {
        size_t begin, end;
        hnsw.neighbor_range(src, level, &begin, &end);
        if (hnsw.neighbors[end - 1] == -1) { // mood
            // there is enough room, find a slot to add it
            size_t i = end;
            while (i > begin) {
                if (hnsw.neighbors[i - 1] != -1) // mod
                    break;
                i--;
            }
            // hnsw.neighbors[i] = dest;
            hnsw.neighbors[i] = dest; // mod
            debug("added link from %d to %d at level %d\n", src, dest, level);
            return;
        }
    
        // otherwise we let them fight out which to keep
    
        // copy to resultSet...
        std::priority_queue<NodeDistCloser> resultSet;
        resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
        for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
            // storage_idx_t neigh = hnsw.neighbors[i];
            // auto [neigh, metadata] = hnsw.neighbors[i]; // mod
            auto neigh = hnsw.neighbors[i];
            auto metadata = hnsw.metadata[neigh];
            resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
        }
    
    
        debug("calling shrink neigbor list, src: %d, dest: %d, level: %d\n", src, dest, level);
        
        if (level == 0) {
            shrink_neighbor_list(qdis, resultSet, end - begin, hnsw.gamma, src, hnsw.metadata[src], hnsw);
    
        }
        
        
    
        // ...and back
        size_t i = begin;
        while (resultSet.size()) {
            // hnsw.neighbors[i++] = resultSet.top().id;
            hnsw.neighbors[i++] = ACORN::NeighNode(resultSet.top().id); // mod
            resultSet.pop();
        }
        // they may have shrunk more than just by 1 element
        while (i < end) {
            // hnsw.neighbors[i++] = -1;
            hnsw.neighbors[i++] = ACORN::NeighNode(-1); //mod
        }
    }
    
    
    // modified from normal hnsw
    /// search neighbors on a single level, starting from an entry point
    // this only gets called in construction
    void search_neighbors_to_add(
            ACORN& hnsw,
            DistanceComputer& qdis,
            std::priority_queue<NodeDistCloser>& results,
            int entry_point,
            float d_entry_point,
            int level,
            VisitedTable& vt,
            std::vector<storage_idx_t> ep_per_level = {}) {
        debug("search_neighbors to add, entrypoint: %d\n", entry_point);
        // top is nearest candidate
        std::priority_queue<NodeDistFarther> candidates;
    
        NodeDistFarther ev(d_entry_point, entry_point);
        candidates.push(ev);
        results.emplace(d_entry_point, entry_point);
        vt.set(entry_point);
    
        // number of neighbors we want for hybrid construction
        // int M = hnsw.nb_neighbors(level);
        int M;
        if (level == 0) { // level 0 may be compressed
            M = 2* hnsw.M * hnsw.gamma;
        }
        else {
            M = hnsw.nb_neighbors(level);
        }
        debug("desired resuts size: %d, at level: %d\n", M, level);
        
    
        int backtrack_level = level;
    
        while (!candidates.empty()) {
            // get nearest
            const NodeDistFarther& currEv = candidates.top();
    
            // MOD greedy break - added break conditional on size of results
            if ((currEv.d > results.top().d && hnsw.gamma == 1) || results.size() >= M) {
                debug("greedy stop in construction, results size: %ld, desired size: %d, gamma = %d\n", results.size(), M, hnsw.gamma);
                break;
            }
            int currNode = currEv.id;
            candidates.pop();
    
            // loop over neighbors
            size_t begin, end;
            hnsw.neighbor_range(currNode, level, &begin, &end);
            
            int numIters = 0;
    
            debug("checking neighbors of %d\n", currNode);
            for (size_t i = begin; i < end; i++) {
                // auto [nodeId, metadata] = hnsw.neighbors[i]; // storage_idx_t, int
                auto nodeId = hnsw.neighbors[i];
                auto metadata = hnsw.metadata[nodeId];
                // storage_idx_t nodeId = hnsw.neighbors[i];
                if (nodeId < 0)
                    break;
                if (vt.get(nodeId))
                    continue;
                vt.set(nodeId);
    
                // limit number neighbors visisted during construciton
                numIters = numIters + 1;
                if (numIters > hnsw.M) {
                    break;
                }
    
                float dis = qdis(nodeId);
                NodeDistFarther evE1(dis, nodeId);
    
                // debug("while checking neighbors of %d, efc: %d, results size: %ld, just visited %d\n", currNode, hnsw.efConstruction, results.size(), nodeId);
                if (results.size() < hnsw.efConstruction || results.top().d > dis) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() > hnsw.efConstruction) {
                        results.pop();
                    }
                }
                debug("while checking neighbors of %d, just visited %d -- efc: %d, results size: %ld, candidates size: %ld, \n", currNode, nodeId, hnsw.efConstruction, results.size(), candidates.size());
    
                
    
                // limit number neighbors visisted during construciton
                numIters = numIters + 1;
                if (numIters > hnsw.M) {
                    break;
                }
    
            
            }
            
            debug("during BFS, gamma: %d, candidates size: %ld, results size: %ld, vt.num_visited: %d, nb on level: %d, backtrack_level: %d, level: %d\n", hnsw.gamma, candidates.size(), results.size(), vt.num_visited(), hnsw.nb_per_level[level], backtrack_level, level);
        }
        debug("search_neighbors to add finds %ld nn's\n", results.size());
        // printf("search_neighbors to add finds %ld nn's\n", results.size());
        // printf("\tdesired resuts size: %d, at level: %d\n", M, level);
    
        vt.advance();
    }
    
    
    /**************************************************************
     * Searching subroutines
     **************************************************************/
    
    /// greedily update a nearest vector at a given level
    /// used for construction (other version below will be used in search)
    void greedy_update_nearest(
            const ACORN& hnsw,
            DistanceComputer& qdis,
            int level,
            storage_idx_t& nearest,
            float& d_nearest) {
        debug("%s\n", "reached");
        for (;;) {
            storage_idx_t prev_nearest = nearest;
    
            size_t begin, end;
            hnsw.neighbor_range(nearest, level, &begin, &end);
            
            
            int numIters = 0;
    
            for (size_t i = begin; i < end; i++) {
                auto v = hnsw.neighbors[i];
                auto metadata = hnsw.metadata[v];
                if (v < 0) {
                    break;
                }
    
                // limit number neighbors visisted during construciton
                numIters = numIters + 1;
                if (numIters > hnsw.M) {
                    break;
                }
    
                float dis = qdis(v);
                if (dis < d_nearest) {
                    nearest = v;
                    d_nearest = dis;
                }
            }
            if (nearest == prev_nearest) {
                return;
            }
        }
    }
    
} // namespace

namespace Opt {
// check if |A-sqrt(B)| >= sqrt(C)
// A is real distance, B and C are squared distances
// this is used in hybrid search for CQ optimize 
bool check_triangle(float A, float B, float C) {
    if (fabs(A) < 1e-6) {
        return B >= C;
    }
    float D = A * A + B - C;
    float E = A + A;
    return B <= (D*D) / (E*E);
}

/// moved from anonymous to Opt namespace
int hybrid_greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int level,
        storage_idx_t& nearest,
        float& d_nearest,
        idx_t query_id = -1,
        size_t query_len = 0,
        const std::vector <float>* tmp_q = nullptr,
        std::vector <std::unordered_map <faiss::idx_t, float> >* tmp_d = nullptr
    ) {
    debug("%s\n", "reached"); 
    // printf("hybrid_greedy_update_nearest called with parameters: filter: %d, op: %d, regex: %s, level: %d\n", filter, op, regex.c_str(), level);
    // printf("hybrid_greedy_update_nearest called with parameters: query_id: %d, query_len: %d, level: %d\n", query_id, query_len, level);
    if (query_id != -1) {
        FAISS_THROW_IF_NOT(query_len > 0 && tmp_q != nullptr && tmp_d != nullptr);
        FAISS_THROW_IF_NOT(tmp_q->size() == query_len * query_len && tmp_d->size() == query_len);
    }

    auto check_dis = [query_id, query_len, tmp_q, tmp_d, d_nearest] (int v) {
        for (size_t j = 0; j < query_len; j++) {
            auto it = (*tmp_d)[j].find(v);
            if (it != (*tmp_d)[j].end()) {
                if (check_triangle((*tmp_q)[query_id * query_len + j], it->second, d_nearest)) {
                    return true;
                }
            }
        }
        return false;
    };

    int ndis = 0;
    for (;;) {
        int num_found = 0;
        storage_idx_t prev_nearest = nearest;
        debug_search("----hybrid_greedy_update visists current nearest: %d, d_nearest: %f\n", nearest, d_nearest);

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        debug_search("%s", "--------checking neighbors: \n");
        
        // for debugging, collect all neighbors looked at in a vector
        std::vector<std::pair<storage_idx_t, int>> neighbors_checked;
        bool keep_expanding = true;

        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            
            if (v < 0)
                break;
                
            // note that this slows down search significantly but can be useful for debugging
            // if (debugSearchFlag) {
            //     neighbors_checked.push_back(std::make_pair(v, metadata)); 
            //     debug_search("------------checking neighbor: %d, metadata: %d, metadata & filter: %d\n", v, metadata, metadata & filter);
            // }

            // filter
            // printf("---at first filter: op: %d, metadata: %s, regex: %s, check_regex result: %d\n", op, hnsw.metadata_strings[v].c_str(), regex.c_str(), CHECK_REGEX(hnsw.metadata_strings[v], regex));
            if (filter_map[v]) {
                num_found = num_found + 1;
            } else {
                // not filter & gamma > 1
                if (hnsw.gamma > 1) {
                    continue;
                }
            }
            

        
            
            // check if filter pass
            if (filter_map[v]) {
                bool flag = false;

                // Clustered Query opt
                if (query_id != -1 && filter_map[nearest]) {
                    // use triangle inequality to prune
                    flag = check_dis(v);
                }

                if (!flag) {
                    float dis = qdis(v);
                    ndis += 1;

                    if (query_id != -1) {
                        (*tmp_d)[query_id][v] = dis; // save this computed distance
                    }

                    if (dis < d_nearest || !filter_map[nearest]) {
                    
                        nearest = v;
                        d_nearest = dis;
                        // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                    }
                }
                if (num_found >= hnsw.M) {
                    // debug_search("----found %d neighbors with filter %d, returning\n", num_found, filter);
                    break;
                }
            }            

          

            // expand neighbor list if gamma=1
            if (hnsw.gamma == 1) {
                size_t begin2, end2;
                hnsw.neighbor_range(v, level, &begin2, &end2);
                for (size_t j = begin2; j < end2; j++) {
                    auto v2 = hnsw.neighbors[j];
                   

                    if (v2 < 0)
                        break;


                    // check filter pass
                    if (filter_map[v2]) {
                        num_found = num_found + 1;

                        bool flag = false;

                        if (query_id != -1 && filter_map[nearest]) {
                            flag = check_dis(v2);
                        }

                        if (!flag) {
                            float dis2 = qdis(v2);
                            ndis += 1;
                            // debug_search("------------found: %d, metadata: %d distance to v: %f\n", v2, metadata2, dis2);
            
                            if (query_id != -1) {
                                (*tmp_d)[query_id][v2] = dis2; // save this computed distance
                            }

                            if (dis2 < d_nearest || !filter_map[nearest]) {
                                nearest = v2;
                                d_nearest = dis2;
                                // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                            }
                        }

                        if (num_found >= hnsw.M) {
                            break;
                        }
                    } 
                   
                }
            }
        }       

        if (nearest == prev_nearest) {
            return ndis;
        }
    }
    return ndis;
}

} // namespace

/**************************************************************
 * Searching
 **************************************************************/

namespace {

using MinimaxHeap = ACORN::MinimaxHeap;
using Node = ACORN::Node;
using NeighNode = ACORN::NeighNode;
/** Do a BFS on the candidates list */
// this is called in search and search_from_level_0
int search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr) {
    debug("%s\n", "reached");
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) { // candidates is heap of size max(efs, k)
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->is_member(v1)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, d, v1);
                } else if (d < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, d, v1);
                }
            }
            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }

    return nres;
}

// has a filter arg for hybrid search, this only gets called on level 0
int hybrid_search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr) {
    // debug("%s\n", "reached");
    // printf("----hybrid_search_from_candidates called with filter: %d, k: %d, op: %d, regex: %s\n", filter, k, op, regex.c_str());
    // debug_search("----hybrid_search_from_candidates called with filter: %d, k: %d\n", filter, k);
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;


    // timing variables
    double t1_candidates_loop = elapsed();
    
    while (candidates.size() > 0) { // candidates is heap of size max(efs, k)
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);
        // debug_search("--------visiting v0: %d, d0: %f, candidates_size: %d\n", v0, d0, candidates.size());

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                // debug("--------%s\n", "n_dis_below >= efSearch BREAK cond reached");
                // debug_search("--------n_dis_below: %d, efSearch: %d - triggers break\n", n_dis_below, efSearch);
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // variable to keep track of search expansion
        int num_found = 0;
        int num_new = 0;
        bool keep_expanding = true;

        // for debugging, collect all neighbors looked at in a vector
        std::vector<std::pair<storage_idx_t, int>> neighbors_checked;

        double t1_neighbors_loop = elapsed();
        for (size_t j = begin; j < end; j++) {
            // auto [v1, metadata] = hnsw.neighbors[j];
            bool promising = 0;
            bool outerskip = false;

            auto v1 = hnsw.neighbors[j];
            // auto metadata = hnsw.metadata[v1];
            // debug_search("------------visiting neighbor (%ld) - %d, metadata: %d\n", j-begin, v1, metadata);


            if (v1 < 0) {
                break;
            }

            // note that this slows down search performance significantly
            // if (debugSearchFlag) {
            //     neighbors_checked.push_back(std::make_pair(v1, metadata)); // for debugging
            // }
            
            if (vt.get(v1)) {
                continue;
            }

            if (filter_map[v1]) {
                num_found = num_found + 1; // increment num found
            }

            // filter
            if (filter_map[v1]) {
                vt.set(v1);
                num_new = num_new + 1; // increment num new
                ndis++;
                float d = qdis(v1);
                // debug_search("------------new candidate %d, distance: %f\n", v1, d);

                if (!sel || sel->is_member(v1)) {
                    if (nres < k) {
                        // debug_search("-----------------pushing new candidate, nres: %d (to be incrd)\n", nres);
                        faiss::maxheap_push(++nres, D, I, d, v1);
                        // debug_search("-----------------pushed new candidate, nres: %d\n", nres);
                        promising = 1;
                        candidates.push(v1, d);
                    } else if (d < D[0]) {
                        // debug_search("-----------------replacing top, nres: %d\n", nres);
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                        promising =1;
                        candidates.push(v1, d);
                    }
                }

                if (num_found >= hnsw.M * 2) {
                    // debug_search("------------num_found: %d, M: %d - triggered outer brea, skpping to M_beta=%d neighbork\n", num_found, hnsw.M * 2, hnsw.M_beta);
                    keep_expanding = false;
                    break;
                }
            }    
            
            if (((j - begin >= hnsw.M_beta) && keep_expanding) || hnsw.gamma == 1) {
                debug_search("------------expanding neighbor list for %d; neighbor %ld, hnsw.M_beta: %d\n", v1, j-begin, hnsw.M_beta);
                size_t begin2, end2;
                hnsw.neighbor_range(v1, level, &begin2, &end2);
                // try to parallelize neighbor expansion
                for (size_t j2 = begin2; j2 < end2; j2+=1) {
                    
                    auto v2 = hnsw.neighbors[j2];

                    // note that this slows down search performance significantly when flag is on
                    // if (debugSearchFlag) {
                    //     neighbors_checked.push_back(std::make_pair(v2, metadata2)); // for debugging
                    // }
                    if (v2 < 0) {
                        // continue;
                        break;
                    }

                    if (vt.get(v2)) {
                        continue;
                    }

                    // if (metadata2 == filter) {
                    if (filter_map[v2]) {
                        num_found = num_found + 1; // increment num found
                    } else {
                        continue;
                    }
                    
                    vt.set(v2);
                    ndis++;
  
                    float d2 = qdis(v2);
                    // debug_search("------------new candidate from expansion %d, distance: %f\n", v2, d2);
                    if (!sel || sel->is_member(v2)) {
                        if (nres < k) {
                            // debug_search("-----------------pushing new candidate, nres: %d (to be incrd)\n", nres);
                            faiss::maxheap_push(++nres, D, I, d2, v2);
                            // debug_search("-----------------pushed new candidate, nres: %d\n", nres);
                            candidates.push(v2, d2);

                        } else if (d2 < D[0]) {
                            // debug_search("-----------------replacing top, nres: %d\n", nres);
                            faiss::maxheap_replace_top(nres, D, I, d2, v2);
                            candidates.push(v2, d2);
                        }
                    }
                    
                    if (num_found >= hnsw.M * 2) {
    
                        // debug_search("------------num_found: %d, 2M: %d - triggers break\n", num_found, hnsw.M * 2);
                        keep_expanding = false;
                        break;
                    }
                }


    
            }
        

        }

     
        

        nstep++; 
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }


    return nres;
}








} // anonymous namespace

ACORNStats CQ::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        const SearchParametersACORN* params,
        idx_t query_id,
        size_t query_len,
        const std::vector <float>* tmp_q,
        std::vector <std::unordered_map <faiss::idx_t, float> >* tmp_d
    ) const {
    debug("%s\n", "reached");
    ACORNStats stats;
    if (entry_point == -1) {
        return stats;
    }
    if (upper_beam == 1) { // common branch
        debug("%s\n", "reached upper beam == 1");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }

        
        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates(
                    *this, qdis, k, I, D, candidates, vt, stats, 0, 0, params);
        } else {
            debug("%s\n", "reached search_bounded_queue == False");
            throw FaissException("UNIMPLEMENTED search unbounded queue");
            
        }

        vt.advance();

    } else {
        debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates(
                        *this, qdis, k, I, D, candidates, vt, stats, 0);
            } else {
                nres = search_from_candidates(
                        *this,
                        qdis,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}


// hybrid search TODO
ACORNStats CQ::hybrid_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        const SearchParametersACORN* params,
        idx_t query_id,
        size_t query_len,
        const std::vector <float>* tmp_q,
        std::vector <std::unordered_map <faiss::idx_t, float> >* tmp_d
    ) const {
    debug("%s\n", "reached");
    // debug_search("Hybrid Search, params -- k: %d, filter: %d\n", k, filter);
    ACORNStats stats;
    if (entry_point == -1) {
        return stats;
    }


    if (upper_beam == 1) { // common branch
        debug("%s\n", "reached upper beam == 1");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        debug_search("-starting at ep: %d, d: %f, metadata: %d\n", nearest, d_nearest, metadata[nearest]);

        int ndis_upper = 0;
        for (int level = max_level; level >= 1; level--) {
            debug_search("-at level %d, searching for greedy nearest from current nearest: %d, dist: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);
            ndis_upper += Opt::hybrid_greedy_update_nearest(*this, qdis, filter_map, level, nearest, d_nearest,
                query_id, query_len, tmp_q, tmp_d);
            // ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter, op, regex, level, nearest, d_nearest);
            debug_search("-at level %d, new nearest: %d, d: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);
            

        }
        stats.n3 += ndis_upper;

        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);
            debug_search("-starting BFS at level 0 with ef: %d, nearest: %d, d: %f, metadata: %d\n", ef, nearest, d_nearest, metadata[nearest]);
            hybrid_search_from_candidates(
                    *this, qdis, filter_map, k, I, D, candidates, vt, stats, 0, 0, params);
            

        } else {
            // TODO
            printf("UNIMPLEMENTED BRANCH for hybid search\n");
            debug("%s\n", "reached search_bounded_queue == False");
            throw FaissException("UNIMPLEMENTED search unbounded queue");

            
        }

        vt.advance();

    } else {
        debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = hybrid_search_from_candidates(
                        *this, qdis, filter_map, k, I, D, candidates, vt, stats, 0);
            
                
            } else {
                nres = hybrid_search_from_candidates(
                        *this,
                        qdis,
                        filter_map,
                        // filter,
                        // op,
                        // regex,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}

} // namespace faiss
