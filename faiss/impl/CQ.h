/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <queue>
#include <unordered_set>
#include <vector>
#include <unordered_map>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <faiss/impl/ACORN.h>

// #include <nlohmann/json.hpp>
// // #include <format>
// // for convenience
// using json = nlohmann::json;

namespace faiss {

struct CQ : ACORN {
    explicit CQ(int M, int gamma, std::vector<int>& metadata, int M_beta);
    ACORNStats search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        const SearchParametersACORN* params = nullptr,
        idx_t query_id = -1,
        size_t query_len = 0,
        const std::vector <float>* tmp_q = nullptr,
        std::vector <std::unordered_map <faiss::idx_t, float> >* tmp_d = nullptr
    ) const;
    ACORNStats hybrid_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        const SearchParametersACORN* params = nullptr,
        idx_t query_id = -1,
        size_t query_len = 0,
        const std::vector <float>* tmp_q = nullptr,
        std::vector <std::unordered_map <faiss::idx_t, float> >* tmp_d = nullptr
    ) const;
};

} // namespace faiss
