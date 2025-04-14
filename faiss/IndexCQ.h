/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexACORN.h>
#include <faiss/impl/ACORN.h>
#include <faiss/impl/CQ.h>
#include <faiss/utils/utils.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

namespace faiss {

struct Node {
	size_t x, y, id;
	bool operator < (const Node &A) const {
		if(x != A.x)
			return x < A.x;
		if(y != A.y)
			return y < A.y;
		return id < A.id;
	}
};

struct IndexCQ : IndexACORN {
	CQ cq;

	explicit IndexCQ(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2); // defaults d = 0, M=32, gamma=1

    explicit IndexCQ(Index* storage, int M, int gamma, std::vector<int>& metadata, int M_beta);

	~IndexCQ() = default;

	// add to CQ, override ACORN
	void add(idx_t, const float* x);

	// hybrid search in CQ, override ACORN
	void search(
		idx_t n,
		const float* x,
		idx_t k,
		float* distances,
		idx_t* labels,
		char* filter_id_map,
		const SearchParameters* params = nullptr) const;

};

struct IndexCQFlat : IndexCQ {
	IndexCQFlat();
	IndexCQFlat(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2);
};


} // namespace faiss
