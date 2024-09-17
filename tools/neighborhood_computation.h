/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2024 Nikolai Maas <nikolai.maas@kit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#pragma once

#include <vector>

#include "mt-kahypar/macros.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/datastructures/static_graph.h"
#include "kahypar-resources/datastructure/fast_reset_flag_array.h"

using namespace mt_kahypar;
using FastResetArray = kahypar::ds::FastResetFlagArray<>;

using Graph = ds::StaticGraph;

struct NeighborhoodResult {
  std::array<HypernodeID, 2> roots;
  const std::vector<HypernodeID>& n1_list;
  const FastResetArray& n1_set;
  const std::vector<HypernodeID>& n2_list;
  const FastResetArray& n2_set;
  bool includes_two_hop;

  bool isInN1(HypernodeID node) {
    return node == roots[0] || node == roots[1] || n1_set[node];
  }

  bool isInN1Exactly(HypernodeID node) {
    return n1_set[node];
  }

  bool isInN2(HypernodeID node) {
    return isInN1(node) || n2_set[node];
  }

  bool isInN2Exactly(HypernodeID node) {
    return n2_set[node];
  }
};


class NeighborhoodComputation {
 public:
  NeighborhoodComputation() = default;

  template<size_t N>
  NeighborhoodResult computeNeighborhood(const Graph& graph, std::array<HypernodeID, N> roots, bool include_two_hop) {
    static_assert(N > 0 && N <= 2);
    ALWAYS_ASSERT(n1_list.empty());
    NeighborhoodResult result {{roots[0], roots[0]}, n1_list, n1_set, n2_list, n2_set, include_two_hop};
    if constexpr (N == 2) {
      result.roots = roots;
    }

    for (HypernodeID root: roots) {
      for (HyperedgeID edge: graph.incidentEdges(root)) {
        HypernodeID neighbor = graph.edgeTarget(edge);
        if (!result.isInN1(neighbor)) {
          n1_list.push_back(neighbor);
          n1_set.set(neighbor);
        }
      }
    }
    if (include_two_hop) {
      for (HypernodeID node: n1_list) {
        for (HyperedgeID edge: graph.incidentEdges(node)) {
          HypernodeID neighbor = graph.edgeTarget(edge);
          if (!result.isInN2(neighbor)) {
            n2_list.push_back(neighbor);
            n2_set.set(neighbor);
          }
        }
      }
    }
    return result;
  }

  void reset() {
    n1_list.clear();
    n1_set.reset();
    n2_list.clear();
    n2_set.reset();
  }

 private:
  std::vector<HypernodeID> n1_list;
  FastResetArray n1_set;
  std::vector<HypernodeID> n2_list;
  FastResetArray n2_set;
};

