/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include "multilevel_coarsener_base.h"
#include "i_coarsener.h"

#pragma push_macro("DBGC")
#pragma push_macro("DBG")
#pragma push_macro("LOG")
#pragma push_macro("LLOG")
#pragma push_macro("V")
#pragma push_macro("RED")
#pragma push_macro("GREEN")
#pragma push_macro("CYAN")
#undef DBGC
#undef DBG
#undef LOG
#undef LLOG
#undef V
#undef RED
#undef GREEN
#undef CYAN

#include <kaminpar.h>
#include <kaminpar-common/datastructures/static_array.h>
#include <kaminpar-shm/coarsening/clustering/lp_clusterer.h>
#include <kaminpar-shm/coarsening/max_cluster_weights.h>

#pragma pop_macro("DBGC")
#pragma pop_macro("DBG")
#pragma pop_macro("LOG")
#pragma pop_macro("LLOG")
#pragma pop_macro("V")
#pragma pop_macro("RED")
#pragma pop_macro("GREEN")
#pragma pop_macro("CYAN")

#include "include/mtkahypartypes.h"

#include "mt-kahypar/utils/utilities.h"
#include "mt-kahypar/utils/progress_bar.h"
#include "mt-kahypar/utils/cast.h"

namespace mt_kahypar {

template<typename TypeTraits>
class ExperimentalCoarsener :  public ICoarsener,
                                          private MultilevelCoarsenerBase<TypeTraits> {
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

public:
  ExperimentalCoarsener(mt_kahypar_hypergraph_t hypergraph,
                        const Context& context,
                        uncoarsening_data_t* uncoarseningData) :
    Base(utils::cast<Hypergraph>(hypergraph),
         context,
         uncoarsening::to_reference<TypeTraits>(uncoarseningData)),
    _initial_num_nodes(utils::cast<Hypergraph>(hypergraph).initialNumNodes()),
    _pass_nr(0),
    _progress_bar(utils::cast<Hypergraph>(hypergraph).initialNumNodes(), 0, false),
    _enable_randomization(true),
    _current_vertices(utils::cast<Hypergraph>(hypergraph).initialNumNodes())
  {
  }

  ~ExperimentalCoarsener() { }

  void DisableRandomization() {
    _enable_randomization = false;
  }

private:
  void initializeImpl() override {
    if ( _context.partition.verbose_output && _context.partition.enable_progress_bar ) {
      _progress_bar.enable();
    }
  }

  bool coarseningPassImpl() override;

  bool shouldNotTerminateImpl() const override {
    return Base::currentNumNodes() > _context.coarsening.contraction_limit;
  }

  void terminateImpl() override {
    _progress_bar += (_initial_num_nodes - _progress_bar.count());   // fill to 100%
    _progress_bar.disable();
    _uncoarseningData.finalizeCoarsening();
  }

  HypernodeID currentLevelContractionLimit() {
    const auto& hg = Base::currentHypergraph();
    return std::max( _context.coarsening.contraction_limit,
               static_cast<HypernodeID>(
                    (hg.initialNumNodes() - hg.numRemovedHypernodes()) / _context.coarsening.maximum_shrink_factor) );
  }

  HypernodeID currentNumberOfNodesImpl() const override {
    return Base::currentNumNodes();
  }

  mt_kahypar_hypergraph_t coarsestHypergraphImpl() override {
    return mt_kahypar_hypergraph_t {
      reinterpret_cast<mt_kahypar_hypergraph_s*>(
        &Base::currentHypergraph()), Hypergraph::TYPE };
  }

  mt_kahypar_partitioned_hypergraph_t coarsestPartitionedHypergraphImpl() override {
    return mt_kahypar_partitioned_hypergraph_t {
      reinterpret_cast<mt_kahypar_partitioned_hypergraph_s*>(
        &Base::currentPartitionedHypergraph()), PartitionedHypergraph::TYPE };
  }

  std::unique_ptr<kaminpar::shm::CSRGraph> buildBipartiteGraphRep() {
      using namespace kaminpar;
      using namespace kaminpar::shm;

      const Hypergraph& hg = Base::currentHypergraph();
      const HypernodeID num_nodes = hg.initialNumNodes();

      const NodeID n = num_nodes + hg.initialNumEdges();
      const EdgeID m = 2 * hg.initialNumPins();
      // scale edge weights to approximate float division
      constexpr EdgeWeight SCALE_EDGE_WEIGHT = 2 << 8;

      StaticArray<EdgeID> nodes(n + 1);
      StaticArray<NodeID> edges(m);
      StaticArray<NodeWeight> node_weights(n);
      StaticArray<EdgeWeight> edge_weights(m);

      // set node weights and node degrees
      nodes[0] = 0;
      tbb::parallel_invoke([&] {
                               tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
                                   NodeID u = _current_vertices[id];
                                   node_weights[u] = hg.nodeWeight(id);
                                   nodes[u + 1] = hg.nodeDegree(id);
                               });
                           }, [&]() {
                               tbb::parallel_for<NodeID>(num_nodes, n, [&](const NodeID u) {
                                   node_weights[u] = 0;
                                   nodes[u + 1] = hg.edgeSize(u - num_nodes);
                               });
                           });
      ASSERT(node_weights[_current_vertices[0]]==hg.nodeWeight(0));
      ASSERT(nodes[_current_vertices[0]+1] == hg.nodeDegree(0));
      ASSERT(node_weights[num_nodes] == 0);
      ASSERT(nodes[num_nodes+1] == hg.edgeSize(0));
      ASSERT(nodes[n] == hg.edgeSize(hg.initialNumEdges()-1));
      // compute offset of neighborhoods in edge array
      parallel_prefix_sum(nodes.begin()+1, nodes.end(), nodes.begin()+1, [&](EdgeID x, EdgeID y) { return x + y; }, 0);

      // obtain edge weight for edge of the graph
      auto graphEdgeWeight = [](const Hypergraph &hypergraph, const HyperedgeID he) {
          return hypergraph.edgeWeight(he) * SCALE_EDGE_WEIGHT / hypergraph.edgeSize(he);
      };

      // set edges and edge weights
      tbb::parallel_invoke([&]() {
                               // neighborhoods representing incident nets
                               tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
                                   NodeID u = _current_vertices[id];
                                   EdgeID counter = 0;
                                   for (const HyperedgeID &he: hg.incidentEdges(id)) {
                                       ASSERT(hg.edgeSize(he) >= 2, "Empty or single nets encountered.");
                                       edges[nodes[u] + counter] = he + num_nodes;
                                       // hyperedges are shifted by num_nodes
                                       edge_weights[nodes[u] + counter] = graphEdgeWeight(hg, he);
                                       ++counter;
                                   }
                               });
                           }, [&]() {
                               // neighborhoods representing pins
                               tbb::parallel_for<NodeID>(UL(0), hg.initialNumEdges(), [&](const NodeID he) {
                                   EdgeID counter = 0;
                                   const EdgeWeight edge_weight = graphEdgeWeight(hg, he);
                                   for (const HypernodeID &hv: hg.pins(he)) {
                                       edges[nodes[he+num_nodes] + counter] = _current_vertices[hv]; // hypervertex ids remain identical
                                       edge_weights[nodes[he+num_nodes] + counter] = edge_weight;
                                       ++counter;
                                   }
                               });
                           });

      constexpr bool neighborhood_sorted = false;
      return std::make_unique<kaminpar::shm::CSRGraph>(std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights),
                      neighborhood_sorted);
  }


  using Base = MultilevelCoarsenerBase<TypeTraits>;
  using Base::_hg;
  using Base::_context;
  using Base::_timer;
  using Base::_uncoarseningData;

  HypernodeID _initial_num_nodes;
  int _pass_nr;
  utils::ProgressBar _progress_bar;
  bool _enable_randomization;

  parallel::scalable_vector<HypernodeID> _current_vertices;
};
}
