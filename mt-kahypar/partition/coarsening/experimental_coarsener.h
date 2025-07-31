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

#include "mt-kahypar/partition/context_enum_classes.h"
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
#include "kaminpar-common/random.h"
#include "kaminpar-shm/graphutils/permutator.h"

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

  std::unique_ptr<kaminpar::shm::CSRGraph> buildBipartiteGraphRep();

  std::unique_ptr<kaminpar::shm::CSRGraph> buildCycleMatchingRep();
  
  std::unique_ptr<kaminpar::shm::CSRGraph> buildCycleRandomMatchingRep();
  
  std::unique_ptr<kaminpar::shm::CSRGraph> buildCliqueRep();

  [[nodiscard]] kaminpar::shm::EdgeID
  countEdgesInEexpansion(HyperedgeID he_size);

  [[nodiscard]] kaminpar::shm::EdgeWeight
  getExpandedEdgeWeight(const HyperedgeID he,
                        const kaminpar::shm::EdgeID num_edges_in_expansion,
                        const kaminpar::shm::EdgeID max_num_edges_in_expansion);

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
template <typename TypeTraits>
inline kaminpar::shm::EdgeWeight
ExperimentalCoarsener<TypeTraits>::getExpandedEdgeWeight(
    const HyperedgeID he, const kaminpar::shm::EdgeID num_edges_in_expansion,
    const kaminpar::shm::EdgeID max_num_edges_in_expansion) {
  using namespace kaminpar::shm;
  const Hypergraph &hg = Base::currentHypergraph();
  const GraphRepEdgeWeight rep = _context.coarsening.rep_edge_weight;

  if (rep == GraphRepEdgeWeight::unit) {
    return 1;
  } else {
    EdgeWeight edge_weight = hg.edgeWeight(he);
    const EdgeWeight SCALE_EDGE_WEIGHT = max_num_edges_in_expansion * 10;

    if (rep == GraphRepEdgeWeight::normalized_hyperedge_weight) {
      edge_weight = edge_weight * SCALE_EDGE_WEIGHT / num_edges_in_expansion;
    }
    return edge_weight;
  }
}

template <typename TypeTraits>
inline kaminpar::shm::EdgeID
ExperimentalCoarsener<TypeTraits>::countEdgesInEexpansion(HyperedgeID he_size) {
  const GraphRepresentation rep = _context.coarsening.rep;
  ASSERT(he_size >= 2);
  if (he_size <= 3) {
    return 3;
  } else if (rep == GraphRepresentation::bipartite) {
    return he_size;
  } else if (rep == GraphRepresentation::clique) {
    return (he_size - 1) * he_size / 2;
  } else {
    ASSERT(rep == GraphRepresentation::cycle_matching ||
           rep == GraphRepresentation::cycle_random_matching)
    return he_size + (he_size / 2);
  }
}
}
