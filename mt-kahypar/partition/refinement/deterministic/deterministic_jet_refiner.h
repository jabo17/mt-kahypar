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

#include "mt-kahypar/datastructures/buffered_vector.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/refinement/i_rebalancer.h"

#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/partition/refinement/gains/gain_cache_ptr.h"
#include "mt-kahypar/utils/reproducible_random.h"

namespace mt_kahypar {

template<typename GraphAndGainTypes>
class DeterministicJetRefiner final : public IRefiner {

  using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
  using GainComputation = typename GraphAndGainTypes::GainComputation;
  using RatingMap = typename GainComputation::RatingMap;
  using AttributedGains = typename GraphAndGainTypes::AttributedGains;
  using GainCache = typename GraphAndGainTypes::GainCache;
  using ActiveNodes = typename parallel::scalable_vector<HypernodeID>;

public:
  explicit DeterministicJetRefiner(const HypernodeID num_hypernodes,
    const HyperedgeID num_hyperedges,
    const Context& context,
    gain_cache_t,
    IRebalancer&) :
    _context(context),
    _gain_computation(context, true /* disable_randomization */),
    _current_k(context.partition.k),
    _top_level_num_nodes(num_hypernodes),
    _current_partition_is_best(true),
    _best_partition(num_hypernodes, kInvalidPartition),
    _current_partition(num_hypernodes, kInvalidPartition),
    _gains_and_target(num_hypernodes),
    _prng(context.partition.seed),
    _active_nodes(),
    _locks(num_hypernodes) {}

private:
  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

  bool refineImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
    const vec<HypernodeID>& refinement_nodes,
    Metrics& best_metrics, double) final;

  void initializeImpl(mt_kahypar_partitioned_hypergraph_t& phg);

  void computeActiveNodesFromGraph(const PartitionedHypergraph& hypergraph, bool first_round);

  Gain performMoveWithAttributedGain(PartitionedHypergraph& phg, const Move& m, bool activate_neighbors);

  void storeCurrentPartition(const PartitionedHypergraph& hypergraph, parallel::scalable_vector<PartitionID>& parts);

  void rollbackToBestPartition(PartitionedHypergraph& hypergraph);

  const Context& _context;
  PartitionID _current_k;
  HypernodeID _top_level_num_nodes;
  bool _current_partition_is_best;
  std::mt19937 _prng;
  ActiveNodes _active_nodes;
  parallel::scalable_vector<HypernodeID> _moves;
  parallel::scalable_vector<PartitionID> _best_partition;
  parallel::scalable_vector<PartitionID> _current_partition;
  GainComputation _gain_computation;
  parallel::scalable_vector<std::pair<Gain, PartitionID>> _gains_and_target;
  kahypar::ds::FastResetFlagArray<> _locks;
  IRebalancer& _rebalancer;
};

}
