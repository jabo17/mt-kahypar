/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
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

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/refinement/i_rebalancer.h"
#include "mt-kahypar/partition/refinement/gains/gain_cache_ptr.h"
#include "mt-kahypar/utils/cast.h"

namespace mt_kahypar {
template <typename GraphAndGainTypes>
class MDRebalancer final : public IRebalancer {
 private:
  using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
  using GainCache = typename GraphAndGainTypes::GainCache;
  using GainCalculator = typename GraphAndGainTypes::GainComputation;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

public:

  explicit MDRebalancer(HypernodeID , const Context& context, GainCache& gain_cache) :
    _context(context),
    _gain_cache(gain_cache),
    _current_k(context.partition.k),
    _gain(context) { }

  explicit MDRebalancer(HypernodeID num_nodes, const Context& context, gain_cache_t gain_cache) :
    MDRebalancer(num_nodes, context, GainCachePtr::cast<GainCache>(gain_cache)) {}

  MDRebalancer(const MDRebalancer&) = delete;
  MDRebalancer(MDRebalancer&&) = delete;

  MDRebalancer & operator= (const MDRebalancer &) = delete;
  MDRebalancer & operator= (MDRebalancer &&) = delete;

  void initializeImpl(mt_kahypar_partitioned_hypergraph_t&) final;

  bool refineImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                  const vec<HypernodeID>&,
                  Metrics& best_metrics,
                  double) override final;

  bool refineAndOutputMovesImpl(mt_kahypar_partitioned_hypergraph_t&,
                                const vec<HypernodeID>&,
                                vec<vec<Move>>&,
                                Metrics&,
                                const double) override final;

  bool refineAndOutputMovesLinearImpl(mt_kahypar_partitioned_hypergraph_t&,
                                      const vec<HypernodeID>&,
                                      vec<Move>&,
                                      Metrics&,
                                      const double) override final;

private:
  bool refineInternal(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                      vec<vec<Move>>* moves_by_part,
                      vec<Move>* moves_linear,
                      Metrics& best_metrics);

  void resizeDataStructuresForCurrentK() {
    // If the number of blocks changes, we resize data structures
    // (can happen during deep multilevel partitioning)
    if ( _current_k != _context.partition.k ) {
      _current_k = _context.partition.k;
      _gain.changeNumberOfBlocks(_current_k);
    }
  }

  const Context& _context;
  GainCache& _gain_cache;
  PartitionID _current_k;
  GainCalculator _gain;
};

}  // namespace kahypar
