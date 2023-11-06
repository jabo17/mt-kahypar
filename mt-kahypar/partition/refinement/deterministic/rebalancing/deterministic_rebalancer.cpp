/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "mt-kahypar/partition/refinement/deterministic/rebalancing/deterministic_rebalancer.h"


#include <boost/dynamic_bitset.hpp>

#include <tbb/parallel_for_each.h>
#include "tbb/enumerable_thread_specific.h"
#include "tbb/parallel_for.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/cast.h"

#include <tbb/parallel_sort.h>

namespace mt_kahypar {

float transformGain(Gain gain_, HypernodeWeight wu) {
  float gain = gain_;
  if (gain > 0) {
    gain /= wu;
  } else if (gain < 0) {
    gain *= wu;
  }
  return gain;
}

template <typename GraphAndGainTypes>
bool DeterministicRebalancer<GraphAndGainTypes>::refineImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
  const vec<HypernodeID>&,
  Metrics& best_metrics,
  double) {
  PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
  resizeDataStructuresForCurrentK();
  if (_max_part_weights == nullptr) {
    _max_part_weights = &_context.partition.max_part_weights[0];
  }
  _gain_computation.reset();
  initializeDataStructures(phg);

  while (_num_imbalanced_parts > 0) {
    weakRebalancingRound(phg);
    HEAVY_REFINEMENT_ASSERT(checkPreviouslyOverweightParts(phg));
    updateImbalance(phg);
  }

  Gain delta = _gain_computation.delta();
  HEAVY_REFINEMENT_ASSERT(best_metrics.quality + delta == metrics::quality(phg, _context),
    V(best_metrics.quality) << V(delta) << V(metrics::quality(phg, _context)));
  best_metrics.quality += delta;
  best_metrics.imbalance = metrics::imbalance(phg, _context);
  DBG << "[REBALANCE] " << V(delta) << "  imbalance=" << best_metrics.imbalance;
  _max_part_weights = nullptr;
  return delta < 0;
}

template <typename  GraphAndGainTypes>
void DeterministicRebalancer< GraphAndGainTypes>::initializeDataStructures(const PartitionedHypergraph& phg) {
  updateImbalance(phg);
}


template <typename  GraphAndGainTypes>
void DeterministicRebalancer<GraphAndGainTypes>::updateImbalance(const PartitionedHypergraph& phg) {
  _num_imbalanced_parts = 0;
  _num_valid_targets = 0;
  for (PartitionID part = 0; part < _context.partition.k; ++part) { // TODO: Not worth parallelizing?!
    if (imbalance(phg, part) > 0) {
      ++_num_imbalanced_parts;
    } else if (isValidTarget(phg, part, 0)) {
      ++_num_valid_targets;
    }
  }
}

template <typename GraphAndGainTypes>
rebalancer::RebalancingMove DeterministicRebalancer<GraphAndGainTypes>::computeGainAndTargetPart(const PartitionedHypergraph& phg,
  const HypernodeID hn,
  bool non_adjacent_blocks) {
  //utils::Timer& timer = utils::Utilities::instance().getTimer(_context.utility_id);
  //timer.start_timer("pre_loop", "Pre-Loop");
  const HypernodeWeight hn_weight = phg.nodeWeight(hn);
  RatingMap& tmp_scores = _gain_computation.localScores();
  Gain isolated_block_gain = 0;
  _gain_computation.precomputeGains(phg, hn, tmp_scores, isolated_block_gain, true);

  Gain best_gain = isolated_block_gain;
  PartitionID best_target = kInvalidPartition;
  for (const auto& entry : tmp_scores) {
    const PartitionID to = entry.key;
    const Gain gain = _gain_computation.gain(entry.value, isolated_block_gain);
    if (gain <= best_gain && isValidTarget(phg, to, hn_weight)) {
      best_gain = gain;
      best_target = to;
    }
  }
  //timer.stop_timer("pre_loop");

  //timer.start_timer("loop", "Loop");
  // if no adjacent block with free capacity exists, we need to consider non-adjacent blocks
  if (non_adjacent_blocks && best_target == kInvalidPartition) {
    // we start with a block that is chosen by random, to ensure a reasonable distribution of nodes
    // to target blocks (note: this does not always result in a uniform distribution since some blocks
    // are not an acceptable target, but it should be good enough)
    const PartitionID start = hn % _current_k;
    PartitionID to = start;
    do {
      if (isValidTarget(phg, to, hn_weight)
        && !tmp_scores.contains(to)) {
        best_target = to;
        break;
      }

      ++to;
      if (to == _context.partition.k) {
        to = 0;
      }
    } while (to != start);
    // assertion does not always hold with tight balance constraint or large node weights
    // ASSERT(best_target != kInvalidPartition);
  }
  //timer.stop_timer("loop");

  tmp_scores.clear();
  const HypernodeWeight weight = phg.nodeWeight(hn);
  return { hn, best_target, transformGain(best_gain, weight) };
}

template <typename  GraphAndGainTypes>
void DeterministicRebalancer<GraphAndGainTypes>::weakRebalancingRound(PartitionedHypergraph& phg) {
  // utils::Timer& timer = utils::Utilities::instance().getTimer(_context.utility_id);
  //timer.start_timer("alloc", "Tmp move allocation");
  for (auto& moves : tmp_potential_moves) {
    moves.clear_parallel();
  }
  //timer.stop_timer("alloc");
    // calculate gain and target for each node in a overweight part
    // group moves by source part
  //timer.start_timer("gain_computation", "Gain Computation");
  phg.doParallelForAllNodes([&](const HypernodeID hn) {
    const PartitionID from = phg.partID(hn);
    const HypernodeWeight weight = phg.nodeWeight(hn);
    if (imbalance(phg, from) > 0 && mayMoveNode(phg, from, weight)) {
      // timer.start_timer("actual_gain_computation", "Actual Gain Computation");
       //timer.stop_timer("actual_gain_computation");

       //timer.start_timer("streaming", "Streaming");
      tmp_potential_moves[from].stream(computeGainAndTargetPart(phg, hn, true));
      //timer.stop_timer("streaming");

    }
  });
  //timer.stop_timer("gain_computation");
  //timer.start_timer("rest", "Move Selection and Execution");
  for (size_t b = 0; b < _context.refinement.deterministic_refinement.jet.num_buckets; ++b) {

    tbb::parallel_for(0UL, _moves.size(), [&](const size_t i) {
      //for (size_t i = 0; i < _moves.size(); ++i) {
        //timer.start_timer("copy_moves", "Copy Moves");
      if (b == 0UL) {
        _moves[i] = tmp_potential_moves[i].copy_parallel();
      }
      const size_t bucket_size = _moves[i].size() / _context.refinement.deterministic_refinement.jet.num_buckets + 1;
      const size_t bucket_start_index = b * bucket_size;
      const size_t bucket_end_index = std::min(_moves[i].size(), bucket_start_index + bucket_size);
      //timer.stop_timer("copy_moves");
      if (_moves[i].size() > 0) {
        // sort the moves from each overweight part by priority
        //timer.start_timer("sorting", "Sorting");
        tbb::parallel_sort(_moves[i].begin() + bucket_start_index, _moves[i].end(), [&](const rebalancer::RebalancingMove& a, const rebalancer::RebalancingMove& b) {
          return a.priority < b.priority || (a.priority == b.priority && a.hn > b.hn);
        });
        //timer.stop_timer("sorting");
        // calculate perfix sum for each source-part to know which moves to execute (prefix_sum > current_weight - max_weight)
        //timer.start_timer("find_moves", "Find Moves");
        _move_weights[i].resize(_moves[i].size());
        tbb::parallel_for(bucket_start_index, _moves[i].size(), [&](const size_t j) {
          _move_weights[i][j] = phg.nodeWeight(_moves[i][j].hn);
        });
        parallel_prefix_sum(_move_weights[i].begin() + bucket_start_index, _move_weights[i].begin() + bucket_end_index, _move_weights[i].begin() + bucket_start_index, std::plus<HypernodeWeight>(), 0);
        const size_t last_move_idx = std::upper_bound(_move_weights[i].begin() + bucket_start_index, _move_weights[i].begin() + bucket_end_index, phg.partWeight(i) - _max_part_weights[i] - 1) - _move_weights[i].begin();
        //timer.stop_timer("find_moves");

        // timer.start_timer("exe_moves", "Execute Moves");
        //DBG << "exe_moves" << V(bucket_start_index) << ", " << V(last_move_idx) << ", " << V(bucket_end_index) << ", " << V(_moves[i].size());
        tbb::parallel_for(b * bucket_size, std::min(last_move_idx + 1, _moves[i].size()), [&](const size_t j) {
          const auto move = _moves[i][j];
          changeNodePart(phg, move.hn, i, move.to, false);
        });
        if (last_move_idx < bucket_end_index) {
          _moves[i].clear();
        }
        //timer.stop_timer("exe_moves");
      }
      // }
    });
    
    for (size_t i = 0; i < _moves.size(); ++i) {
      const size_t bucket_size = _moves[i].size() / _context.refinement.deterministic_refinement.jet.num_buckets + 1;
      const size_t start = (b + 1) * bucket_size;
      tbb::parallel_for(start, _moves[i].size(), [&](const size_t j) {
        _moves[i][j] = computeGainAndTargetPart(phg, _moves[i][j].hn, true);
      });
    }
  }
  // });
   //timer.stop_timer("rest");
}

// explicitly instantiate so the compiler can generate them when compiling this cpp file
namespace {
#define DETERMINISTIC_REBALANCER(X) DeterministicRebalancer<X>
}

// explicitly instantiate so the compiler can generate them when compiling this cpp file
INSTANTIATE_CLASS_WITH_VALID_TRAITS(DETERMINISTIC_REBALANCER)
}
