/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2021 Noah Wahl <noah.wahl@student.kit.edu>
 * Copyright (C) 2021 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 *
 * KaHyPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaHyPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaHyPar.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <mt-kahypar/definitions.h>
#include <mt-kahypar/datastructures/priority_queue.h>
#include <mt-kahypar/partition/context.h>

namespace mt_kahypar {
class JudiciousGainCache final {
public:
  using PriorityQueue = ds::ExclusiveHandleHeap<ds::MaxHeap<HypernodeWeight, PartitionID>>;

  explicit JudiciousGainCache(const Context& context, HypernodeID num_nodes) :
    _context(context),
    _toPQs(static_cast<size_t>(context.partition.k), PriorityQueue(num_nodes)),
    _blockPQ(static_cast<size_t>(context.partition.k)),
    _part_loads(static_cast<size_t>(context.partition.k)) { }

  void insert(const PartitionedHypergraph& phg, const HypernodeID v) {
    const Gain benefit = phg.moveFromBenefit(v);
    for (PartitionID i = 0; i < _context.partition.k; ++i) {
      if (i != phg.partID(v)) {
        _toPQs[i].insert(v, benefit - phg.moveToPenalty(v, i));
      }
    }
  }

  void initBlockPQ(const PartitionedHypergraph &phg) {
    ASSERT(_blockPQ.empty());
    for (PartitionID i = 0; i < _context.partition.k; ++i) {
      _part_loads.insert(i, phg.partLoad(i));
    }
    for (PartitionID i = 0; i < _context.partition.k; ++i) {
      if (!_toPQs[i].empty()) {
        _blockPQ.insert(i, blockGain(phg, i));
      }
    }
  }

  void setActivePart(const PartitionID part_id) {
    _active_part = part_id;
  }

  void decreasePenalty(const HypernodeID v,
                    const HyperedgeWeight w, const PartitionID to) {
    _toPQs[to].increaseKey(
        v, _toPQs[to].keyOf(v) + w);
  }

  void increaseBenefit(const HypernodeID v,
                    const HyperedgeWeight w, const PartitionID from) {
    for (PartitionID i = 0; i < _context.partition.k; ++i) {
      if (i != from) {
        _toPQs[i].increaseKey(
          v, _toPQs[i].keyOf(v) + w);
      }
    }
  }

  bool findNextMove(const PartitionedHypergraph& phg, Move& m) {
    ASSERT(!_blockPQ.contains(_active_part));
    if (!updatePQs(phg)) {
      return false;
    }
    ASSERT(!_blockPQ.empty());
    const PartitionID to = _blockPQ.top();
    ASSERT(to != _active_part);
    ASSERT(!_toPQs[to].empty());
    const HypernodeID u = _toPQs[to].top();
    ASSERT(phg.partID(u) == _active_part);
    const Gain gain = use_block_load_only ? calculateRealGain(phg, to) : _blockPQ.topKey();
    m.node = u;
    m.from = phg.partID(u);
    m.to = to;
    m.gain = gain;
    _toPQs[to].deleteTop();
    for (PartitionID i = 0; i < _context.partition.k; ++i) {
      if (i != to && i != m.from) {
        _toPQs[i].remove(u);
      }
    }
    updateOrRemoveToPQFromBlocks(to, phg);
    return true;
  }

  void resetGainCache() {
    for (PartitionID i = 0; i < _context.partition.k; ++i) {
      _toPQs[i].clear();
    }
    _blockPQ.clear();
    _part_loads.clear();
  }

  // TODO: maybe share part_load PQ with refiner <2022-05-20, noahares>
  void updatePartLoads(const PartitionedHypergraph &phg,
                           const PartitionID from, const PartitionID to) {
    _part_loads.adjustKey(to, phg.partLoad(to));
    _part_loads.adjustKey(from, phg.partLoad(from));
  }

private:

  bool updatePQs(const PartitionedHypergraph &phg) {
    for (PartitionID i = 0; i < _context.partition.k; ++i) {
      updateOrRemoveToPQFromBlocks(i, phg);
    }
    return !_blockPQ.empty();
  }

  void updateOrRemoveToPQFromBlocks(const PartitionID i, const PartitionedHypergraph &phg) {
      if (!_toPQs[i].empty()) {
        _blockPQ.insertOrAdjustKey(i, blockGain(phg, i));
      } else if (_blockPQ.contains(i)) {
        _blockPQ.remove(i);
      }
  }

  Gain blockGain(const PartitionedHypergraph &phg,
                                    const PartitionID p) const {
    return use_block_load_only ? -phg.partLoad(p) : calculateRealGain(phg, p);
  }

  Gain calculateRealGain(const PartitionedHypergraph& phg, const PartitionID p) const {
    const HyperedgeWeight load_of_first = _part_loads.topKey();
    const HypernodeID u = _toPQs[p].top();
    const Gain penalty = phg.moveToPenalty(u, p) + phg.weightOfDisabledEdges(u);
    if (load_of_first == phg.partLoad(p)) return -penalty;
    const Gain benefit = phg.moveFromBenefit(u) + phg.weightOfDisabledEdges(u);
    const PartitionID from = phg.partID(u);
    const HyperedgeWeight from_load_after = phg.partLoad(from) - benefit;
    const HyperedgeWeight to_load_after = phg.partLoad(p) + penalty;
    const HyperedgeWeight load_of_second = _part_loads.keyOfSecond();
    if (_part_loads.top() == from) {
       if (from_load_after > to_load_after && from_load_after > load_of_second) return benefit;
       else if (from_load_after == to_load_after && from_load_after >= load_of_second) return 0;
       else return load_of_first - std::max(to_load_after, load_of_second);
    }
    else return std::min(load_of_first - to_load_after, 0);
  }

  const Context& _context;
  vec<PriorityQueue> _toPQs;
  PriorityQueue _blockPQ;
  ds::ExclusiveHandleHeap<ds::MaxHeap<HypernodeWeight, PartitionID>>
      _part_loads;
  // only used for assertions
  PartitionID _active_part;
  bool use_block_load_only = true;
};
}
