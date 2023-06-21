/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include <limits>
#include <cstdint>

#include <mt-kahypar/datastructures/concurrent_bucket_map.h>
#include <mt-kahypar/datastructures/priority_queue.h>
#include <mt-kahypar/partition/context.h>
#include <mt-kahypar/parallel/work_stack.h>

#include "external_tools/kahypar/kahypar/datastructure/fast_reset_flag_array.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace mt_kahypar {


struct GlobalMoveTracker {
  vec<Move> moveOrder;
  vec<MoveID> moveOfNode;
  vec<uint8_t> rebalancingMoves;
  CAtomic<MoveID> runningMoveID;
  MoveID firstMoveID = 1;

  explicit GlobalMoveTracker(size_t numNodes = 0) :
          moveOrder(numNodes),
          moveOfNode(numNodes, 0),
          rebalancingMoves(),
          runningMoveID(1) { }

  // Returns true if stored move IDs should be reset
  bool reset() {
    rebalancingMoves.clear();
    if (runningMoveID.load() >= std::numeric_limits<MoveID>::max() - moveOrder.size() - 20) {
      tbb::parallel_for(UL(0), moveOfNode.size(), [&](size_t i) { moveOfNode[i] = 0; }, tbb::static_partitioner());
      firstMoveID = 1;
      runningMoveID.store(1);
      return true;
    } else {
      firstMoveID = ++runningMoveID;
      return false;
    }
  }

  MoveID insertMove(Move &m) {
    const MoveID move_id = runningMoveID.fetch_add(1, std::memory_order_relaxed);
    assert(move_id - firstMoveID < moveOrder.size());
    moveOrder[move_id - firstMoveID] = m;
    moveOrder[move_id - firstMoveID].gain = 0;      // set to zero so the recalculation can safely distribute
    moveOfNode[m.node] = move_id;
    return move_id;
  }

  Move& getMove(MoveID move_id) {
    assert(move_id - firstMoveID < moveOrder.size());
    return moveOrder[move_id - firstMoveID];
  }

  bool wasNodeMovedInThisRound(HypernodeID u) const {
    const MoveID m_id = moveOfNode[u];
    return m_id >= firstMoveID
           && m_id < runningMoveID.load(std::memory_order_relaxed)  // active move ID
           && moveOrder[m_id - firstMoveID].isValid();      // not reverted already
  }

  MoveID numPerformedMoves() const {
    return runningMoveID.load(std::memory_order_relaxed) - firstMoveID;
  }

  bool isMoveStale(const MoveID move_id) const {
    return move_id < firstMoveID;
  }

  bool isRebalancingMove(const MoveID move_id) const {
    return !rebalancingMoves.empty() && static_cast<bool>(rebalancingMoves[move_id - firstMoveID]);
  }
};

struct NodeTracker {
  vec<CAtomic<SearchID>> searchOfNode;
  kahypar::ds::FastResetFlagArray<> lockedVertices;

  SearchID releasedMarker = 1;
  SearchID deactivatedNodeMarker = 2;
  CAtomic<SearchID> highestActiveSearchID { 2 };
  bool vertex_locking = false;

  explicit NodeTracker(size_t numNodes = 0) : searchOfNode(numNodes, CAtomic<SearchID>(0)), lockedVertices() {
    if (numNodes > 0) {
      lockedVertices.setSize(numNodes);
    }
  }

  // only the search that owns u is allowed to call this
  void deactivateNode(HypernodeID u, SearchID search_id) {
    assert(searchOfNode[u].load() == search_id);
    unused(search_id);
    searchOfNode[u].store(deactivatedNodeMarker, std::memory_order_release);
  }

  bool isLocked(HypernodeID u) {
    return searchOfNode[u].load(std::memory_order_relaxed) == deactivatedNodeMarker;
  }

  void releaseNode(HypernodeID u) {
    searchOfNode[u].store(releasedMarker, std::memory_order_relaxed);
  }

  bool isSearchInactive(SearchID search_id) const {
    return search_id < deactivatedNodeMarker;
  }

  bool canNodeStartNewSearch(HypernodeID u) const {
    return isSearchInactive( searchOfNode[u].load(std::memory_order_relaxed) );
  }

  bool tryAcquireNode(HypernodeID u, SearchID new_search) {
    SearchID current_search = searchOfNode[u].load(std::memory_order_relaxed);
    return isSearchInactive(current_search) && !vertexIsLocked(u)
            && searchOfNode[u].compare_exchange_strong(current_search, new_search, std::memory_order_acq_rel);
  }

  void requestNewSearches(SearchID max_num_searches) {
    if (highestActiveSearchID.load(std::memory_order_relaxed) >= std::numeric_limits<SearchID>::max() - max_num_searches - 20) {
      tbb::parallel_for(UL(0), searchOfNode.size(), [&](const size_t i) {
        searchOfNode[i].store(0, std::memory_order_relaxed);
      });
      highestActiveSearchID.store(1, std::memory_order_relaxed);
    }
    deactivatedNodeMarker = ++highestActiveSearchID;
    releasedMarker = deactivatedNodeMarker - 1;
  }

  bool vertexIsLocked(const HypernodeID node) const {
    return vertex_locking && lockedVertices[node];
  }

  bool vertexIsSoftLocked(const HypernodeID node) const {
    // TODO: this is quite hacky...
    return lockedVertices[node];
  }
};


// Contains data required for unconstrained FM: We group non-border nodes in buckets based on their
// incident weight to node weight ratio. This allows to give a (pessimistic) estimate of the effective
// gain for moves that violate the balance constraint
struct UnconstrainedFMData {
  using BucketMap = ds::ConcurrentBucketMap<HypernodeID>;
  using AtomicWeight = parallel::IntegralAtomicWrapper<HypernodeWeight>;

  // TODO(maas): in weighted graphs the constant number of buckets might be problematic
  static constexpr size_t NUM_BUCKETS = 16;
  static constexpr size_t BUCKET_FACTOR = 32;

  bool initialized = false;
  PartitionID current_k;
  parallel::scalable_vector<HypernodeWeight> bucket_weights;
  parallel::scalable_vector<HypernodeWeight> upper_weight_limits;
  parallel::scalable_vector<AtomicWeight> consumed_bucket_weights;
  tbb::enumerable_thread_specific<parallel::scalable_vector<HypernodeWeight>> local_bucket_weights;
  kahypar::ds::FastResetFlagArray<> rebalancing_nodes;
  parallel::scalable_vector<HyperedgeWeight> incident_weight_of_node;

  explicit UnconstrainedFMData():
    initialized(false),
    current_k(0),
    bucket_weights(),
    consumed_bucket_weights(),
    local_bucket_weights(),
    rebalancing_nodes() { }

  template<typename PartitionedHypergraphT>
  void precomputeForLevel(const PartitionedHypergraphT& phg);

  template<typename PartitionedHypergraphT>
  void initialize(const Context& context, const PartitionedHypergraphT& phg);

  HypernodeWeight maximumImbalance(PartitionID to) const {
    ASSERT(static_cast<size_t>(to) < upper_weight_limits.size());
    return upper_weight_limits[to];
  }

  Gain estimatedPenaltyForImbalance(PartitionID to, HypernodeWeight total_imbalance) const;

  Gain estimatedPenaltyForDelta(PartitionID to, HypernodeWeight old_weight, HypernodeWeight new_weight) const;

  Gain estimatedPenaltyForImbalancedMove(PartitionID to, HypernodeWeight weight) const;

  Gain applyEstimatedPenaltyForImbalancedMove(PartitionID to, HypernodeWeight weight);

  void revertImbalancedMove(PartitionID to, HypernodeWeight weight);

  bool isRebalancingNode(HypernodeID hn) const {
    return initialized && rebalancing_nodes[hn];
  }

  void changeNumberOfBlocks(PartitionID new_k) {
    if (new_k != current_k) {
      current_k = new_k;
      local_bucket_weights = tbb::enumerable_thread_specific<vec<HypernodeWeight>>(new_k * NUM_BUCKETS);
      reset();
    }
  }

  void reset() {
    rebalancing_nodes.reset();
    bucket_weights.assign(current_k * NUM_BUCKETS, 0);
    upper_weight_limits.assign(current_k, 0);
    consumed_bucket_weights.assign(current_k * NUM_BUCKETS, AtomicWeight(0));
    for (auto& local_weights: local_bucket_weights) {
      local_weights.assign(current_k * NUM_BUCKETS, 0);
    }
    initialized = false;
  }

 private:
  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  double estimatedPenaltyFromIndex(PartitionID to, size_t bucketId, HypernodeWeight remaining) const;

  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE size_t indexForBucket(PartitionID block, size_t bucketId) const {
    ASSERT(bucketId < NUM_BUCKETS && block * NUM_BUCKETS + bucketId < bucket_weights.size());
    return block * NUM_BUCKETS + bucketId;
  }

  // upper bound of gain values in bucket
  double gainPerWeightForBucket(size_t bucketId) const {
    ASSERT(bucketId < NUM_BUCKETS);
    if (bucketId > 1) {
      return std::pow(1.5, bucketId - 2);
    } else if (bucketId == 1) {
      return 0.5;
    } else {
      return 0;
    }
  }

  size_t bucketForGainPerWeight(double gainPerWeight) const {
    if (gainPerWeight >= 1) {
      return 2 + std::ceil(std::log(gainPerWeight) / std::log(1.5));
    } else if (gainPerWeight > 0.5) {
      return 2;
    } else if (gainPerWeight > 0) {
      return 1;
    } else {
      return 0;
    }
  }
};


struct FMSharedData {
  // ! Number of Nodes
  size_t numberOfNodes;

  // ! Nodes to initialize the localized FM searches with
  WorkContainer<HypernodeID> refinementNodes;

  // ! PQ handles shared by all threads (each vertex is only held by one thread)
  vec<PosT> vertexPQHandles;

  // ! Stores the sequence of performed moves and assigns IDs to moves that can be used in the global rollback code
  GlobalMoveTracker moveTracker;

  // ! Tracks the current search of a node, and if a node can still be added to an active search
  NodeTracker nodeTracker;

  // ! Stores the designated target part of a vertex, i.e. the part with the highest gain to which moving is feasible
  vec<PartitionID> targetPart;

  // ! Additional data for unconstrained FM algorithm
  UnconstrainedFMData unconstrained;

  // ! Stop parallel refinement if finishedTasks > finishedTasksLimit to avoid long-running single searches
  CAtomic<size_t> finishedTasks;
  size_t finishedTasksLimit = std::numeric_limits<size_t>::max();

  // ! Switch to applying moves directly if the use of local delta partitions exceeded a memory limit
  bool deltaExceededMemoryConstraints = false;
  size_t deltaMemoryLimitPerThread = 0;

  bool release_nodes = true;
  bool perform_moves_global = true;

  FMSharedData(size_t numNodes, size_t numThreads, bool initialize_unconstrained) :
    numberOfNodes(numNodes),
    refinementNodes(), //numNodes, numThreads),
    vertexPQHandles(), //numPQHandles, invalid_position),
    moveTracker(), //numNodes),
    nodeTracker(), //numNodes),
    targetPart(),
    unconstrained() {
    finishedTasks.store(0, std::memory_order_relaxed);

    // 128 * 3/2 GB --> roughly 1.5 GB per thread on our biggest machine
    deltaMemoryLimitPerThread = 128UL * (UL(1) << 30) * 3 / ( 2 * std::max(UL(1), numThreads) );

    tbb::parallel_invoke([&] {
      moveTracker.moveOrder.resize(numNodes);
    }, [&] {
      moveTracker.moveOfNode.resize(numNodes);
    }, [&] {
      nodeTracker.searchOfNode.resize(numNodes, CAtomic<SearchID>(0));
    }, [&] {
      nodeTracker.lockedVertices.setSize(numNodes);
    }, [&] {
      vertexPQHandles.resize(numNodes, invalid_position);
    }, [&] {
      refinementNodes.tls_queues.resize(numThreads);
    }, [&] {
      targetPart.resize(numNodes, kInvalidPartition);
    }, [&] {
      if (initialize_unconstrained) {
        unconstrained.rebalancing_nodes.setSize(numNodes);
        unconstrained.incident_weight_of_node.resize(numNodes);
      }
    });
  }

  FMSharedData(size_t numNodes, bool initialize_unconstrained) :
    FMSharedData(
      numNodes,
      TBBInitializer::instance().total_number_of_threads(),
      initialize_unconstrained)  { }

  FMSharedData() :
    FMSharedData(0, 0, false) { }

  bool lockVertexForNextRound(const HypernodeID node, const Context& context) {
    ASSERT(!moveTracker.isRebalancingMove(moveTracker.moveOfNode[node]));
    if (context.refinement.fm.vertex_locking > 0) {
      mt_kahypar::utils::Randomize& randomize = mt_kahypar::utils::Randomize::instance();
      if (context.refinement.fm.vertex_locking == 1 ||
          randomize.getRandomFloat(0.0, 1.0, SCHED_GETCPU) < context.refinement.fm.vertex_locking) {
        nodeTracker.lockedVertices.set(node);
        return true;
      }
    }
    return false;
  }

  void memoryConsumption(utils::MemoryTreeNode* parent) const {
    ASSERT(parent);

    utils::MemoryTreeNode* shared_fm_data_node = parent->addChild("Shared FM Data");

    utils::MemoryTreeNode* pq_handles_node = shared_fm_data_node->addChild("PQ Handles");
    pq_handles_node->updateSize(vertexPQHandles.capacity() * sizeof(PosT));
    utils::MemoryTreeNode* move_tracker_node = shared_fm_data_node->addChild("Move Tracker");
    move_tracker_node->updateSize(moveTracker.moveOrder.capacity() * sizeof(Move) +
                                  moveTracker.moveOfNode.capacity() * sizeof(MoveID));
    utils::MemoryTreeNode* node_tracker_node = shared_fm_data_node->addChild("Node Tracker");
    node_tracker_node->updateSize(nodeTracker.searchOfNode.capacity() * sizeof(SearchID));
    refinementNodes.memoryConsumption(shared_fm_data_node);
    // TODO(maas): unconstrained FM data
  }
};

struct FMStats {
  size_t retries = 0;
  size_t extractions = 0;
  size_t pushes = 0;
  size_t moves = 0;
  size_t local_reverts = 0;
  size_t task_queue_reinsertions = 0;
  size_t best_prefix_mismatch = 0;
  size_t rebalancing_node_moves = 0;
  Gain estimated_improvement = 0;


  void clear() {
    retries = 0;
    extractions = 0;
    pushes = 0;
    moves = 0;
    local_reverts = 0;
    task_queue_reinsertions = 0;
    best_prefix_mismatch = 0;
    rebalancing_node_moves = 0;
    estimated_improvement = 0;
  }

  void merge(FMStats& other) {
    other.retries += retries;
    other.extractions += extractions;
    other.pushes += pushes;
    other.moves += moves;
    other.local_reverts += local_reverts;
    other.task_queue_reinsertions += task_queue_reinsertions;
    other.best_prefix_mismatch += best_prefix_mismatch;
    other.rebalancing_node_moves += rebalancing_node_moves;
    other.estimated_improvement += estimated_improvement;
    clear();
  }

  std::string serialize() const {
    std::stringstream os;
    os  << V(retries) << " " << V(extractions) << " " << V(pushes) << " "
        << V(moves) << " " << V(local_reverts) << " " << V(estimated_improvement) << " "
        << V(best_prefix_mismatch) <<  " " << V(rebalancing_node_moves);
    return os.str();
  }
};


}