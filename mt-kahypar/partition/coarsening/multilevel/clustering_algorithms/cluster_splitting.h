/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Nikolai Maas <nikolai.maas@kit.edu>
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

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/array.h"
#include "mt-kahypar/datastructures/parallel_pq.h"
#include "mt-kahypar/datastructures/streaming_vector.h"
#include "mt-kahypar/datastructures/thread_safe_fast_reset_flag_array.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"

#include "mt-kahypar/partition/coarsening/multilevel/clustering_context.h"
#include "mt-kahypar/partition/coarsening/multilevel/concurrent_clustering_data.h"
#include "mt-kahypar/partition/coarsening/multilevel/multilevel_vertex_pair_rater.h"
#include "mt-kahypar/partition/coarsening/multilevel/num_nodes_tracker.h"


namespace mt_kahypar {

template<typename ScorePolicy, typename HeavyNodePenaltyPolicy, typename AcceptancePolicy>
class ClusterSplitting {
  using Rating = MultilevelVertexPairRater::Rating;
  using PQ = ds::MultiQueue<double, HypernodeID>;

 public:
  ClusterSplitting(const HypernodeID num_nodes, const Context& context):
    _context(context),
    _cluster_sizes(num_nodes),
    _community_ids(),
    _active_nodes(),
    _node_scanned(),
    _cluster_locks() {
      tbb::parallel_invoke([&] {
        _cluster_sizes.assign(num_nodes, CAtomic<HypernodeID>(0));
      }, [&] {
        _node_scanned.setSize(num_nodes);
      }, [&] {
        _cluster_locks.setSize(num_nodes);
      });
    }

  template<bool has_fixed_vertices, typename Hypergraph, typename DegreeSimilarityPolicy>
  void performClustering(Hypergraph& hg,
                         const parallel::scalable_vector<HypernodeID>& node_mapping,
                         const DegreeSimilarityPolicy& similarity_policy,
                         ClusteringContext<Hypergraph>& cc,
                         std::function<double (HypernodeID)> weight_ratio_for_node_fn = [](const HypernodeID) { return 1.0; },
                         int pass_nr = 0) {
    unused(pass_nr);
    unused(weight_ratio_for_node_fn);  // parameter only exists for compatibility with TwoHopClustering

    const double dividend = _context.coarsening.contraction_limit *
      std::pow(cc.hierarchy_contraction_limit / _context.coarsening.contraction_limit,
               _context.coarsening.splitting_node_weight_exponent);
    const double hypernode_weight_fraction =
        _context.coarsening.splitting_node_weight_factor * _context.coarsening.max_allowed_weight_multiplier / dividend;
    HypernodeWeight tmp_max_allowed_node_weight = std::ceil(hypernode_weight_fraction * hg.totalWeight());
    // TODO: copy instead of swap
    std::swap(cc.max_allowed_node_weight, tmp_max_allowed_node_weight);
    LOG << "";
    LOG << V(cc.max_allowed_node_weight) << V(tmp_max_allowed_node_weight);

    // compute current cluster sizes, copy community ids (for later restoration)
    tbb::parallel_invoke([&] {
      _cluster_sizes.assign(hg.initialNumNodes(), CAtomic<HypernodeID>(0));
      hg.doParallelForAllNodes([&](const HypernodeID hn) {
        const HypernodeID cluster_id = cc.clusterID(hn);
        _cluster_sizes[cluster_id].fetch_add(1, std::memory_order_relaxed);
      });
    }, [&] {
      _community_ids.resize(hg.initialNumNodes());
      hg.copyCommunityIDs(_community_ids);
    });

    // prepare active nodes
    ds::StreamingVector<HypernodeID> local_active_nodes;
    tbb::parallel_for(ID(0), hg.initialNumNodes(), [&](const HypernodeID id) {
      const HypernodeID hn = node_mapping[id];
      if (hg.nodeIsEnabled(hn)) {
        const HypernodeID cluster_id = cc.clusterID(hn);
        hg.setCommunityID(hn, cluster_id);
        if (clusterShouldBeSplit(cluster_id, cc)) {
          local_active_nodes.stream(hn);
        }
      }
    });
    local_active_nodes.copy_parallel(_active_nodes);
    local_active_nodes.clear_sequential();
    tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
      cc.resetCluster(hg, _active_nodes[i]);
    });
    LOG << V(_active_nodes.size()) << V(hg.initialNumNodes());

    // now the actual sub-clustering logic starts
    auto handle_node = [&](const HypernodeID hn) {
      ASSERT(hg.nodeIsEnabled(hn));
      const Rating rating = cc.template rate<ScorePolicy, HeavyNodePenaltyPolicy,
                                             AcceptancePolicy, has_fixed_vertices>(hg, hn, similarity_policy);
      if (rating.target != kInvalidHypernode) {
        ASSERT(hg.communityID(hn) == hg.communityID(rating.target));
        cc.template joinClusterIgnoringMatchingState<has_fixed_vertices>(hg, hn, rating.target);
      }
    };

    for (size_t i = 0; i < _context.coarsening.splitting_num_rounds; ++i) {
      if (_context.coarsening.prioritize_high_degree) {
        // TODO: priority
      } else {
        // We iterate in parallel over the active vertices and update their clusters
        tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
          handle_node(_active_nodes[i]);
        });
      }
    }

    // update matching states and number of nodes
    while (_active_nodes.size() > 0) {
      tbb::enumerable_thread_specific<vec<HypernodeID>> local_stacks;
      tbb::enumerable_thread_specific<vec<HypernodeID>> local_node_list;
      tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
        const HypernodeID root = _active_nodes[i];
        const HypernodeID cluster_id = cc.clusterID(root);
        if (!_node_scanned[root] && _cluster_locks.compare_and_set_to_true(cluster_id)) {
          vec<HypernodeID>& dfs_stack = local_stacks.local();
          dfs_stack.clear();
          vec<HypernodeID>& found_nodes = local_node_list.local();
          found_nodes.clear();
          dfs_stack.push_back(root);
          _node_scanned.set(root);
          while (dfs_stack.size() > 0) {
            const HypernodeID current_hn = dfs_stack.back();
            found_nodes.push_back(current_hn);
            dfs_stack.pop_back();
            for (HyperedgeID he: hg.incidentEdges(current_hn)) {
              for (HypernodeID pin: hg.pins(he)) {
                if (cc.clusterID(pin) == cluster_id && !_node_scanned[pin]) {
                  dfs_stack.push_back(pin);
                  _node_scanned.set(pin);
                }
              }
            }
          }
          _cluster_locks.set(cluster_id, false);
          for (const HypernodeID& current_hn: found_nodes) {
            cc.setClusterID(current_hn, root);
          }
          _cluster_sizes[hg.communityID(root)].fetch_sub(found_nodes.size(), std::memory_order_relaxed);
          _cluster_sizes[root].fetch_add(found_nodes.size(), std::memory_order_relaxed);
          if (found_nodes.size() == 1) {
            cc.makeVertexUnmatched(root);
          }
        }
      });
      tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
        const HypernodeID hn = _active_nodes[i];
        if (!_node_scanned[hn]) {
          local_active_nodes.stream(hn);
        }
      });
      _active_nodes.clear();
      local_active_nodes.copy_parallel(_active_nodes);
      local_active_nodes.clear_sequential();
      if (_active_nodes.size() > 0) {
        LOG << "continuing with" << _active_nodes.size() << "nodes ...";
      }
    }

    ASSERT([&] {
      for (HypernodeID hn: hg.nodes()) {
        HypernodeID cluster_id = cc.clusterID(hn);
        if (_cluster_sizes[cluster_id] == 0) {
          return false;
        }
        if (hn != cluster_id && _cluster_sizes[hn] > 0) {
          return false;
        }
      }
      return true;
    }());
    HypernodeID num_clusters = tbb::parallel_reduce(tbb::blocked_range<HypernodeID>(ID(0), hg.initialNumNodes()), 0,
                                      [&](const tbb::blocked_range<HypernodeID>& range, HypernodeID init) {
                                        HypernodeID count = init;
                                        for (HypernodeID hn = range.begin(); hn < range.end(); ++hn) {
                                            count += (_cluster_sizes[hn].load(std::memory_order_relaxed) > 0) ? 1 : 0;
                                        }
                                        return count;
                                      }, std::plus<>());
    cc.setNumberOfNodes(num_clusters);

    // restore parameter and community ids
    std::swap(cc.max_allowed_node_weight, tmp_max_allowed_node_weight);
    hg.doParallelForAllNodes([&](const HypernodeID hn) {
      hg.setCommunityID(hn, _community_ids[hn]);
    });
    _active_nodes.clear();
    _node_scanned.reset();
    _cluster_locks.reset();
  }

 private:
  template<typename Hypergraph>
  bool clusterShouldBeSplit(HypernodeID cluster_id, ClusteringContext<Hypergraph>& cc) {
    // TODO: tolerance factor (~2) for weight??
    return _cluster_sizes[cluster_id].load(std::memory_order_relaxed) >= _context.coarsening.splitting_min_num_nodes
           && cc.clusterWeight(cluster_id) > cc.max_allowed_node_weight * _context.coarsening.splitting_tolerance_factor;
  }

  const Context& _context;
  ds::Array<CAtomic<HypernodeID>> _cluster_sizes;
  vec<PartitionID> _community_ids;
  vec<HypernodeID> _active_nodes;
  kahypar::ds::FastResetFlagArray<> _node_scanned;
  ds::ThreadSafeFastResetFlagArray<> _cluster_locks;
};

}  // namespace mt_kahypar
