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
    _old_cluster_sizes(num_nodes),
    _community_ids(),
    _active_nodes(),
    _cluster_mapping(),
    _node_scanned(),
    _cluster_scanned() {
      tbb::parallel_invoke([&] {
        _old_cluster_sizes.assign(num_nodes, CAtomic<HypernodeID>(0));
      }, [&] {
        _cluster_mapping.resize(num_nodes, CAtomic<HypernodeID>(kInvalidHypernode));
      }, [&] {
        _node_scanned.setSize(num_nodes);
      }, [&] {
        _cluster_scanned.setSize(num_nodes);
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
      _old_cluster_sizes.assign(hg.initialNumNodes(), CAtomic<HypernodeID>(0));
      hg.doParallelForAllNodes([&](const HypernodeID hn) {
        const HypernodeID cluster_id = cc.clusterID(hn);
        _old_cluster_sizes[cluster_id].fetch_add(1, std::memory_order_relaxed);
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
    // TODO: multiple rounds

    if (_context.coarsening.prioritize_high_degree) {
    //   PQ parallel_pq(_context.shared_memory.num_threads);
    //   hg.doParallelForAllNodes([&](const HypernodeID hn) {
    //     double rating = 0;
    //     for (const HyperedgeID& he : hg.incidentEdges(hn)) {
    //       rating += hg.edgeWeight(he);
    //     }
    //     if (_context.coarsening.prioritize_with_node_weight) {
    //       rating /= hg.nodeWeight(hn);
    //     }
    //     parallel_pq.insert(rating, hn);
    //   });
    } else {
      // We iterate in parallel over the active vertices and update their clusters
      tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
        handle_node(_active_nodes[i]);
      });
    }

    // update matching states and number of nodes
    while (_active_nodes.size() > 0) {
      tbb::enumerable_thread_specific<vec<HypernodeID>> local_queues;
      tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
        const HypernodeID hn = _active_nodes[i];
        const HypernodeID cluster_id = cc.clusterID(hn);
        if (!_node_scanned[hn] && _cluster_scanned.compare_and_set_to_true(cluster_id)) {
          _cluster_mapping[cluster_id].store(hn);
          HypernodeID expected = kInvalidHypernode;
          _cluster_mapping[hn].compare_exchange_strong(expected, cluster_id);  // ---> avoids collisions!!!
          // depth first search to collect the nodes of this cluster
          size_t cluster_size = 0;
          vec<HypernodeID>& queue = local_queues.local();
          queue.clear();
          queue.push_back(hn);
          _node_scanned.set(hn);
          while (queue.size() > 0) {
            cluster_size++;
            const HypernodeID current = queue.back();
            queue.pop_back();
            for (HyperedgeID he: hg.incidentEdges(current)) {
              for (HypernodeID pin: hg.pins(he)) {
                if (cc.clusterID(pin) == cluster_id && !_node_scanned[pin]) {
                  queue.push_back(pin);
                  _node_scanned.set(pin);
                }
              }
            }
          }
          _cluster_scanned.set(cluster_id, false);
          // TODO: size update
        }
      });
      tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
        const HypernodeID hn = _active_nodes[i];
        const HypernodeID cluster_id = cc.clusterID(hn);
        if (!_node_scanned[hn]) {
          local_active_nodes.stream(hn);
        } else if (_cluster_mapping[cluster_id] != kInvalidHypernode) {
          cc.setClusterID(hn, _cluster_mapping[cluster_id]);
        }
      });
      // reset mapping
      tbb::parallel_for(UL(0), _active_nodes.size(), [&](const size_t i) {
        const HypernodeID hn = _active_nodes[i];
        _cluster_mapping[cc.clusterID(hn)] = CAtomic<HypernodeID>(kInvalidHypernode);
      });
      _active_nodes.clear();
      local_active_nodes.copy_parallel(_active_nodes);
      local_active_nodes.clear_sequential();
      if (_active_nodes.size() > 0) {
        LOG << "continuing with" << _active_nodes.size() << "nodes ...";
      }
    }

    // restore parameter and community ids
    std::swap(cc.max_allowed_node_weight, tmp_max_allowed_node_weight);
    hg.doParallelForAllNodes([&](const HypernodeID hn) {
      hg.setCommunityID(hn, _community_ids[hn]);
      _cluster_mapping[hn] = CAtomic<HypernodeID>(kInvalidHypernode);
    });
    _active_nodes.clear();
    _node_scanned.reset();
    _cluster_scanned.reset();
    ASSERT([&] {
      for (const auto& val: _cluster_mapping) {
        if (val.load() != kInvalidHypernode) {
          return false;
        }
      }
      return true;
    }());
  }

 private:
  template<typename Hypergraph>
  bool clusterShouldBeSplit(HypernodeID cluster_id, ClusteringContext<Hypergraph>& cc) {
    // TODO: tolerance factor (~2) for weight??
    return _old_cluster_sizes[cluster_id].load(std::memory_order_relaxed) >= _context.coarsening.splitting_min_num_nodes
           && cc.clusterWeight(cluster_id) > cc.max_allowed_node_weight;
  }

  const Context& _context;
  ds::Array<CAtomic<HypernodeID>> _old_cluster_sizes;
  vec<PartitionID> _community_ids;
  vec<HypernodeID> _active_nodes;
  vec<CAtomic<HypernodeID>> _cluster_mapping;
  kahypar::ds::FastResetFlagArray<> _node_scanned;
  ds::ThreadSafeFastResetFlagArray<> _cluster_scanned;
};

}  // namespace mt_kahypar
