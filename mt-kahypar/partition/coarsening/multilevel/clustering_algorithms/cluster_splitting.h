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
    _active_nodes() {
      _old_cluster_sizes.assign(num_nodes, CAtomic<HypernodeID>(0));
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
    }, [&] {
      _active_nodes.reserve(hg.initialNumNodes());
    });

    // prepare active nodes
    ds::StreamingVector<HypernodeID> local_active_nodes;
    hg.doParallelForAllNodes([&](const HypernodeID hn) {
      const HypernodeID cluster_id = cc.clusterID(hn);
      hg.setCommunityID(hn, cluster_id);
      if (clusterShouldBeSplit(cluster_id, cc)) {
        cc.resetCluster(hg, hn);
        local_active_nodes.stream(hn);
      }
    });
    local_active_nodes.copy_parallel(_active_nodes);
    LOG << V(_active_nodes.size()) << V(hg.initialNumNodes());

    auto handle_node = [&](const HypernodeID hn) {
      ASSERT(hg.nodeIsEnabled(hn));
      const Rating rating = cc.template rate<ScorePolicy, HeavyNodePenaltyPolicy,
                                              AcceptancePolicy, has_fixed_vertices>(hg, hn, similarity_policy);
      if (rating.target != kInvalidHypernode) {
        cc.template joinClusterIgnoringMatchingState<has_fixed_vertices>(hg, hn, rating.target);
      }
    };

    // if (_context.coarsening.prioritize_high_degree) {
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

    //   auto task = [&]{
    //     while (true) {
    //       HypernodeID hn;
    //       bool success = parallel_pq.tryPop(hn);
    //       if (!success) {
    //         return;
    //       }
    //       handle_node(hn);
    //     }
    //   };

    //   tbb::task_group tg;
    //   for (size_t i = 0; i < _context.shared_memory.num_threads; ++i) { tg.run(task); }
    //   tg.wait();
    // } else {
    //   // We iterate in parallel over all vertices of the hypergraph and compute its contraction partner.
    //   tbb::parallel_for(ID(0), hg.initialNumNodes(), [&](const HypernodeID id) {
    //     ASSERT(id < node_mapping.size());
    //     handle_node(node_mapping[id]);
    //   });
    // }

    // restore parameter and community ids
    std::swap(cc.max_allowed_node_weight, tmp_max_allowed_node_weight);
    hg.doParallelForAllNodes([&](const HypernodeID hn) {
      hg.setCommunityID(hn, _community_ids[hn]);
    });
    _active_nodes.clear();
  }

 private:
  template<typename Hypergraph>
  bool clusterShouldBeSplit(HypernodeID cluster_id, ClusteringContext<Hypergraph>& cc) {
    return _old_cluster_sizes[cluster_id].load(std::memory_order_relaxed) >= _context.coarsening.splitting_min_num_nodes
           && cc.clusterWeight(cluster_id) > cc.max_allowed_node_weight;
  }

  const Context& _context;
  ds::Array<CAtomic<HypernodeID>> _old_cluster_sizes;
  vec<PartitionID> _community_ids;
  vec<HypernodeID> _active_nodes;
};

}  // namespace mt_kahypar
