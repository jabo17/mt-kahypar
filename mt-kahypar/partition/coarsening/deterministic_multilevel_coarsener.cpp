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

#include "deterministic_multilevel_coarsener.h"

#include <tbb/parallel_sort.h>

#include "mt-kahypar/definitions.h"

namespace mt_kahypar {

template<typename TypeTraits>
bool DeterministicMultilevelCoarsener<TypeTraits>::coarseningPassImpl() {
  auto& timer = utils::Utilities::instance().getTimer(_context.utility_id);
  const auto pass_start_time = std::chrono::high_resolution_clock::now();
  timer.start_timer("coarsening_pass", "Clustering");

  const Hypergraph& hg = Base::currentHypergraph();
  HypernodeID num_nodes = Base::currentNumNodes();
  const double num_nodes_before_pass = num_nodes;
  vec<HypernodeID> clusters(num_nodes, kInvalidHypernode);
  tbb::parallel_for(HypernodeID(0), num_nodes, [&](HypernodeID u) {
    cluster_weight[u] = hg.nodeWeight(u);
    opportunistic_cluster_weight[u] = cluster_weight[u];
    propositions[u] = u;
    clusters[u] = u;
  });

  if (_context.coarsening.use_adaptive_edge_size) {
    hyperedge_size.resize(hg.initialNumEdges());
    tbb::parallel_for(HyperedgeID(0), hg.initialNumEdges(), [&](HyperedgeID e) {
      hyperedge_size[e] = hg.edgeSize(e);
    });
  }

  permutation.shuffle(utils::IntegerRange<HypernodeID>{0, num_nodes}, _context.shared_memory.static_balancing_work_packages, config.prng); // need shuffle for prefix-doubling
  round_seed = config.prng();

  // TODO add these to the cli 
  constexpr size_t num_sequential_steps = 1000;
  constexpr double growth_factor = 1.8;
  constexpr double max_subround_size_fraction = 0.02;
  const size_t max_subround_size = std::max<size_t>(1, num_nodes_before_pass * max_subround_size_fraction);

  size_t last;
  for (size_t first = 0; num_nodes > currentLevelContractionLimit() && first < num_nodes_before_pass; first = last) {
    if (first < num_sequential_steps) {
      last = first + 1;
    } else {
      size_t dist = std::max<size_t>(growth_factor * (last - first), max_subround_size);
      last = std::min<size_t>(num_nodes_before_pass, first + dist);
    }
    // each vertex finds a cluster it wants to join
    tbb::parallel_for(first, last, [&](size_t pos) {
      const HypernodeID u = permutation.at(pos);
      if (cluster_weight[u] == hg.nodeWeight(u) && hg.nodeIsEnabled(u)) {
        calculatePreferredTargetCluster(u, clusters);
      }
    });

    tbb::enumerable_thread_specific<size_t> num_contracted_nodes { 0 };

    // already approve if we can grant all requests for proposed cluster
    // otherwise insert to shared vector so that we can group vertices by cluster
    tbb::parallel_for(first, last, [&](size_t pos) {
      HypernodeID u = permutation.at(pos);
      HypernodeID target = propositions[u];
      if (target != u) {
        if (opportunistic_cluster_weight[target] <= _context.coarsening.max_allowed_node_weight) {
          // if other nodes joined cluster u but u itself leaves for a different cluster, it doesn't count
          if (opportunistic_cluster_weight[u] == hg.nodeWeight(u)) {
            num_contracted_nodes.local() += 1;
          }
          clusters[u] = target;
          cluster_weight[target] = opportunistic_cluster_weight[target];
        } else {
          nodes_in_too_heavy_clusters.push_back_buffered(u);
        }
      }
    });

    num_nodes -= num_contracted_nodes.combine(std::plus<>());
    nodes_in_too_heavy_clusters.finalize();

    if (nodes_in_too_heavy_clusters.size() > 0) {
      num_nodes -= approveVerticesInTooHeavyClusters(clusters);
    }

    if (_context.coarsening.use_adaptive_edge_size) {
      // update hyperedge sizes
      tbb::parallel_for(first, last, [&](size_t pos) {
        HypernodeID u = permutation.at(pos);
        if (u == propositions[u] || u == clusters[u]) {
          return;
        }

        // another idea to speed this up. this is slow if degree(clusters[u]) is unnecessarily large --> can mark smaller?
        // mark hg.incidentEdges(clusters[u]) in bitset
        // for each e in hg.incidentEdges(u)
        //     if e is marked --> reduce its size by 1
        // this is assuming that the vertex clusters[u] is still in that cluster. If it left in this subround, the number of edge size reductions is reduced by 1

        auto& ratings = default_rating_maps.local();
        for (HyperedgeID he : hg.incidentEdges(u)) {
          // this could be optimized to run once per affected hyperedge
          if (hg.edgeSize(he) >= _context.partition.ignore_hyperedge_size_threshold) continue;
          ratings.clear();
          for (HypernodeID v : hg.pins(he)) {
            ratings[clusters[v]] += 1;
          }
          hyperedge_size[he] = ratings.size();  // benign race
        }
      });
    }
    
    nodes_in_too_heavy_clusters.clear();
  }

  timer.stop_timer("coarsening_pass");
  ++pass;
  if (num_nodes_before_pass / num_nodes <= _context.coarsening.minimum_shrink_factor) {
    return false;
  }
  _timer.start_timer("contraction", "Contraction");
  _uncoarseningData.performMultilevelContraction(std::move(clusters), true /* deterministic */, pass_start_time);
  _timer.stop_timer("contraction");
  return true;
}

template<typename TypeTraits>
void DeterministicMultilevelCoarsener<TypeTraits>::calculatePreferredTargetCluster(HypernodeID u, const vec<HypernodeID>& clusters) {
  const Hypergraph& hg = Base::currentHypergraph();
  auto& ratings = default_rating_maps.local();
  ratings.clear();
  auto& bloom_filter = bloom_filters.local();

  // calculate ratings
  for (HyperedgeID he : hg.incidentEdges(u)) {
    HypernodeID he_size = hg.edgeSize(he);
    if (he_size < _context.partition.ignore_hyperedge_size_threshold) {
      he_size = _context.coarsening.use_adaptive_edge_size ? hyperedge_size[he] : he_size;
      double he_score = static_cast<double>(hg.edgeWeight(he)) / he_size;
      for (HypernodeID v : hg.pins(he)) {
        const HypernodeID target = clusters[v];
        const HypernodeID bloom_rep = target & bloom_filter_mask;
        if (!bloom_filter[bloom_rep]) {
          ratings[target] += he_score;
          bloom_filter.set(bloom_rep, true);
        }
      }
      bloom_filter.reset();
    }
  }
  
  // find highest rated, feasible cluster
  const PartitionID comm_u = hg.communityID(u);
  const HypernodeWeight weight_u = hg.nodeWeight(u);
  vec<HypernodeID>& best_targets = ties.local();
  double best_score = 0.0;

  for (const auto& entry : ratings) {
    HypernodeID target_cluster = entry.key;
    double target_score = entry.value;
    if (target_score >= best_score && target_cluster != u && hg.communityID(target_cluster) == comm_u
        && cluster_weight[target_cluster] + weight_u <= _context.coarsening.max_allowed_node_weight) {
      if (target_score > best_score) {
        best_targets.clear();
        best_score = target_score;
      }
      best_targets.push_back(target_cluster);
    }
  }

  HypernodeID best_target;
  if (best_targets.size() == 1) {
    best_target = best_targets[0];
  } else if (best_targets.empty()) {
    best_target = u;
  } else {
    // TODO is this actually slow? Or do I just think it'll be slow
    std::mt19937 prng(u + round_seed);
    // This isn't quite it. Let k = best_targets.size()
    // How about x ~ uniform(1, 2^{k} - 1). then take pos = k - 1 - log_2(x). P(pos = 0) = 0.5, P(pos = 1) = 0.25, ...
    size_t pos = std::geometric_distribution<>(0.5)(prng);
    if (pos >= best_targets.size()) { 
      pos = 0;
    }
    assert(pos < best_targets.size());
    best_target = best_targets[pos];
  }
  best_targets.clear();

  if (best_target != u) {
    propositions[u] = best_target;
    __atomic_fetch_add(&opportunistic_cluster_weight[best_target], hg.nodeWeight(u), __ATOMIC_RELAXED);
  }
}

template<typename TypeTraits>
size_t DeterministicMultilevelCoarsener<TypeTraits>::approveVerticesInTooHeavyClusters(vec<HypernodeID>& clusters) {
  const Hypergraph& hg = Base::currentHypergraph();
  tbb::enumerable_thread_specific<size_t> num_contracted_nodes { 0 };

  // group vertices by desired cluster, if their cluster is too heavy. approve the lower weight nodes first
  auto comp = [&](HypernodeID lhs, HypernodeID rhs) {
    HypernodeWeight wl = hg.nodeWeight(lhs), wr = hg.nodeWeight(rhs);
    return std::tie(propositions[lhs], wl, lhs) < std::tie(propositions[rhs], wr, rhs);
  };
  tbb::parallel_sort(nodes_in_too_heavy_clusters.begin(), nodes_in_too_heavy_clusters.end(), comp);

  tbb::parallel_for(UL(0), nodes_in_too_heavy_clusters.size(), [&](size_t pos) {
    HypernodeID target = propositions[nodes_in_too_heavy_clusters[pos]];
    // the first vertex for this cluster handles the approval
    size_t num_contracted_local = 0;
    if (pos == 0 || propositions[nodes_in_too_heavy_clusters[pos - 1]] != target) {
      HypernodeWeight target_weight = cluster_weight[target];
      size_t first_rejected = pos;
      // could be parallelized without extra memory but factor 2 work overhead and log(n) depth via binary search
      for (; ; ++first_rejected) {
        // we know that this cluster is too heavy, so the loop will terminate before
        assert(first_rejected < nodes_in_too_heavy_clusters.size());
        assert(propositions[nodes_in_too_heavy_clusters[first_rejected]] == target);
        HypernodeID v = nodes_in_too_heavy_clusters[first_rejected];
        if (target_weight + hg.nodeWeight(v) > _context.coarsening.max_allowed_node_weight) {
          break;
        }
        clusters[v] = target;
        target_weight += hg.nodeWeight(v);
        if (opportunistic_cluster_weight[v] == hg.nodeWeight(v)) {
          num_contracted_local += 1;
        }
      }
      cluster_weight[target] = target_weight;
      opportunistic_cluster_weight[target] = target_weight;
      num_contracted_nodes.local() += num_contracted_local;
    }
  });

  return num_contracted_nodes.combine(std::plus<>());
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(DeterministicMultilevelCoarsener)

}
