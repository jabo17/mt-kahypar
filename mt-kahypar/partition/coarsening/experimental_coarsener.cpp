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

#include "experimental_coarsener.h"

#include <tbb/parallel_reduce.h>

#include "kaminpar-common/random.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {

static constexpr bool debug = true;
static constexpr bool enable_heavy_assert = true;

template<typename TypeTraits>
bool ExperimentalCoarsener<TypeTraits>::coarseningPassImpl() {
  auto& timer = utils::Utilities::instance().getTimer(_context.utility_id);
  const auto pass_start_time = std::chrono::high_resolution_clock::now();
  timer.start_timer("coarsening_pass", "Clustering");

  // first, initialize the cluster ids
  const Hypergraph& hg = Base::currentHypergraph();
  DBG << V(_pass_nr)
      << V(hg.initialNumNodes())
      << V(hg.initialNumEdges())
      << V(hg.initialNumPins());

  size_t num_nodes = Base::currentNumNodes();
  const double num_nodes_before_pass = num_nodes;
  vec<HypernodeID> clusters(num_nodes, kInvalidHypernode);
  _current_vertices.resize(hg.initialNumNodes());
  tbb::parallel_for(UL(0), num_nodes, [&](HypernodeID u) {
    // cluster_weight[u] = hg.nodeWeight(u);
    clusters[u] = u;
    _current_vertices[u] = u;
  });

  DisableRandomization();
  if ( _enable_randomization ) {
    utils::Randomize::instance().parallelShuffleVector( _current_vertices, UL(0), _current_vertices.size());
  }

  // START implementation of actual coarsening

  // build graph representation
  // all graph representation have in common that hypervertices have identical IDs in representation
  const kaminpar::shm::Graph graph(buildBipartiteGraphRep());

  // apply LPClustering (from KaMinPar)
  auto ctx = kaminpar::shm::create_default_context();
  ctx.parallel.num_threads = _context.shared_memory.num_threads;
  ctx.partition.setup(graph, _context.partition.k, _context.partition.epsilon);

  kaminpar::Random::reseed(_context.partition.seed);
  kaminpar::shm::LPClustering lp_clustering(ctx.coarsening);
  ctx.coarsening.clustering.lp.num_iterations = _context.coarsening.lp_iterations;
  //ctx.coarsening.clustering.shrink_factor = _context.coarsening.minimum_shrink_factor;

  lp_clustering.set_max_cluster_weight(kaminpar::shm::compute_max_cluster_weight<kaminpar::shm::NodeWeight>(
    ctx.coarsening, ctx.partition, graph.n(), graph.total_node_weight()));
  lp_clustering.set_desired_cluster_count(0);

  kaminpar::StaticArray<kaminpar::shm::NodeID> graph_clustering(graph.n());
  kaminpar::StaticArray<kaminpar::shm::NodeID> remap_clusters(graph.n());
  lp_clustering.compute_clustering(graph_clustering, graph, false);

  // remap cluster labels to hypervertices as representatives
  tbb::parallel_for(UL(0), num_nodes, [&](HypernodeID id) {
      const HypernodeID u = _current_vertices[id];
      const kaminpar::shm::NodeID root_u = graph_clustering[u];
      remap_clusters[root_u] = id;
  });

  // set cluster
  tbb::parallel_for(UL(0), num_nodes, [&](HypernodeID id) {
        const HypernodeID u = _current_vertices[id];
        const kaminpar::shm::NodeID root_u = graph_clustering[u];
        clusters[id] = remap_clusters[root_u];
  });

  // reduce number of cluster containing hypervertices
  num_nodes = tbb::parallel_reduce(tbb::blocked_range<HypernodeID>(UL(0), num_nodes), 0,
    [&](const tbb::blocked_range<HypernodeID>& range, HypernodeID init) -> HypernodeID {
    for (HypernodeID i = range.begin(); i != range.end(); ++i) {
      init += clusters[i] == i;
    }
    return init;
  }, std::plus<>());

  // END implementation of actual coarsening

  // Check clustering
  HEAVY_COARSENING_ASSERT([&] {
    /*parallel::scalable_vector<HypernodeWeight> expected_weights(hg.initialNumNodes());
    // Verify that clustering is correct
    for ( const HypernodeID& hn : hg.nodes() ) {
      const HypernodeID u = hn;
      const HypernodeID root_u = clusters[u];
      expected_weights[root_u] += hg.nodeWeight(hn);
    }

    // Verify that cluster weights are aggregated correct
    for ( const HypernodeID& hn : hg.nodes() ) {
      const HypernodeID u = hn;
      const HypernodeID root_u = clusters[u];
      if ( root_u == u && expected_weights[u] != _cluster_weight[u] ) {
        LOG << "The expected weight of cluster" << u << "is" << expected_weights[u]
            << ", but currently it is" << _cluster_weight[u];
        return false;
      }
    }*/

    for ( const HypernodeID& hn : hg.nodes()) {
      const HypernodeID u = hn;
      const HypernodeID root_u = clusters[u];
      if (root_u != clusters[root_u]) {
        LOG << "Vertex " << u << " is in cluster with id " << root_u << " but " << root_u << " is not root of its own cluster.";
        return false;
      }
    }

    return true;
  }(), "Clustering computed invalid cluster ids and weights");

  timer.stop_timer("coarsening_pass");
  ++_pass_nr;
  if (num_nodes_before_pass / num_nodes <= _context.coarsening.minimum_shrink_factor) {
    return false;
  }

  _timer.start_timer("contraction", "Contraction");
  // at this point, the coarsening is finished and we use the final cluster ids to perform the contraction
  _uncoarseningData.performMultilevelContraction(std::move(clusters), false /* deterministic */, pass_start_time);
  _timer.stop_timer("contraction");
  return true;
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(ExperimentalCoarsener)

}
