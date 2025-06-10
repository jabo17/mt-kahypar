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

#include <oneapi/tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {

static constexpr bool debug = true;
static constexpr bool enable_heavy_assert = true;

  template<typename TypeTraits>
  std::unique_ptr<kaminpar::shm::CSRGraph> ExperimentalCoarsener<TypeTraits>::buildBipartiteGraphRep() {
      using namespace kaminpar;
      using namespace kaminpar::shm;

      const Hypergraph& hg = Base::currentHypergraph();
      const HypernodeID num_nodes = hg.initialNumNodes();

      const NodeID n = num_nodes + hg.initialNumEdges();
      const EdgeID m = 2 * hg.initialNumPins();


      StaticArray<EdgeID> nodes(n + 1);
      StaticArray<NodeID> edges(m);
      StaticArray<NodeWeight> node_weights(n);
      StaticArray<EdgeWeight> edge_weights(m);

      // set node weights and node degrees
      nodes[0] = 0;
      tbb::parallel_invoke([&] {
                               tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
                                   NodeID u = _current_vertices[id];
                                   node_weights[u] = hg.nodeWeight(id);
                                   nodes[u + 1] = hg.nodeDegree(id);
                               });
                           }, [&]() {
                               tbb::parallel_for<NodeID>(num_nodes, n, [&](const NodeID u) {
                                   node_weights[u] = 0;
                                   nodes[u + 1] = hg.edgeSize(u - num_nodes);
                               });
                           });

      ASSERT(node_weights[_current_vertices[0]]==hg.nodeWeight(0));
      ASSERT(nodes[_current_vertices[0]+1] == hg.nodeDegree(0));
      ASSERT(node_weights[num_nodes] == 0);
      ASSERT(nodes[num_nodes+1] == hg.edgeSize(0));
      ASSERT(nodes[n] == hg.edgeSize(hg.initialNumEdges()-1));
      // compute offset of neighborhoods in edge array
      parallel_prefix_sum(nodes.begin()+1, nodes.end(), nodes.begin()+1, [&](EdgeID x, EdgeID y) { return x + y; }, 0);

      // obtain edge weight for edge of the graph
      auto graphEdgeWeight = [SCALE_EDGE_WEIGHT = static_cast<EdgeWeight>(hg.maxEdgeSize())*10, &hg, rep_edge_weight=_context.coarsening.rep_edge_weight](const HyperedgeID he) {
          if ( rep_edge_weight == GraphRepEdgeWeight::unit) {
            return 1;
          }
          EdgeWeight edge_weight = hg.edgeWeight(he);
          if ( rep_edge_weight == GraphRepEdgeWeight::normalized_hyperedge_weight ) {

            return (edge_weight * SCALE_EDGE_WEIGHT) / static_cast<EdgeWeight>(hg.edgeSize(he));
          }
          return edge_weight;
      };

      // set edges and edge weights
      tbb::parallel_invoke([&]() {
                               // neighborhoods representing incident nets
                               tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
                                   const NodeID u = _current_vertices[id];
                                   EdgeID pos = nodes[u];
                                   for (const HyperedgeID &he: hg.incidentEdges(id)) {
                                       ASSERT(hg.edgeSize(he) >= 2, "Empty or single nets encountered.");
                                       edges[pos] = he + num_nodes; // hyperedges are shifted by num_nodes
                                       edge_weights[pos] = graphEdgeWeight(he);
                                       ++pos;
                                   }
                               });
                           }, [&]() {
                               // neighborhoods representing pins
                               tbb::parallel_for<NodeID>(UL(0), hg.initialNumEdges(), [&](const NodeID he) {
                                   EdgeID pos = nodes[he+num_nodes];
                                   const EdgeWeight edge_weight = graphEdgeWeight(he);
                                   for (const HypernodeID &hv: hg.pins(he)) {
                                       edges[pos] = _current_vertices[hv]; // hypervertex ids remain identical
                                       edge_weights[pos] = edge_weight;
                                       ++pos;
                                   }
                               });
                           });

      constexpr bool neighborhood_sorted = false;
      return std::make_unique<kaminpar::shm::CSRGraph>(std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights),
                      neighborhood_sorted);
  }

  template<typename TypeTraits>
  std::unique_ptr<kaminpar::shm::CSRGraph> ExperimentalCoarsener<TypeTraits>::buildCycleMatchingRep() {
    using namespace kaminpar;
    using namespace kaminpar::shm;

    const Hypergraph& hg = Base::currentHypergraph();
    const HypernodeID num_nodes = hg.initialNumNodes();

    const NodeID n = num_nodes;
    const EdgeID m = 3 * hg.initialNumPins();

    StaticArray<EdgeID> nodes(n + 1);
    StaticArray<EdgeID> nodes_agg(n + 1);
    StaticArray<NodeID> edges(m);
    StaticArray<NodeID> edges_agg(m);
    StaticArray<NodeWeight> node_weights(n);
    StaticArray<EdgeWeight> edge_weights(m);
    StaticArray<EdgeWeight> edge_weights_agg(m);

    nodes[0] = 0;
    nodes_agg[0] = 0;
    tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
      NodeID u = _current_vertices[id];
      node_weights[u] = hg.nodeWeight(id);
      nodes[u + 1] = 3 * hg.nodeDegree(id);
    });
    parallel_prefix_sum(nodes.begin()+1, nodes.end(), nodes.begin()+1, [&](EdgeID x, EdgeID y) { return x + y; }, 0);

    auto countEdgesForExpansion = [](HyperedgeID edge_size) {
      ASSERT(edge_size >= 2);
      EdgeID edges_in_expansion = 1;
      if (edge_size == 3) {
        edges_in_expansion = 3;
      } else {
        edges_in_expansion = edge_size + (edge_size / 2);
      }
      return edges_in_expansion;
    };

    EdgeWeight scale_edge_weight = 1;
    if (_context.coarsening.rep_edge_weight == GraphRepEdgeWeight::normalized_hyperedge_weight) {
      EdgeID max_edges_in_expansion = countEdgesForExpansion(hg.maxEdgeSize());
      if (max_edges_in_expansion == 3) {
        scale_edge_weight = static_cast<EdgeWeight>(max_edges_in_expansion)*2;
      } else if (max_edges_in_expansion >= 4) {
        scale_edge_weight = static_cast<EdgeWeight>(max_edges_in_expansion)*10;
      }
    }

    auto graphEdgeWeight = [SCALE_EDGE_WEIGHT=scale_edge_weight, &hg, &countEdgesForExpansion, rep_edge_weight=_context.coarsening.rep_edge_weight](const HyperedgeID he) {
      if (rep_edge_weight == GraphRepEdgeWeight::unit) {
        return 1;
      }

      EdgeWeight edge_weight = hg.edgeWeight(he);
      EdgeWeight edge_size = hg.edgeSize(he);

      EdgeWeight edges_in_expansion = 1;
      if (rep_edge_weight == GraphRepEdgeWeight::normalized_hyperedge_weight) {
        edges_in_expansion = countEdgesForExpansion(edge_size);
      }

      return edge_weight * SCALE_EDGE_WEIGHT / edges_in_expansion;
    };


    tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
        const NodeID u = _current_vertices[id];
        EdgeID pos = nodes[u];

        // expand hyperedges
        for (const HyperedgeID &he: hg.incidentEdges(id)) {
            const HyperedgeID edge_size = hg.edgeSize(he);
            auto pins = hg.pins(he);
            const HypernodeID rank = std::distance(pins.begin(), std::find(pins.begin(), pins.end(), id));
            ASSERT(rank < edge_size);
            const EdgeWeight weight = graphEdgeWeight(he);

            // cycle edges
            ASSERT(*(pins.begin() + ((rank+1) % edge_size)) < num_nodes);
            edges[pos] = _current_vertices[*(pins.begin() + ((rank+1) % edge_size))];
            edge_weights[pos] = weight;
            ++pos;

            if (edge_size >= 3) {
              ASSERT(*(pins.begin() + ((rank+edge_size-1) % edge_size)) < num_nodes);
              edges[pos] = _current_vertices[*(pins.begin() + ((rank+edge_size-1) % edge_size))];
              edge_weights[pos] = weight;
              ++pos;
            }
            else if (edge_size >= 4) {
              if ((edge_size & 1) == 0 || rank+1 < edge_size) {
                // edge_size is even or rank < edge_size-1
                EdgeID edge_size_half = edge_size / 2;
                if (rank >= edge_size_half) {
                  ASSERT(*(pins.begin() + (rank-edge_size_half)) < num_nodes);
                  edges[pos] = _current_vertices[*(pins.begin() + (rank-edge_size_half))];
                }else {
                  ASSERT(*(pins.begin() + (rank+edge_size_half)) < num_nodes);
                  edges[pos] = _current_vertices[*(pins.begin() + (rank+edge_size_half))];
                }

                edge_weights[pos] = weight;
                ++pos;
              }
            }

            ASSERT(edge_size >= 2, "Empty or single nets encountered.");
        }
        ASSERT(pos <= nodes[u+1]);

        // sort neighborhood to filter duplicates
        std::iota(edges_agg.begin()+nodes[u], edges_agg.begin()+pos, nodes[u]);
        if (pos > nodes[u]) {
          ASSERT(*(edges_agg.begin()+nodes[u]) == nodes[u]);
          ASSERT(*(edges_agg.begin()+(pos-1)) == (pos-1));
        }

        std::sort(edges_agg.begin()+nodes[u], edges_agg.begin()+pos, [&](EdgeID first, EdgeID second){return edges[first] < edges[second];});

        // agg duplicates
        NodeID last_target = n;
        EdgeID new_pos = nodes[u];
        for (EdgeID i = nodes[u]; i < pos; ++i) {
            EdgeID edge = edges_agg[i];
            NodeID target = edges[edge];
            if (last_target != target) {
              ASSERT(last_target == n || last_target < target, "edges are not sorted by target");
              last_target = target;
              edges_agg[new_pos] = target;
              edge_weights_agg[new_pos] = edge_weights[edge];
              ++new_pos;
            }else {
              edge_weights_agg[new_pos-1] += edge_weights[edge];
            }
        }
        // compute node degree
        nodes_agg[u+1]=new_pos-nodes[u];

    });

    // determine positions of agg. neighborhood
    parallel_prefix_sum(nodes_agg.begin()+1, nodes_agg.end(), nodes_agg.begin()+1, [&](EdgeID x, EdgeID y) { return x + y; }, 0);

    using std::swap;
    // store non-flattened agg. neighborhood in edges
    swap(edges, edges_agg);
    swap(edge_weights, edge_weights_agg);

    // flatten agg. neighborhoods
    tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID u) {
      for (EdgeID i = nodes_agg[u]; i < nodes_agg[u+1]; ++i) {
          const EdgeID i_old = i - nodes_agg[u] + nodes[u];
          edges_agg[i] = edges[i_old];
          edge_weights_agg[i] = edge_weights[i_old];
      }
    });

    constexpr bool neighborhood_sorted = true;
    return std::make_unique<kaminpar::shm::CSRGraph>(std::move(nodes_agg), std::move(edges_agg), std::move(node_weights), std::move(edge_weights_agg),
              neighborhood_sorted);
  }

template<typename TypeTraits>
  std::unique_ptr<kaminpar::shm::CSRGraph> ExperimentalCoarsener<TypeTraits>::buildCycleRandomMatchingRep() {
    using namespace kaminpar;
    using namespace kaminpar::shm;


    const Hypergraph& hg = Base::currentHypergraph();
    const HypernodeID num_nodes = hg.initialNumNodes();
    const HypernodeID num_edges = hg.initialNumEdges();

    const NodeID n = num_nodes;
    const EdgeID m = 3 * hg.initialNumPins();

    StaticArray<EdgeID> nodes(n + 1);
    StaticArray<EdgeID> nodes_agg(n + 1);
    StaticArray<NodeID> edges(m);
    StaticArray<NodeID> edges_agg(m);
    StaticArray<NodeWeight> node_weights(n);
    StaticArray<EdgeWeight> edge_weights(m);
    StaticArray<EdgeWeight> edge_weights_agg(m);

    nodes[0] = 0;
    nodes_agg[0] = 0;
    tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
      NodeID u = _current_vertices[id];
      node_weights[u] = hg.nodeWeight(id);
      nodes[u + 1] = 3 * hg.nodeDegree(id);
    });
    parallel_prefix_sum(nodes.begin()+1, nodes.end(), nodes.begin()+1, [&](EdgeID x, EdgeID y) { return x + y; }, 0);

    auto countEdgesForExpansion = [](HyperedgeID edge_size) {
      ASSERT(edge_size >= 2);
      EdgeID edges_in_expansion = 1;
      if (edge_size == 3) {
        edges_in_expansion = 3;
      } else {
        edges_in_expansion = edge_size + (edge_size / 2);
      }
      return edges_in_expansion;
    };

    EdgeWeight scale_edge_weight = 1;
    if (_context.coarsening.rep_edge_weight == GraphRepEdgeWeight::normalized_hyperedge_weight) {
      EdgeID max_edges_in_expansion = countEdgesForExpansion(hg.maxEdgeSize());
      if (max_edges_in_expansion == 3) {
        scale_edge_weight = static_cast<EdgeWeight>(max_edges_in_expansion)*2;
      } else if (max_edges_in_expansion >= 4) {
        scale_edge_weight = static_cast<EdgeWeight>(max_edges_in_expansion)*10;
      }
    }

    auto graphEdgeWeight = [SCALE_EDGE_WEIGHT=scale_edge_weight, &hg, &countEdgesForExpansion, rep_edge_weight=_context.coarsening.rep_edge_weight](const HyperedgeID he) {
      if (rep_edge_weight == GraphRepEdgeWeight::unit) {
        return 1;
      }

      EdgeWeight edge_weight = hg.edgeWeight(he);
      EdgeWeight edge_size = hg.edgeSize(he);

      EdgeWeight edges_in_expansion = 1;
      if (rep_edge_weight == GraphRepEdgeWeight::normalized_hyperedge_weight) {
        edges_in_expansion = countEdgesForExpansion(edge_size);
      }

      return edge_weight * SCALE_EDGE_WEIGHT / edges_in_expansion;
    };

    const NodeID NO_EDGE = std::numeric_limits<NodeID>::max();
    tbb::parallel_for<NodeID>(UL(0), num_edges, [&](const EdgeID he) {
      auto pins = hg.pins(he);
      std::size_t edge_size = hg.edgeSize(he);

      auto edge_rank = [&](NodeID v) {
        return std::distance(hg.incidentEdges(v).begin(), std::find(hg.incidentEdges(v).begin(), hg.incidentEdges(v).end(), he));
      };


      const EdgeWeight weight = graphEdgeWeight(he);

      // based on D. Seemaier's implementation
      ASSERT(edge_size >= 2);
      if (edge_size == 2) {
        for (std::size_t current = 0; current < edge_size; ++current) {
          std::size_t next = (current == 0) ? edge_size - 1 : current - 1;
          
          std::size_t edge_displ = nodes[_current_vertices[*(pins.begin() + current)]] + edge_rank(*(pins.begin() + current)) * 3;
          ASSERT(edge_displ+2 < edges.size());
          edges[edge_displ] = _current_vertices[*(pins.begin() + next)];
          edges[edge_displ+1] = NO_EDGE;
          edges[edge_displ+2] = NO_EDGE;
          edge_weights[edge_displ] = weight;
          edge_weights[edge_displ+1] = 0;
          edge_weights[edge_displ+2] = 0;
        }

        return;
      }
      if (edge_size == 3) {


        for (std::size_t current = 0; current < edge_size; ++current) {
          std::size_t next = (current + 1) % edge_size;
          std::size_t prev = (current == 0) ? edge_size - 1 : current - 1;
          
          std::size_t edge_displ = nodes[_current_vertices[*(pins.begin() + current)]] + edge_rank(*(pins.begin() + current)) * 3;
          ASSERT(edge_displ+2 < edges.size());

          edges[edge_displ] = _current_vertices[*(pins.begin() + next)];
          edges[edge_displ+1] = _current_vertices[*(pins.begin() + prev)];
          edges[edge_displ+2] = NO_EDGE;
          edge_weights[edge_displ] = weight;
          edge_weights[edge_displ+1] = weight;
          edge_weights[edge_displ+2] = 0;
        }

        return;
      }

      std::vector<HypernodeID> open(edge_size);
      std::iota(open.begin(), open.end(), 0);
      std::mt19937 g(_context.partition.seed + he + _pass_nr);
      std::shuffle(open.begin(), open.end(), g);
      std::vector<bool> matched(edge_size);
      std::vector<HypernodeID> matched_pins(edge_size);

      for (std::size_t current = 0; current < edge_size; ++current) {
        std::size_t next = (current + 1) % edge_size;
        std::size_t prev = (current == 0 ? edge_size - 1 : current - 1);

        if (!matched[current]) {
          std::size_t matched_pin = edge_size;
          do {
            std::size_t i = 0;
            while (i < open.size() &&
                  (open[i] == current || open[i] == prev || open[i] == next)) {
              ++i;
            }
            if (i < open.size()) {
              matched_pin = open[i];
              std::swap(open[i], open.back());
              open.pop_back();
            } else {
              break;
            }
          } while (matched[matched_pin]);

          if (matched_pin != edge_size) {
            matched[current] = true;
            matched[matched_pin] = true;
            matched_pins[current] = matched_pin;
            matched_pins[matched_pin] = current;
          }
        }

        std::size_t edge_displ = nodes[_current_vertices[*(pins.begin() + current)]] + edge_rank(*(pins.begin() + current)) * 3;
        ASSERT(edge_displ+2 < edges.size());
        edges[edge_displ] = _current_vertices[*(pins.begin() + next)];
        edges[edge_displ+1] = _current_vertices[*(pins.begin() + prev)];
        edge_weights[edge_displ] = weight;
        edge_weights[edge_displ+1] = weight;
        if(matched[current]){
          edges[edge_displ+2] =_current_vertices[*(pins.begin() + matched_pins[current])];
          edge_weights[edge_displ+2] = weight;
        }else{
          // unmatched
          edges[edge_displ+2] = NO_EDGE; 
          edge_weights[edge_displ+2] = 0;
        }
      }
    });


    tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
        const NodeID u = _current_vertices[id];

        // sort neighborhood to filter duplicates
        std::iota(edges_agg.begin()+nodes[u], edges_agg.begin()+nodes[u+1], nodes[u]);

        auto less = [&](EdgeID first, EdgeID second){return edges[first] < edges[second];};
        std::sort(edges_agg.begin()+nodes[u], edges_agg.begin()+nodes[u+1], less);
        
        // agg duplicates
        NodeID last_target = n;
        EdgeID new_pos = nodes[u];
        for (EdgeID i = nodes[u]; i < nodes[u+1]; ++i) {
            EdgeID edge = edges_agg[i];
            NodeID target = edges[edge];
            if(target == NO_EDGE){
              break; // finished
            }

            if (last_target != target) {
              ASSERT(last_target == n || last_target < target, "edges are not sorted by target");
              last_target = target;
              edges_agg[new_pos] = target;
              edge_weights_agg[new_pos] = edge_weights[edge];
              ++new_pos;
            }else {
              edge_weights_agg[new_pos-1] += edge_weights[edge];
            }
        }
        // compute node degree
        nodes_agg[u+1]=new_pos-nodes[u];

    });

    // determine positions of agg. neighborhood
    parallel_prefix_sum(nodes_agg.begin()+1, nodes_agg.end(), nodes_agg.begin()+1, [&](EdgeID x, EdgeID y) { return x + y; }, 0);

    using std::swap;
    // store non-flattened agg. neighborhood in edges
    swap(edges, edges_agg);
    swap(edge_weights, edge_weights_agg);

    // flatten agg. neighborhoods
    tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID u) {
      for (EdgeID i = nodes_agg[u]; i < nodes_agg[u+1]; ++i) {
          const EdgeID i_old = i - nodes_agg[u] + nodes[u];
          edges_agg[i] = edges[i_old];
          edge_weights_agg[i] = edge_weights[i_old];
      }
    });

    constexpr bool neighborhood_sorted = true;
    return std::make_unique<kaminpar::shm::CSRGraph>(std::move(nodes_agg), std::move(edges_agg), std::move(node_weights), std::move(edge_weights_agg),
              neighborhood_sorted);
  }


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
  tbb::parallel_for<HypernodeID>(UL(0), num_nodes, [&](const HypernodeID u) {
      ASSERT(hg.nodeIsEnabled(u));
  });
 
  const double num_nodes_before_pass = num_nodes;
  vec<HypernodeID> clusters(num_nodes, kInvalidHypernode);
  _current_vertices.resize(num_nodes);
  tbb::parallel_for(UL(0), num_nodes, [&](HypernodeID u) {
    // cluster_weight[u] = hg.nodeWeight(u);
    clusters[u] = u;
    _current_vertices[u] = u;
  });

  //DisableRandomization();
  if ( _enable_randomization ) {
    utils::Randomize::instance().parallelShuffleVector( _current_vertices, UL(0), _current_vertices.size());
  }

  // START implementation of actual coarsening

  // build graph representation
  // all graph representation have in common that hypervertices have identical IDs in representation
  kaminpar::shm::Graph graph([&]() {
    switch (_context.coarsening.rep) {
      case GraphRepresentation::bipartite:
        DBG << "Build bipartite rep"; 
        return buildBipartiteGraphRep();
      case GraphRepresentation::cycle_matching:
        return buildCycleMatchingRep();
      case GraphRepresentation::cycle_random_matching:
        return buildCycleRandomMatchingRep();
      case GraphRepresentation::UNDEFINED:
        throw std::runtime_error("Undefined representation");
    }
    return std::unique_ptr<kaminpar::shm::CSRGraph>(nullptr);
  }());
  if (_context.coarsening.lp_sort) {
    graph = kaminpar::shm::graph::rearrange_by_degree_buckets(graph.csr_graph());
  }


  DBG << "Starting KaMinPar"; 

  // configure LPClustering
  auto ctx = kaminpar::shm::create_default_context();
  ctx.parallel.num_threads = _context.shared_memory.num_threads;
  ctx.partition.setup(graph, _context.partition.k, _context.partition.epsilon);
  ctx.coarsening.clustering.lp.num_iterations = _context.coarsening.lp_iterations;
  kaminpar::Random::reseed(_context.partition.seed + _pass_nr);

  // initialize and set config for LPClustering
  kaminpar::shm::LPClustering lp_clustering(ctx.coarsening);
  lp_clustering.set_max_cluster_weight(kaminpar::shm::compute_max_cluster_weight<kaminpar::shm::NodeWeight>(
    ctx.coarsening, ctx.partition, graph.n(), graph.total_node_weight()));
  lp_clustering.set_desired_cluster_count(0);

  kaminpar::StaticArray<kaminpar::shm::NodeID> graph_clustering(graph.n());
  kaminpar::StaticArray<kaminpar::shm::NodeID> remap_clusters(graph.n());
  lp_clustering.compute_clustering(graph_clustering, graph, false);


  DBG << "Copying Cluster Labels"; 

  if (_context.coarsening.lp_sort) {
    auto perm = graph.csr_graph().take_raw_permutation();
    // remap cluster labels to hypervertices as representatives
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
        const HypernodeID u = perm[_current_vertices[id]];
        const kaminpar::shm::NodeID root_u = graph_clustering[u];
        remap_clusters[root_u] = id;
    });

    // set cluster
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
          const HypernodeID u = perm[_current_vertices[id]];
          const kaminpar::shm::NodeID root_u = graph_clustering[u];
          ASSERT(remap_clusters[root_u] < num_nodes);
          clusters[id] = remap_clusters[root_u];
    });
  }else {
    // remap cluster labels to hypervertices as representatives
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
        const HypernodeID u = _current_vertices[id];
        const kaminpar::shm::NodeID root_u = graph_clustering[u];
        remap_clusters[root_u] = id;
    });

    // set cluster
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
          const HypernodeID u = _current_vertices[id];
          const kaminpar::shm::NodeID root_u = graph_clustering[u];
          ASSERT(remap_clusters[root_u] < num_nodes);
          clusters[id] = remap_clusters[root_u];
    });
  }

  // reduce number of cluster containing hypervertices
  num_nodes = tbb::parallel_reduce(tbb::blocked_range<HypernodeID>(UL(0), num_nodes), 0,
    [&](const tbb::blocked_range<HypernodeID>& range, HypernodeID init) -> HypernodeID {
    for (HypernodeID i = range.begin(); i != range.end(); ++i) {
      init += clusters[i] == i;
    }
    return init;
  }, std::plus<HypernodeID>());


  DBG << "Finished Coarsening step"; 

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

    HypernodeID num_clusters = 0;
    for ( const HypernodeID& hn : hg.nodes()) {
      const HypernodeID u = hn;
      const HypernodeID root_u = clusters[u];
      if (root_u != clusters[root_u]) {
        LOG << "Vertex " << u << " is in cluster with id " << root_u << " but " << root_u << " is not root of its own cluster.";
        return false;
      }
      if (clusters[root_u] >= num_nodes_before_pass) {
         LOG << "Vertex " << u << " is in cluster with id " << root_u << " but the cluster id " << root_u << " is not in range {0,..," << num_nodes << "}.";
      }
      if(clusters[u] == u) {
        ++num_clusters;
      }
    }

    if(num_clusters != num_nodes) {
      LOG << "Checked number of cluster (" << num_clusters << ") does not match new number of nodes (" << num_nodes << ")";
      return false;
    }


    return true;
  }(), "Clustering computed invalid cluster ids and weights");

  timer.stop_timer("coarsening_pass");
  ++_pass_nr;
  DBG << V(num_nodes_before_pass / num_nodes);
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
