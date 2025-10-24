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
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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

#include <algorithm>
#include <oneapi/tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/macros.h"

namespace mt_kahypar {

static constexpr bool debug = true;
static constexpr bool enable_heavy_assert = true;

// some helpers for constructing the different graph models

template <typename TypeTraits>
std::unique_ptr<kaminpar::shm::CSRGraph>
ExperimentalCoarsener<TypeTraits>::construct_graph_model_from_buffers(
    const parallel::scalable_vector<kaminpar::shm::NodeID> &nodes_buf,
    const parallel::scalable_vector<kaminpar::shm::EdgeID> &edges_buf,
    const parallel::scalable_vector<double> &edge_weights_buf,
    const bool neighborhood_sorted) {
  using namespace kaminpar;
  using namespace kaminpar::shm;

  const auto &hg = Base::currentHypergraph();
  ASSERT(nodes_buf.size() >= 1, "Node buffer must contain at least one entry.");
  const NodeID n = nodes_buf.size() - 1;

  // buffers for graph model
  StaticArray<EdgeID> nodes(n + 1);
  StaticArray<NodeWeight> node_weights(n);
  StaticArray<NodeID> edges(nodes_buf[n]);
  StaticArray<EdgeWeight> edge_weights(nodes_buf[n]);

  // fill buffers of graph model
  tbb::parallel_for<NodeID>(
      UL(0), n + 1, [&](const NodeID id) { nodes[id] = nodes_buf[id]; });

  tbb::parallel_for<NodeID>(UL(0), edges.size(),
                            [&](const EdgeID e) { edges[e] = edges_buf[e]; });

  // convert edge weights to edge_weights
  tbb::parallel_for<NodeID>(UL(0), edge_weights.size(), [&](const EdgeID e) {
    edge_weights[e] = toEdgeWeight(edge_weights_buf[e]);
  });

  tbb::parallel_for<NodeID>(UL(0), n, [&](const NodeID id) {
    NodeID u = _current_vertices[id];
    node_weights[u] = hg.nodeWeight(id);
  });

  return std::make_unique<CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights),
      std::move(edge_weights), neighborhood_sorted);
}

template <typename TypeTraits>
void ExperimentalCoarsener<TypeTraits>::defragment_neighborhoods(
    const parallel::scalable_vector<kaminpar::shm::EdgeID> &nodes_old,
    const parallel::scalable_vector<kaminpar::shm::NodeID> &edges_old,
    const parallel::scalable_vector<double> &edge_weights_old,
    const parallel::scalable_vector<kaminpar::shm::EdgeID> &nodes_new,
    parallel::scalable_vector<kaminpar::shm::NodeID> &edges_new,
    parallel::scalable_vector<double> &edge_weights_new) {
  using namespace kaminpar;
  using namespace kaminpar::shm;
  ASSERT(nodes_old.size() == nodes_new.size(),
         "Node array sizes do not match.");
  tbb::parallel_for<NodeID>(UL(0), nodes_new.size() - 1, [&](const NodeID u) {
    const EdgeID begin = nodes_new[u];
    const EdgeID end = nodes_new[u + 1];
    EdgeID read_pos = nodes_old[u];
    for (EdgeID i = begin; i < end; ++i) {
      ASSERT(edges_old[read_pos] < nodes_old.size() - 1,
             V(edges_old[read_pos]));
      edges_new[i] = edges_old[read_pos];
      edge_weights_new[i] = edge_weights_old[read_pos];
      ++read_pos;
    }
  });
}

template <typename TypeTraits>
kaminpar::shm::EdgeID
ExperimentalCoarsener<TypeTraits>::merge_multiedges_in_neighborhood(
    std::size_t from, std::size_t to,
    const parallel::scalable_vector<kaminpar::shm::NodeID> &edges,
    const parallel::scalable_vector<double> &edge_weights,
    parallel::scalable_vector<kaminpar::shm::NodeID> &merged_edges,
    parallel::scalable_vector<double> &merged_edge_weights) {
  using namespace kaminpar;
  using namespace kaminpar::shm;
  // use merged_edges[from..to] as temporary buffer
  // compute permutation so that edges are sorted
  std::iota(merged_edges.begin() + from, merged_edges.begin() + to, from);
  std::sort(merged_edges.begin() + from, merged_edges.begin() + to,
            [&](EdgeID l, EdgeID r) {
              // sort by target node, then by edge weight
              // this way, sums of doubles are deterministic
              return edges[l] < edges[r] || (edges[l] == edges[r] &&
                                             edge_weights[l] < edge_weights[r]);
            });

  // merge multi edges
  EdgeID i = from;
  std::size_t current = i;
  while (i < to) {
    ASSERT(i >= current);
    const EdgeID edge_id = merged_edges[i];
    const NodeID target = edges[edge_id];
    if (target == kNoEdge) {
      // after kNoEdges do not follow any other valid edge
      static_assert(kNoEdge ==
                    std::numeric_limits<kaminpar::shm::NodeID>::max());
      break;
    }
    merged_edge_weights[current] = edge_weights[edge_id];
    merged_edges[current] = target;

    // while there are edges with the same target
    ++i;
    for (; i < to; ++i) {
      const auto next_edge_id = merged_edges[i];
      if (target != edges[next_edge_id]) {
        break;
      }
      // same target -> merge
      merged_edge_weights[current] += edge_weights[next_edge_id];
    }
    ++current;
  }
  return current - from;
}

template <typename TypeTraits>
std::unique_ptr<kaminpar::shm::CSRGraph>
ExperimentalCoarsener<TypeTraits>::buildBipartiteGraphRep() {
  using namespace kaminpar;
  using namespace kaminpar::shm;

  const Hypergraph &hg = Base::currentHypergraph();
  const HypernodeID num_nodes = hg.initialNumNodes();

  // TODO ignore disabled nodes ?

  const NodeID n = num_nodes + hg.initialNumEdges();
  const EdgeID m = 2 * hg.initialNumPins();

  StaticArray<EdgeID> nodes(n + 1);
  StaticArray<NodeID> edges(m);
  StaticArray<NodeWeight> node_weights(n);
  StaticArray<EdgeWeight> edge_weights(m);

  // set node weights and node degrees
  nodes[0] = 0;
  tbb::parallel_invoke(
      [&] {
        tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
          NodeID u = _current_vertices[id];
          node_weights[u] = hg.nodeWeight(id);
          nodes[u + 1] = hg.nodeDegree(id);
        });
      },
      [&]() {
        tbb::parallel_for<NodeID>(num_nodes, n, [&](const NodeID u) {
          node_weights[u] = 0;
          nodes[u + 1] = hg.edgeSize(u - num_nodes);
        });
      });

  ASSERT(node_weights[_current_vertices[0]] == hg.nodeWeight(0));
  ASSERT(nodes[_current_vertices[0] + 1] == hg.nodeDegree(0));
  ASSERT(node_weights[num_nodes] == 0);
  ASSERT(nodes[num_nodes + 1] == hg.edgeSize(0));
  ASSERT(nodes[n] == hg.edgeSize(hg.initialNumEdges() - 1));
  // compute offset of neighborhoods in edge array
  parallel_prefix_sum(
      nodes.begin() + 1, nodes.end(), nodes.begin() + 1,
      [&](EdgeID x, EdgeID y) { return x + y; }, 0);

  const EdgeID max_edges_in_expansion = hg.maxEdgeSize();
  // set edges and edge weights
  tbb::parallel_invoke(
      [&]() {
        // neighborhoods representing incident nets
        tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
          const NodeID u = _current_vertices[id];
          EdgeID pos = nodes[u];
          for (const HyperedgeID &he : hg.incidentEdges(id)) {
            ASSERT(hg.edgeSize(he) >= 2, "Empty or single nets encountered.");
            edges[pos] = he + num_nodes; // hyperedges are shifted by num_nodes
            edge_weights[pos] = toEdgeWeight(getExpandedEdgeWeight(
                he, hg.edgeSize(he), max_edges_in_expansion));
            ++pos;
          }
        });
      },
      [&]() {
        // neighborhoods representing pins
        tbb::parallel_for<NodeID>(
            UL(0), hg.initialNumEdges(), [&](const NodeID he) {
              EdgeID pos = nodes[he + num_nodes];
              const EdgeWeight edge_weight = toEdgeWeight(getExpandedEdgeWeight(
                  he, hg.edgeSize(he), max_edges_in_expansion));
              for (const HypernodeID &hv : hg.pins(he)) {
                edges[pos] =
                    _current_vertices[hv]; // hypervertex ids remain identical
                edge_weights[pos] = edge_weight;
                ++pos;
              }
            });
      });

  constexpr bool neighborhood_sorted = false;
  return std::make_unique<kaminpar::shm::CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights),
      std::move(edge_weights), neighborhood_sorted);
}

template <typename TypeTraits>
std::unique_ptr<kaminpar::shm::CSRGraph>
ExperimentalCoarsener<TypeTraits>::buildCycleMatchingRep() {
  using namespace kaminpar;
  using namespace kaminpar::shm;

  const Hypergraph &hg = Base::currentHypergraph();
  const HypernodeID num_nodes = hg.initialNumNodes();

  const NodeID n = num_nodes;
  const EdgeID m = 3 * hg.initialNumPins();

  _nodes_buf.resize(n + 1);
  _nodes_buf2.resize(n + 1);
  _edges_buf.resize(m);
  _edges_buf2.resize(m);
  _edge_weights_buf.resize(m);
  _edge_weights_buf2.resize(m);

  _nodes_buf[0] = 0;
  tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
    NodeID u = _current_vertices[id];
    _nodes_buf[u + 1] = 3 * hg.nodeDegree(id);
  });
  parallel_prefix_sum(
      _nodes_buf.begin() + 1, _nodes_buf.end(), _nodes_buf.begin() + 1,
      [&](EdgeID x, EdgeID y) { return x + y; }, 0);

  EdgeWeight max_edges_in_expansion = countEdgesInEexpansion(hg.maxEdgeSize());

  tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
    const NodeID u = _current_vertices[id];
    EdgeID pos = _nodes_buf[u];

    // expand hyperedges
    for (const HyperedgeID &he : hg.incidentEdges(id)) {
      const HyperedgeID edge_size = hg.edgeSize(he);
      auto pins = hg.pins(he);
      const HypernodeID rank =
          std::distance(pins.begin(), std::find(pins.begin(), pins.end(), id));
      ASSERT(rank < edge_size);
      const double weight = getExpandedEdgeWeight(
          he, countEdgesInEexpansion(hg.edgeSize(he)), max_edges_in_expansion);

      ASSERT(edge_size >= 2, "Empty or single nets encountered.");
      // cycle edges
      // rank -> rank+1
      {
        const auto target_rank = rank == edge_size - 1 ? 0 : rank + 1;
        ASSERT(*(pins.begin() + target_rank) < num_nodes);
        _edges_buf[pos] = _current_vertices[*(pins.begin() + target_rank)];
        _edge_weights_buf[pos] = weight;
        ++pos;
      }

      if (edge_size >= 3) {
        // rank -> rank-1
        const auto target_rank = rank == 0 ? edge_size - 1 : rank - 1;
        ASSERT(*(pins.begin() + target_rank) < num_nodes);
        _edges_buf[pos] = _current_vertices[*(pins.begin() + target_rank)];
        _edge_weights_buf[pos] = weight;
        ++pos;
      }
      if (edge_size >= 4) {
        if ((edge_size & 1) == 0 || rank + 1 < edge_size) {

          // edge_size is even or rank < edge_size-1
          // rank -> rank + edge_size/2
          EdgeID edge_size_half = edge_size / 2;
          if (rank >= edge_size_half) {
            ASSERT(*(pins.begin() + (rank - edge_size_half)) < num_nodes);
            _edges_buf[pos] =
                _current_vertices[*(pins.begin() + (rank - edge_size_half))];
          } else {
            ASSERT(*(pins.begin() + (rank + edge_size_half)) < num_nodes);
            _edges_buf[pos] =
                _current_vertices[*(pins.begin() + (rank + edge_size_half))];
          }

          _edge_weights_buf[pos] = weight;
          ++pos;
        }
      }
    }
    ASSERT(pos <= _nodes_buf[u + 1]);

    // compute node degree
    _nodes_buf2[u + 1] = merge_multiedges_in_neighborhood(
        _nodes_buf[u], pos, _edges_buf, _edge_weights_buf, _edges_buf2,
        _edge_weights_buf2);
  });
  using std::swap;
  swap(_nodes_buf2, _nodes_buf);

  // determine positions of merged neighborhood to defragment merged
  // neighborhoods
  _nodes_buf[0] = 0;
  parallel_prefix_sum(
      _nodes_buf.begin() + 1, _nodes_buf.end(), _nodes_buf.begin() + 1,
      [&](EdgeID x, EdgeID y) { return x + y; }, 0);
  defragment_neighborhoods(_nodes_buf2, _edges_buf2, _edge_weights_buf2,
                           _nodes_buf, _edges_buf, _edge_weights_buf);

  ASSERT(_nodes_buf[n] <= _edges_buf.size());

  constexpr bool neighborhood_sorted = true;
  return construct_graph_model_from_buffers(
      _nodes_buf, _edges_buf, _edge_weights_buf, neighborhood_sorted);
}

template <typename TypeTraits>
std::unique_ptr<kaminpar::shm::CSRGraph>
ExperimentalCoarsener<TypeTraits>::buildCycleRandomMatchingRep() {
  using namespace kaminpar;
  using namespace kaminpar::shm;

  const Hypergraph &hg = Base::currentHypergraph();
  const HypernodeID num_nodes = hg.initialNumNodes();
  const HypernodeID num_edges = hg.initialNumEdges();

  const NodeID n = num_nodes;
  const EdgeID m = 3 * hg.initialNumPins();

  _nodes_buf.resize(n + 1);
  _nodes_buf2.resize(n + 1);
  _edges_buf.resize(m);
  _edges_buf2.resize(m);
  _edge_weights_buf.resize(m);
  _edge_weights_buf2.resize(m);

  _nodes_buf[0] = 0;
  tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
    NodeID u = _current_vertices[id];
    _nodes_buf[u + 1] = 3 * hg.nodeDegree(id);
  });
  parallel_prefix_sum(
      _nodes_buf.begin() + 1, _nodes_buf.end(), _nodes_buf.begin() + 1,
      [&](EdgeID x, EdgeID y) { return x + y; }, 0);

  EdgeWeight max_edges_in_expansion = countEdgesInEexpansion(hg.maxEdgeSize());

  DBG << V(kNoEdge);
  tbb::parallel_for<NodeID>(UL(0), num_edges, [&](const EdgeID he) {
    auto pins = hg.pins(he);
    std::size_t edge_size = hg.edgeSize(he);

    auto edge_rank = [&](NodeID v) {
      auto neigh = hg.incidentEdges(v);
      auto target_it = std::find(neigh.begin(), neigh.end(), he);
      ASSERT(target_it != neigh.end());
      return std::distance(neigh.begin(), target_it);
    };

    const double weight = getExpandedEdgeWeight(
        he, countEdgesInEexpansion(hg.edgeSize(he)), max_edges_in_expansion);

    // based on D. Seemaier's implementation
    ASSERT(edge_size >= 2);
    if (edge_size == 2) {
      for (std::size_t current = 0; current < edge_size; ++current) {
        std::size_t next = (current == 0) ? edge_size - 1 : current - 1;

        std::size_t edge_displ =
            _nodes_buf[_current_vertices[*(pins.begin() + current)]] +
            edge_rank(*(pins.begin() + current)) * 3;

        ASSERT(edge_displ + 2 < _edges_buf.size());
        ASSERT(edge_displ + 2 <
               _nodes_buf[_current_vertices[*(pins.begin() + current)] + 1]);
        _edges_buf[edge_displ] = _current_vertices[*(pins.begin() + next)];
        _edges_buf[edge_displ + 1] = kNoEdge;
        _edges_buf[edge_displ + 2] = kNoEdge;
        _edge_weights_buf[edge_displ] = weight;
        _edge_weights_buf[edge_displ + 1] = 0;
        _edge_weights_buf[edge_displ + 2] = 0;
      }

      return;
    }
    if (edge_size == 3) {

      for (std::size_t current = 0; current < edge_size; ++current) {
        std::size_t next = (current + 1 == edge_size) ? 0 : current + 1;
        std::size_t prev = (current == 0) ? edge_size - 1 : current - 1;

        std::size_t edge_displ =
            _nodes_buf[_current_vertices[*(pins.begin() + current)]] +
            edge_rank(*(pins.begin() + current)) * 3;
        ASSERT(edge_displ + 2 < _edges_buf.size());
        ASSERT(edge_displ + 2 <
               _nodes_buf[_current_vertices[*(pins.begin() + current)] + 1]);

        _edges_buf[edge_displ] = _current_vertices[*(pins.begin() + next)];
        _edges_buf[edge_displ + 1] = _current_vertices[*(pins.begin() + prev)];
        _edges_buf[edge_displ + 2] = kNoEdge;
        _edge_weights_buf[edge_displ] = weight;
        _edge_weights_buf[edge_displ + 1] = weight;
        _edge_weights_buf[edge_displ + 2] = 0;
      }

      return;
    }

    std::vector<HypernodeID> open(edge_size);
    std::iota(open.begin(), open.end(), 0);
    std::mt19937 g(_context.partition.seed + he + _pass_nr);
    std::shuffle(open.begin(), open.end(), g);
    std::vector<bool> matched(edge_size, false);
    std::vector<HypernodeID> matched_pins(edge_size);

    for (std::size_t current = 0; current < edge_size; ++current) {
      std::size_t next = (current + 1 == edge_size) ? 0 : current + 1;
      std::size_t prev = (current == 0) ? edge_size - 1 : current - 1;

      if (!matched[current]) {
        std::size_t matched_pin = edge_size;
        do {
          matched_pin = edge_size;
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

      const auto id = *(pins.begin() + current);
      std::size_t edge_displ =
          _nodes_buf[_current_vertices[id]] + edge_rank(id) * 3;
      ASSERT(edge_displ + 2 < _edges_buf.size());
      ASSERT(edge_displ + 2 < _nodes_buf[_current_vertices[id] + 1]);
      _edges_buf[edge_displ] = _current_vertices[*(pins.begin() + next)];
      _edges_buf[edge_displ + 1] = _current_vertices[*(pins.begin() + prev)];
      _edge_weights_buf[edge_displ] = weight;
      _edge_weights_buf[edge_displ + 1] = weight;
      if (matched[current]) {
        ASSERT(matched_pins[current] < edge_size);
        ASSERT(matched_pins[current] != next);
        ASSERT(matched_pins[current] != prev);
        ASSERT(matched_pins[current] != current);
        ASSERT(matched_pins[matched_pins[current]] == current);
        _edges_buf[edge_displ + 2] =
            _current_vertices[*(pins.begin() + matched_pins[current])];
        _edge_weights_buf[edge_displ + 2] = weight;
      } else {
        // unmatched
        _edges_buf[edge_displ + 2] = kNoEdge;
        _edge_weights_buf[edge_displ + 2] = 0;
      }
    }
  });

  tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
    const NodeID u = _current_vertices[id];

    for (std::size_t i = _nodes_buf[u]; i < _nodes_buf[u + 1]; ++i) {
      ASSERT(_edges_buf[i] == kNoEdge || _edges_buf[i] < num_nodes,
             "Invalid edge detected in neighborhood.");
    }

    // compute node degree
    _nodes_buf2[u + 1] = merge_multiedges_in_neighborhood(
        _nodes_buf[u], _nodes_buf[u + 1], _edges_buf, _edge_weights_buf,
        _edges_buf2, _edge_weights_buf2);
  });
  _nodes_buf2[0] = 0;

  // determine positions of agg. neighborhood
  parallel_prefix_sum(
      _nodes_buf2.begin() + 1, _nodes_buf2.end(), _nodes_buf2.begin() + 1,
      [&](EdgeID x, EdgeID y) { return x + y; }, 0);

  using std::swap;
  // store fragmented agg. neighborhood in edges
  swap(_nodes_buf, _nodes_buf2);

  // defragment agg. neighborhoods with respect to nodes_agg
  defragment_neighborhoods(_nodes_buf2, _edges_buf2, _edge_weights_buf2,
                           _nodes_buf, _edges_buf, _edge_weights_buf);

  constexpr bool neighborhood_sorted = true;
  return construct_graph_model_from_buffers(
      _nodes_buf, _edges_buf, _edge_weights_buf, neighborhood_sorted);
}

template <typename TypeTraits>
std::unique_ptr<kaminpar::shm::CSRGraph>
ExperimentalCoarsener<TypeTraits>::buildCliqueRep() {
  using namespace kaminpar;
  using namespace kaminpar::shm;

  const Hypergraph &hg = Base::currentHypergraph();
  const HypernodeID num_nodes = hg.initialNumNodes();

  const NodeID n = num_nodes;
  _nodes_buf.resize(n + 1);
  _nodes_buf2.resize(n + 1);

  _nodes_buf[0] = 0;
  tbb::parallel_for<NodeID>(UL(0), num_nodes, [&](const NodeID id) {
    NodeID u = _current_vertices[id];
    _nodes_buf[u + 1] = 0;
    for (const HyperedgeID he : hg.incidentEdges(id)) {
      _nodes_buf[u + 1] += hg.edgeSize(he) - 1;
    }
  });

  parallel_prefix_sum(
      _nodes_buf.begin() + 1, _nodes_buf.end(), _nodes_buf.begin() + 1,
      [&](EdgeID x, EdgeID y) { return x + y; }, 0);

  const EdgeID m = _nodes_buf[n];
  _edges_buf.resize(m);
  _edges_buf2.resize(m);
  _edge_weights_buf.resize(m);
  _edge_weights_buf2.resize(m);

  const EdgeWeight max_edges_in_expansion =
      countEdgesInEexpansion(hg.maxEdgeSize());
  tbb::parallel_for<HypernodeID>(UL(0), num_nodes, [&](const HyperedgeID hn) {
    // build neighborhood
    const NodeID u = _current_vertices[hn];
    EdgeID pos = _nodes_buf[u];
    for (const HyperedgeID he : hg.incidentEdges(hn)) {
      const double weight = getExpandedEdgeWeight(
          he, countEdgesInEexpansion(hg.edgeSize(he)), max_edges_in_expansion);
      for (const HypernodeID pin : hg.pins(he)) {
        if (pin != hn) {
          _edges_buf[pos] = _current_vertices[pin];
          _edge_weights_buf[pos++] = weight;
        }
      }
    }
    ASSERT(pos == _nodes_buf[u + 1], pos << " " << _nodes_buf[u + 1]);

    _nodes_buf2[u + 1] = merge_multiedges_in_neighborhood(
        _nodes_buf[u], _nodes_buf[u + 1], _edges_buf, _edge_weights_buf,
        _edges_buf2, _edge_weights_buf2);
  });
  _nodes_buf2[0] = 0;

  // determine defragmented positions of merged neighborhood
  parallel_prefix_sum(
      _nodes_buf2.begin() + 1, _nodes_buf2.end(), _nodes_buf2.begin() + 1,
      [&](EdgeID x, EdgeID y) { return x + y; }, 0);

  using std::swap;
  swap(_nodes_buf, _nodes_buf2);

  // defragment agg. neighborhoods with respect to nodes_agg
  defragment_neighborhoods(_nodes_buf2, _edges_buf2, _edge_weights_buf2,
                           _nodes_buf, _edges_buf, _edge_weights_buf);

  constexpr bool neighborhood_sorted = true;
  return construct_graph_model_from_buffers(
      _nodes_buf, _edges_buf, _edge_weights_buf, neighborhood_sorted);
}

template <typename TypeTraits>
bool ExperimentalCoarsener<TypeTraits>::coarseningPassImpl() {
  auto &timer = utils::Utilities::instance().getTimer(_context.utility_id);
  const auto pass_start_time = std::chrono::high_resolution_clock::now();
  timer.start_timer("coarsening_pass", "Clustering");

  // first, initialize the cluster ids
  const Hypergraph &hg = Base::currentHypergraph();
  DBG << V(_pass_nr) << V(hg.initialNumNodes()) << V(hg.initialNumEdges())
      << V(hg.initialNumPins());

  size_t num_nodes = Base::currentNumNodes();
  const double num_nodes_before_pass = num_nodes;
  vec<HypernodeID> clusters(num_nodes, kInvalidHypernode);
  _current_vertices.resize(num_nodes);
  tbb::parallel_for(UL(0), num_nodes, [&](HypernodeID u) {
    // cluster_weight[u] = hg.nodeWeight(u);
    clusters[u] = u;
    _current_vertices[u] = u;
  });

  DisableRandomization();
  if (_enable_randomization) {
    ASSERT(num_nodes == _current_vertices.size());
    utils::Randomize::instance().parallelShuffleVector(
        _current_vertices, UL(0), _current_vertices.size());
  }

  // build graph model
  kaminpar::shm::Graph graph([&]() {
    switch (_context.coarsening.rep) {
    case GraphRepresentation::bipartite:
      return buildBipartiteGraphRep();
    case GraphRepresentation::cycle_matching:
      return buildCycleMatchingRep();
    case GraphRepresentation::cycle_random_matching:
      return buildCycleRandomMatchingRep();
    case GraphRepresentation::clique:
      return buildCliqueRep();
    case GraphRepresentation::UNDEFINED:
      throw std::runtime_error("Undefined representation");
    }
    return std::unique_ptr<kaminpar::shm::CSRGraph>(nullptr);
  }());

  ASSERT(kaminpar::shm::debug::validate_graph(graph.csr_graph()),
         "Constructed graph model is invalid.");

  if (_context.coarsening.lp_sort) {
    graph =
        kaminpar::shm::graph::rearrange_by_degree_buckets(graph.csr_graph());
  }

  DBG << "expanded graph:" << V(graph.n()) << V(graph.m())
      << V(graph.csr_graph().max_degree());

  // configure LPClustering
  auto ctx = kaminpar::shm::create_default_context();
  ctx.parallel.num_threads = _context.shared_memory.num_threads;
  ctx.partition.setup(graph, _context.partition.k, _context.partition.epsilon);
  ctx.coarsening.clustering.lp.num_iterations =
      _context.coarsening.lp_iterations;
  kaminpar::Random::reseed(_context.partition.seed + _pass_nr);

  // initialize and set config for LPClustering
  kaminpar::shm::LPClustering lp_clustering(ctx.coarsening);
  lp_clustering.set_max_cluster_weight(
      kaminpar::shm::compute_max_cluster_weight<kaminpar::shm::NodeWeight>(
          ctx.coarsening, ctx.partition, graph.n(), graph.total_node_weight()));
  std::size_t desired_num_clusters = 0;
  if (_context.coarsening.rep != GraphRepresentation::bipartite) {
    desired_num_clusters = static_cast<std::size_t>(
        graph.n() / _context.coarsening.maximum_shrink_factor);
  }
  lp_clustering.set_desired_cluster_count(desired_num_clusters);

  kaminpar::StaticArray<kaminpar::shm::NodeID> graph_clustering(graph.n());
  vec<std::atomic<kaminpar::shm::NodeID>> remap_clusters(graph.n());

  DBG << "start to compute clustering";
  lp_clustering.compute_clustering(graph_clustering, graph, false);
  DBG << "computed clustering";

  // compute cluster for hypernodes with clustering for the expanded graph
  if (_context.coarsening.lp_sort) {
    auto perm = graph.csr_graph().take_raw_permutation();
    // remap cluster labels to hypervertices as representatives
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
      const HypernodeID u = perm[_current_vertices[id]];
      const kaminpar::shm::NodeID label_u = graph_clustering[u];
      remap_clusters[label_u] = id;
    });

    // set cluster
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
      const HypernodeID u = perm[_current_vertices[id]];
      const kaminpar::shm::NodeID label_u = graph_clustering[u];
      ASSERT(remap_clusters[label_u] < num_nodes);
      clusters[id] = remap_clusters[label_u];
    });
  } else {
    // remap cluster labels to hypervertices as representatives
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
      const HypernodeID u = _current_vertices[id];
      const kaminpar::shm::NodeID label_u = graph_clustering[u];
      remap_clusters[label_u] = id;
    });

    // set cluster
    tbb::parallel_for(UL(0), num_nodes, [&](const HypernodeID id) {
      const HypernodeID u = _current_vertices[id];
      const kaminpar::shm::NodeID label_u = graph_clustering[u];
      ASSERT(remap_clusters[label_u] < num_nodes);
      clusters[id] = remap_clusters[label_u];
    });
  }

  HEAVY_COARSENING_ASSERT(
      [&] {
        for (NodeID i = 0; i < num_nodes; ++i) {
          const auto rep = clusters[i];
          const auto rrep = clusters[rep];
          if (rep != i && rrep != rep) {
            return false;
          }
        }
        return true;
      }(),
      "clustering is no flat clustering");

  // reduce number of clusters containing hypervertices
  HypernodeID new_num_nodes = tbb::parallel_reduce(
      tbb::blocked_range<HypernodeID>(UL(0), num_nodes), 0,
      [&](const tbb::blocked_range<HypernodeID> &range,
          HypernodeID init) -> HypernodeID {
        for (HypernodeID i = range.begin(); i != range.end(); ++i) {
          init += (clusters[i] == i);
        }
        return init;
      },
      std::plus<HypernodeID>());
  num_nodes = new_num_nodes;

  // END implementation of actual coarsening

  // Check clustering
  HEAVY_COARSENING_ASSERT(
      [&] {
        HypernodeID num_clusters = 0;
        for (const HypernodeID &hn : hg.nodes()) {
          const HypernodeID u = hn;
          const HypernodeID root_u = clusters[u];
          if (root_u != clusters[root_u]) {
            LOG << "Vertex " << u << " is in cluster with id " << root_u
                << " but " << root_u << " is not root of its own cluster.";
            return false;
          }
          if (clusters[root_u] >= num_nodes_before_pass) {
            LOG << "Vertex " << u << " is in cluster with id " << root_u
                << " but the cluster id " << root_u << " is not in range {0,..,"
                << num_nodes << "}.";
          }
          if (clusters[u] == u) {
            ++num_clusters;
          }
        }

        if (num_clusters != num_nodes) {
          LOG << "Checked number of cluster (" << num_clusters
              << ") does not match new number of nodes (" << num_nodes << ")";
          return false;
        }

        return true;
      }(),
      "Clustering computed invalid cluster ids and weights");

  timer.stop_timer("coarsening_pass");
  ++_pass_nr;
  DBG << V(num_nodes_before_pass / num_nodes);
  if (num_nodes_before_pass / num_nodes <=
      _context.coarsening.minimum_shrink_factor) {
    return false;
  }

  _timer.start_timer("contraction", "Contraction");
  // at this point, the coarsening is finished and we use the final cluster ids
  // to perform the contraction
  _uncoarseningData.performMultilevelContraction(
      std::move(clusters), false /* deterministic */, pass_start_time);
  _timer.stop_timer("contraction");
  return true;
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(ExperimentalCoarsener)

} // namespace mt_kahypar
