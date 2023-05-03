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

#include "mt-kahypar/partition/refinement/label_propagation/jet_refiner.h"

#include <common/datastructures/static_array.h>
#include <common/parallel/algorithm.h>
#include <common/timer.h>
#include <kaminpar/context.h>
#include <kaminpar/datastructures/graph.h>
#include <kaminpar/datastructures/partitioned_graph.h>
#include <kaminpar/kaminpar.h>
#include <kaminpar/metrics.h>
#include <kaminpar/refinement/jet_refiner.h>
#include <tbb/parallel_for.h>

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/randomize.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/utilities.h"

namespace mt_kahypar {

template <typename TypeTraits, typename GainCache>
bool JetRefiner<TypeTraits, GainCache>::refineImpl(
    mt_kahypar_partitioned_hypergraph_t& phg,
    const parallel::scalable_vector<HypernodeID>&, Metrics& best_metrics,
    const double) {
    using namespace kaminpar;
    using namespace kaminpar::shm;

    PartitionedHypergraph& hypergraph = utils::cast<PartitionedHypergraph>(phg);

    DISABLE_TIMERS();

    LOG << "[MtKaHyPar] Metrics *before* calling JetRefiner: cut="
        << metrics::hyperedgeCut(hypergraph)
        << " imbalance=" << metrics::imbalance(hypergraph, _context);
    LOG << "[MtKaHyPar] Number of removed nodes: "
        << hypergraph.numRemovedHypernodes();

    const NodeID n =
        hypergraph.initialNumNodes() - hypergraph.numRemovedHypernodes();
    const EdgeID m = hypergraph.initialNumEdges();

    StaticArray<NodeID> dense(hypergraph.initialNumNodes() + 1);
    hypergraph.doParallelForAllNodes(
        [&](const HypernodeID u) { dense[u + 1] = 1; });
    kaminpar::parallel::prefix_sum(dense.begin(), dense.end(), dense.begin());

    StaticArray<EdgeID> xadj(n + 1);
    StaticArray<NodeID> adjncy(m);
    StaticArray<NodeWeight> vwgt(n);
    StaticArray<EdgeWeight> adjwgt(m);
    StaticArray<BlockID> part(n);

    hypergraph.doParallelForAllNodes([&](const HypernodeID u) {
        const NodeID du = dense[u];
        xadj[du + 1] = hypergraph.nodeDegree(u);
        vwgt[du] = hypergraph.nodeWeight(u);
        part[du] = hypergraph.partID(u);
    });

    kaminpar::parallel::prefix_sum(xadj.begin(), xadj.end(), xadj.begin());
    // LOG << xadj.back() << m;

    hypergraph.doParallelForAllNodes([&](const HypernodeID u) {
        const NodeID du = dense[u];
        HypernodeID offset = 0;

        for (const HyperedgeID e : hypergraph.incidentEdges(u)) {
            const HypernodeID v = hypergraph.edgeTarget(e);
            const NodeID dv = dense[v];

            adjncy[xadj[du] + offset] = dv;
            adjwgt[xadj[du] + offset] = hypergraph.edgeWeight(e);

            ++offset;
        }
    });

    kaminpar::shm::Graph graph(std::move(xadj), std::move(adjncy),
                               std::move(vwgt), std::move(adjwgt), false);
    kaminpar::shm::PartitionedGraph p_graph(graph, hypergraph.k(),
                                            std::move(part));
    // kaminpar::shm::validate_graph(graph);

    kaminpar::shm::Context ctx = kaminpar::shm::create_default_context();
    ctx.partition.k = hypergraph.k();
    ctx.partition.epsilon = 1.0 * _context.partition.max_part_weights[0] *
                                hypergraph.k() / graph.total_node_weight() -
                            1.0;
    ctx.setup(graph);

    LOG << "[KaMinPar] Metrics *before* calling JetRefiner: cut="
        << kaminpar::shm::metrics::edge_cut(p_graph)
        << " imbalance=" << kaminpar::shm::metrics::imbalance(p_graph)
        << " epsilon=" << ctx.partition.epsilon
        << " max_block_weight=" << ctx.partition.block_weights.max(0)
        << " max_block_weight'=" << _context.partition.max_part_weights[0];

    kaminpar::shm::JetRefiner jet(ctx);
    jet.initialize(p_graph);
    jet.refine(p_graph, ctx.partition);

    LOG << "[KaMinPar] Metrics *after* calling JetRefiner: cut="
        << kaminpar::shm::metrics::edge_cut(p_graph)
        << " imbalance=" << kaminpar::shm::metrics::imbalance(p_graph);

    hypergraph.doParallelForAllNodes([&](const HypernodeID u) {
        const NodeID du = dense[u];
        hypergraph.changeNodePart(u, hypergraph.partID(u), p_graph.block(du));
    });

    LOG << "[MtKaHyPar] Metrics *after* calling JetRefiner: cut="
        << metrics::hyperedgeCut(hypergraph)
        << " imbalance=" << metrics::imbalance(hypergraph, _context);

    best_metrics.km1 = metrics::hyperedgeCut(hypergraph);
    best_metrics.cut = metrics::hyperedgeCut(hypergraph);
    best_metrics.imbalance = metrics::imbalance(hypergraph, _context);
    return false /* converged */;
}

template <typename TypeTraits, typename GainCache>
void JetRefiner<TypeTraits, GainCache>::initializeImpl(
    mt_kahypar_partitioned_hypergraph_t& phg) {
    PartitionedHypergraph& hypergraph = utils::cast<PartitionedHypergraph>(phg);
    unused(hypergraph);
}

namespace {
#define JET_REFINER(X, Y) JetRefiner<X, Y>
}  // namespace

// explicitly instantiate so the compiler can generate them when compiling this
// cpp file
INSTANTIATE_CLASS_WITH_TYPE_TRAITS_AND_GAIN_CACHE(JET_REFINER)
}  // namespace mt_kahypar
