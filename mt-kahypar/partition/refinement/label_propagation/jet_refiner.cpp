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

    LOG << "[MtKaHyPar] Metrics *before* calling JetRefiner: cut="
        << metrics::hyperedgeCut(hypergraph)
        << " imbalance=" << metrics::imbalance(hypergraph, _context);

    StaticArray<EdgeID> xadj(hypergraph.initialNumNodes() + 1);
    StaticArray<NodeID> adjncy(hypergraph.initialNumEdges());
    StaticArray<NodeWeight> vwgt(hypergraph.initialNumNodes());
    StaticArray<EdgeWeight> adjwgt(hypergraph.initialNumEdges());
    StaticArray<BlockID> part(hypergraph.initialNumNodes());

    hypergraph.doParallelForAllNodes([&](const HypernodeID u) {
        xadj[u + 1] = hypergraph.nodeDegree(u);
        vwgt[u] = hypergraph.nodeWeight(u);
        part[u] = hypergraph.partID(u);
    });

    ::kaminpar::parallel::prefix_sum(xadj.begin(), xadj.end(), xadj.begin());

    hypergraph.doParallelForAllNodes([&](const HypernodeID u) {
        HypernodeID offset = 0;
        for (const HyperedgeID e : hypergraph.incidentEdges(u)) {
            adjncy[xadj[u] + offset] = hypergraph.edgeTarget(e);
            adjwgt[xadj[u] + offset] = hypergraph.edgeWeight(e);
            ++offset;
        }
    });

    kaminpar::shm::Graph graph(std::move(xadj), std::move(adjncy),
                               std::move(vwgt), std::move(adjwgt), false);
    kaminpar::shm::PartitionedGraph p_graph(graph, hypergraph.k(),
                                            std::move(part));

    kaminpar::shm::Context ctx = kaminpar::shm::create_default_context();
    ctx.partition.k = hypergraph.k();
    ctx.partition.epsilon = _context.partition.epsilon;
    ctx.setup(graph);

    ctx.refinement.algorithms = {
        kaminpar::shm::RefinementAlgorithm::GREEDY_BALANCER,
        kaminpar::shm::RefinementAlgorithm::JET};
    ctx.refinement.jet.num_iterations = 12;

    LOG << "[KaMinPar] Metrics *before* calling JetRefiner: cut="
        << kaminpar::shm::metrics::edge_cut(p_graph)
        << " imbalance=" << kaminpar::shm::metrics::imbalance(p_graph);

    kaminpar::shm::JetRefiner jet(ctx);
    jet.initialize(p_graph);
    jet.refine(p_graph, ctx.partition);

    LOG << "[KaMinPar] Metrics *after* calling JetRefiner: cut="
        << kaminpar::shm::metrics::edge_cut(p_graph)
        << " imbalance=" << kaminpar::shm::metrics::imbalance(p_graph);

    hypergraph.doParallelForAllNodes([&](const HypernodeID u) {
        hypergraph.changeNodePart(u, hypergraph.partID(u), p_graph.block(u));
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
