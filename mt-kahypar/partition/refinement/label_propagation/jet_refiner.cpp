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

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/randomize.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/utilities.h"
#include "tbb/parallel_for.h"

namespace mt_kahypar {

template <typename TypeTraits, typename GainCache>
bool JetRefiner<TypeTraits, GainCache>::refineImpl(
    mt_kahypar_partitioned_hypergraph_t& phg,
    const parallel::scalable_vector<HypernodeID>&, Metrics& best_metrics,
    const double) {
    PartitionedHypergraph& hypergraph = utils::cast<PartitionedHypergraph>(phg);
    unused(hypergraph);


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
