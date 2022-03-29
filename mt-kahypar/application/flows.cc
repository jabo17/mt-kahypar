/*******************************************************************************
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 *
 * Mt-KaHyPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Mt-KaHyPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Mt-KaHyPar.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <iostream>


#include "mt-kahypar/io/command_line_options.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/io/sql_plottools_serializer.h"
#include "mt-kahypar/io/partitioning_output.h"
#include "mt-kahypar/partition/refinement/flows/scheduler.h"

#include "mt-kahypar/utils/randomize.h"

int main(int argc, char* argv[]) {

  mt_kahypar::Context context;
  mt_kahypar::processCommandLineInput(context, argc, argv);
  if (context.partition.verbose_output) {
    mt_kahypar::io::printBanner();
  }

  mt_kahypar::utils::Randomize::instance().setSeed(context.partition.seed);
  if ( context.shared_memory.use_localized_random_shuffle ) {
    mt_kahypar::utils::Randomize::instance().enableLocalizedParallelShuffle(
      context.shared_memory.shuffle_block_size);
  }

  size_t num_available_cpus = mt_kahypar::HardwareTopology::instance().num_cpus();
  if ( num_available_cpus < context.shared_memory.num_threads ) {
    WARNING("There are currently only" << num_available_cpus << "cpus available."
      << "Setting number of threads from" << context.shared_memory.num_threads
      << "to" << num_available_cpus);
    context.shared_memory.num_threads = num_available_cpus;
  }

  // Initialize TBB task arenas on numa nodes
  mt_kahypar::TBBInitializer::instance(context.shared_memory.num_threads);

  // We set the membind policy to interleaved allocations in order to
  // distribute allocations evenly across NUMA nodes
  hwloc_cpuset_t cpuset = mt_kahypar::TBBInitializer::instance().used_cpuset();
  mt_kahypar::parallel::HardwareTopology<>::instance().activate_interleaved_membind_policy(cpuset);
  hwloc_bitmap_free(cpuset);

  // Read Hypergraph
  mt_kahypar::Hypergraph hypergraph = mt_kahypar::io::readInputFile(
      context.partition.graph_filename,
      context.partition.file_format,
      context.preprocessing.stable_construction_of_incident_edges);
  context.setupPartWeights(hypergraph.totalWeight());
  context.setupThreadsPerFlowSearch();

  std::vector<mt_kahypar::PartitionID> input_partition;
  mt_kahypar::io::readPartitionFile(context.partition.input_partition_filename, input_partition);
  mt_kahypar::PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph, mt_kahypar::parallel_tag_t { });
  partitioned_hypergraph.doParallelForAllNodes([&](const mt_kahypar::HypernodeID& hn) {
    partitioned_hypergraph.setOnlyNodePart(hn, input_partition[hn]);
  });
  partitioned_hypergraph.initializePartition();

  mt_kahypar::io::printInitialPartitioningBanner(context);
  mt_kahypar::io::printPartitioningResults(partitioned_hypergraph, context, "Initial Partition");

  mt_kahypar::Metrics metrics;
  metrics.cut = mt_kahypar::metrics::hyperedgeCut(partitioned_hypergraph);
  metrics.km1 = mt_kahypar::metrics::km1(partitioned_hypergraph);
  metrics.imbalance = mt_kahypar::metrics::imbalance(partitioned_hypergraph, context);
  mt_kahypar::vec<mt_kahypar::HypernodeID> refinement_nodes;

  // Partition Hypergraph
  mt_kahypar::HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
  context.refinement.flows.skip_unpromising_blocks = false;
  double relative_improvement = 1.0;
  mt_kahypar::FlowRefinementScheduler flows(hypergraph, context);
  while ( relative_improvement > context.refinement.relative_improvement_threshold ) {
    mt_kahypar::HyperedgeWeight metric_before = metrics.getMetric(context.partition.mode, context.partition.objective);
    flows.initialize(partitioned_hypergraph);
    flows.refine(partitioned_hypergraph, refinement_nodes, metrics, std::numeric_limits<double>::max());
    mt_kahypar::HyperedgeWeight metric_after = metrics.getMetric(context.partition.mode, context.partition.objective);
    relative_improvement = static_cast<double>(metric_before) / metric_after - 1.0;
  }
  mt_kahypar::HighResClockTimepoint end = std::chrono::high_resolution_clock::now();

  // Print Stats
  std::chrono::duration<double> elapsed_seconds(end - start);
  mt_kahypar::io::printPartitioningResults(partitioned_hypergraph, context, elapsed_seconds);


  if (context.partition.write_partition_file) {
    mt_kahypar::io::writePartitionFile(partitioned_hypergraph, context.partition.graph_partition_filename);
  }

  mt_kahypar::TBBInitializer::instance().terminate();
  return 0;
}
