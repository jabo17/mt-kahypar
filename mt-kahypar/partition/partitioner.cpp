/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 * Copyright (C) 2019 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 *
 * KaHyPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaHyPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaHyPar.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "partitioner.h"

#include "mt-kahypar/io/partitioning_output.h"
#include "mt-kahypar/partition/multilevel.h"
#include "mt-kahypar/partition/preprocessing/sparsification/degree_zero_hn_remover.h"
#include "mt-kahypar/partition/preprocessing/sparsification/large_he_remover.h"
#include "mt-kahypar/partition/preprocessing/community_detection/parallel_louvain.h"
#include "mt-kahypar/utils/hypergraph_statistics.h"
#include "mt-kahypar/utils/stats.h"
#include "mt-kahypar/utils/timer.h"


namespace mt_kahypar {

  void setupContext(Hypergraph& hypergraph, Context& context) {
    context.partition.large_hyperedge_size_threshold = std::max(hypergraph.initialNumNodes() *
                                                                context.partition.large_hyperedge_size_threshold_factor, 100.0);
    context.sanityCheck();
    context.setupPartWeights(hypergraph.totalWeight());
    context.setupContractionLimit(hypergraph.totalWeight());

    // Setup enabled IP algorithms
    if ( context.initial_partitioning.enabled_ip_algos.size() > 0 &&
         context.initial_partitioning.enabled_ip_algos.size() <
         static_cast<size_t>(InitialPartitioningAlgorithm::UNDEFINED) ) {
      ERROR("Size of enabled IP algorithms vector is smaller than number of IP algorithms!");
    } else if ( context.initial_partitioning.enabled_ip_algos.size() == 0 ) {
      context.initial_partitioning.enabled_ip_algos.assign(
        static_cast<size_t>(InitialPartitioningAlgorithm::UNDEFINED), true);
    } else {
      bool is_one_ip_algo_enabled = false;
      for ( size_t i = 0; i < context.initial_partitioning.enabled_ip_algos.size(); ++i ) {
        is_one_ip_algo_enabled |= context.initial_partitioning.enabled_ip_algos[i];
      }
      if ( !is_one_ip_algo_enabled ) {
        ERROR("At least one initial partitioning algorithm must be enabled!");
      }
    }
  }

  void configurePreprocessing(const Hypergraph& hypergraph, Context& context) {
    const double density = static_cast<double>(hypergraph.initialNumEdges()) /
                           static_cast<double>(hypergraph.initialNumNodes());
    if (context.preprocessing.community_detection.edge_weight_function == LouvainEdgeWeight::hybrid) {
      if (density < 0.75) {
        context.preprocessing.community_detection.edge_weight_function = LouvainEdgeWeight::degree;
      } else {
        context.preprocessing.community_detection.edge_weight_function = LouvainEdgeWeight::uniform;
      }
    }
  }

  void sanitize(Hypergraph& hypergraph, Context& context,
                DegreeZeroHypernodeRemover& degree_zero_hn_remover,
                LargeHyperedgeRemover& large_he_remover) {

    utils::Timer::instance().start_timer("degree_zero_hypernode_removal", "Degree Zero Hypernode Removal");
    const HypernodeID num_removed_degree_zero_hypernodes =
            degree_zero_hn_remover.removeDegreeZeroHypernodes(hypergraph);
    utils::Timer::instance().stop_timer("degree_zero_hypernode_removal");

    utils::Timer::instance().start_timer("large_hyperedge_removal", "Large Hyperedge Removal");
    const HypernodeID num_removed_large_hyperedges =
            large_he_remover.removeLargeHyperedges(hypergraph);
    utils::Timer::instance().stop_timer("large_hyperedge_removal");

    const HyperedgeID num_removed_single_node_hes = hypergraph.numRemovedHyperedges();
    if (context.partition.verbose_output &&
        ( num_removed_single_node_hes > 0 ||
          num_removed_degree_zero_hypernodes > 0 ||
          num_removed_large_hyperedges > 0 )) {
      LOG << "Performed single-node/large HE removal and degree-zero HN contractions:";
      LOG << "\033[1m\033[31m" << " # removed"
          << num_removed_single_node_hes << "single-pin hyperedges during hypergraph file parsing"
          << "\033[0m";
      LOG << "\033[1m\033[31m" << " # removed"
          << num_removed_large_hyperedges << "large hyperedges with |e| >" << large_he_remover.largeHyperedgeThreshold() << "\033[0m";
      LOG << "\033[1m\033[31m" << " # contracted"
          << num_removed_degree_zero_hypernodes << "hypernodes with d(v) = 0 and w(v) = 1"
          << "\033[0m";
      io::printStripe();
    }
  }

  bool is_mesh_graph(const Hypergraph& graph) {
    const HypernodeID num_nodes = graph.initialNumNodes();
    const double avg_hn_degree = utils::avgHypernodeDegree(graph);
    std::vector<HyperedgeID> hn_degrees;
    hn_degrees.resize(graph.initialNumNodes());
    graph.doParallelForAllNodes([&](const HypernodeID& hn) {
      hn_degrees[hn] = graph.nodeDegree(hn);
    });
    const double stdev_hn_degree = utils::parallel_stdev(hn_degrees, avg_hn_degree, num_nodes);
    if (stdev_hn_degree > avg_hn_degree / 2) {
      return false;
    }

    // test whether 99.9th percentile hypernode degree is at most 4 times the average degree
    tbb::enumerable_thread_specific<size_t> num_high_degree_nodes(0);
    graph.doParallelForAllNodes([&](const HypernodeID& node) {
      if (graph.nodeDegree(node) > 4 * avg_hn_degree) {
        num_high_degree_nodes.local() += 1;
      }
    });
    return num_high_degree_nodes.combine(std::plus<>()) <= num_nodes / 1000;
  }

  void preprocess(Hypergraph& hypergraph, Context& context) {
    bool use_community_detection = context.preprocessing.use_community_detection;

    #ifdef USE_GRAPH_PARTITIONER
    if (use_community_detection && context.preprocessing.disable_community_detection_for_mesh_graphs) {
      utils::Timer::instance().start_timer("detect_mesh_graph", "Detect Mesh Graph");
      use_community_detection = !is_mesh_graph(hypergraph);
      utils::Timer::instance().stop_timer("detect_mesh_graph");
    }
    #endif

    if ( use_community_detection ) {
      io::printTopLevelPreprocessingBanner(context);

      utils::Timer::instance().start_timer("community_detection", "Community Detection");
      utils::Timer::instance().start_timer("construct_graph", "Construct Graph");
      Graph graph(hypergraph, context.preprocessing.community_detection.edge_weight_function);
      if ( !context.preprocessing.community_detection.low_memory_contraction ) {
        graph.allocateContractionBuffers();
      }
      utils::Timer::instance().stop_timer("construct_graph");
      utils::Timer::instance().start_timer("perform_community_detection", "Perform Community Detection");
      ds::Clustering communities = community_detection::run_parallel_louvain(graph, context);
      graph.restrictClusteringToHypernodes(hypergraph, communities);
      hypergraph.setCommunityIDs(std::move(communities));
      utils::Timer::instance().stop_timer("perform_community_detection");
      utils::Timer::instance().stop_timer("community_detection");

      if (context.partition.verbose_output) {
        io::printCommunityInformation(hypergraph);
      }
    }
    parallel::MemoryPool::instance().release_mem_group("Preprocessing");
  }

  PartitionedHypergraph partitionVCycle(Hypergraph& hypergraph,
                                        PartitionedHypergraph&& partitioned_hypergraph,
                                        Context& context,
                                        LargeHyperedgeRemover& large_he_remover) {
    ASSERT(context.partition.num_vcycles > 0);

    for ( size_t i = 0; i < context.partition.num_vcycles; ++i ) {
      // Reset memory pool
      hypergraph.reset();
      parallel::MemoryPool::instance().reset();
      parallel::MemoryPool::instance().release_mem_group("Preprocessing");

      if ( context.partition.paradigm == Paradigm::nlevel ) {
        // Workaround: reset() function of hypergraph reinserts all removed
        // hyperedges to incident net lists of each vertex again.
        large_he_remover.removeLargeHyperedgesInNLevelVCycle(hypergraph);
      }

      // Store partition and assign it as community ids in order to
      // restrict contractions in v-cycle to partition ids
      hypergraph.doParallelForAllNodes([&](const HypernodeID& hn) {
        hypergraph.setCommunityID(hn, partitioned_hypergraph.partID(hn));
      });

      // V-Cycle Multilevel Partitioning
      io::printVCycleBanner(context, i + 1);
      // TODO why does this have to make a copy
      partitioned_hypergraph = multilevel::partition(
              hypergraph, context, true /* vcycle */);
    }

    return std::move(partitioned_hypergraph);
  }

  PartitionedHypergraph partition(Hypergraph& hypergraph, Context& context, const bool initialize_context) {
    if ( initialize_context ) {
      configurePreprocessing(hypergraph, context);
      setupContext(hypergraph, context);
      io::printContext(context);
      io::printMemoryPoolConsumption(context);
      io::printInputInformation(context, hypergraph);
    }

    // ################## PREPROCESSING ##################
    utils::Timer::instance().start_timer("preprocessing", "Preprocessing");
    preprocess(hypergraph, context);

    DegreeZeroHypernodeRemover degree_zero_hn_remover(context);
    LargeHyperedgeRemover large_he_remover(context);
    sanitize(hypergraph, context, degree_zero_hn_remover, large_he_remover);
    utils::Timer::instance().stop_timer("preprocessing");

    // ################## MULTILEVEL ##################
    PartitionedHypergraph partitioned_hypergraph = multilevel::partition(hypergraph, context);

    // ################## V-Cycle s##################
    if ( context.partition.num_vcycles > 0 ) {
      partitioned_hypergraph = partitionVCycle(
        hypergraph, std::move(partitioned_hypergraph),
        context, large_he_remover);
    }

    // ################## POSTPROCESSING ##################
    utils::Timer::instance().start_timer("postprocessing", "Postprocessing");
    large_he_remover.restoreLargeHyperedges(partitioned_hypergraph);
    degree_zero_hn_remover.restoreDegreeZeroHypernodes(partitioned_hypergraph);
    utils::Timer::instance().stop_timer("postprocessing");

    if (context.partition.verbose_output) {
      io::printHypergraphInfo(partitioned_hypergraph.hypergraph(), "Uncoarsened Hypergraph",
                              context.partition.show_memory_consumption);
      io::printStripe();
    }

    return partitioned_hypergraph;
  }

  #define NOOP_FUNC [] (const HyperedgeID, const HyperedgeWeight, const HypernodeID, const HypernodeID, const HypernodeID) { }

  PartitionedHypergraph social_partition(Hypergraph& hypergraph, Context& context) {
    configurePreprocessing(hypergraph, context);
    setupContext(hypergraph, context);

    io::printContext(context);
    io::printMemoryPoolConsumption(context);
    io::printInputInformation(context, hypergraph);

    // Extract High-Degree Subhypergraph
    PartitionedHypergraph tmp_phg(2, hypergraph, parallel_tag_t { });
    tmp_phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      tmp_phg.setOnlyNodePart(hn, 1);
    });
    tmp_phg.initializePartition();

    vec<HypernodeID> nodes(hypergraph.initialNumNodes());
    hypergraph.doParallelForAllNodes([&](const HypernodeID& hn) {
      nodes[hn] = hn;
    });
    tbb::parallel_sort(nodes.begin(), nodes.end(),
      [&](const HypernodeID& lhs, const HypernodeID& rhs) {
        return hypergraph.nodeDegree(lhs) > hypergraph.nodeDegree(rhs);
      });
    CAtomic<size_t> idx(0);
    CAtomic<HypernodeID> current_num_pins(0);
    tbb::parallel_for(0UL, nodes.size(), [&](const size_t) {
      const size_t i = idx.fetch_add(1, std::memory_order_relaxed);
      const HypernodeID hn = nodes[i];
      const HypernodeID num_pins = current_num_pins.fetch_add(hypergraph.nodeDegree(hn), std::memory_order_relaxed);
      const double pin_fraction = static_cast<double>(num_pins) / hypergraph.initialNumPins();
      if ( pin_fraction < 0.95 ) {
        tmp_phg.changeNodePart(hn, 1, 0, NOOP_FUNC);
      }
    });

    auto extracted_high_degree_core = tmp_phg.extract(0, true, false);
    Hypergraph& high_degree_hg = extracted_high_degree_core.first;
    high_degree_hg.setTotalWeight(hypergraph.totalWeight());

    if ( context.partition.verbose_output ) {
      io::printHypergraphInfo(high_degree_hg, "High-Degree Subhypergraph",
        context.partition.show_memory_consumption);
    }

    PartitionedHypergraph high_degree_phg = partition(high_degree_hg, context, false);

    vec<HypernodeID>& high_degree_mapping = extracted_high_degree_core.second;
    vec<HypernodeID> communities(hypergraph.initialNumNodes(), 0);
    tbb::parallel_for(ID(0), hypergraph.initialNumNodes(), [&](const HypernodeID& hn) {
      if ( high_degree_mapping[hn] != kInvalidHypernode ) {
        communities[hn] = high_degree_phg.partID(high_degree_mapping[hn]);
      } else {
        communities[hn] = ID(context.partition.k) + hn;
      }
    });
    Hypergraph contracted_hg = hypergraph.contract(communities);

    if ( context.partition.verbose_output ) {
      io::printHypergraphInfo(contracted_hg, "Contracted Hypergraph",
        context.partition.show_memory_consumption);
    }

    parallel::MemoryPool::instance().reset();
    PartitionedHypergraph contracted_phg = partition(contracted_hg, context, false);

    PartitionedHypergraph phg(context.partition.k, hypergraph, parallel_tag_t { });
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      phg.setOnlyNodePart(hn, contracted_phg.partID(communities[hn]));
    });
    phg.initializePartition();
    return phg;
  }


}