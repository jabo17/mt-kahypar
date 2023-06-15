/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "context.h"

#include <algorithm>

namespace mt_kahypar {

  std::ostream & operator<< (std::ostream& str, const PartitioningParameters& params) {
    str << "Partitioning Parameters:" << std::endl;
    str << "  Hypergraph:                         " << params.graph_filename << std::endl;
    if ( params.write_partition_file ) {
      str << "  Partition File:                     " << params.graph_partition_filename << std::endl;
    }
    str << "  Mode:                               " << params.mode << std::endl;
    str << "  Objective:                          " << params.objective << std::endl;
    str << "  Gain Policy:                        " << params.gain_policy << std::endl;
    str << "  Input File Format:                  " << params.file_format << std::endl;
    if ( params.instance_type != InstanceType::UNDEFINED ) {
      str << "  Instance Type:                      " << params.instance_type << std::endl;
    }
    if ( params.preset_type != PresetType::UNDEFINED ) {
      str << "  Preset Type:                        " << params.preset_type << std::endl;
    }
    str << "  Partition Type:                     " << params.partition_type << std::endl;
    str << "  k:                                  " << params.k << std::endl;
    str << "  epsilon:                            " << params.epsilon << std::endl;
    str << "  seed:                               " << params.seed << std::endl;
    str << "  Number of V-Cycles:                 " << params.num_vcycles << std::endl;
    str << "  Ignore HE Size Threshold:           " << params.ignore_hyperedge_size_threshold << std::endl;
    str << "  Large HE Size Threshold:            " << params.large_hyperedge_size_threshold << std::endl;
    if ( params.use_individual_part_weights ) {
      str << "  Individual Part Weights:            ";
      for ( const HypernodeWeight& w : params.max_part_weights ) {
        str << w << " ";
      }
      str << std::endl;
    }
    if ( params.mode == Mode::deep_multilevel ) {
      str << "  Perform Parallel Recursion:         " << std::boolalpha
          << params.perform_parallel_recursion_in_deep_multilevel << std::endl;
    }
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const CommunityDetectionParameters& params) {
    str << "  Community Detection Parameters:" << std::endl;
    str << "    Edge Weight Function:                " << params.edge_weight_function << std::endl;
    str << "    Maximum Louvain-Pass Iterations:     " << params.max_pass_iterations << std::endl;
    str << "    Minimum Vertex Move Fraction:        " << params.min_vertex_move_fraction << std::endl;
    str << "    Vertex Degree Sampling Threshold:    " << params.vertex_degree_sampling_threshold << std::endl;
    str << "    Number of subrounds (deterministic): " << params.num_sub_rounds_deterministic << std::endl;
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const PreprocessingParameters& params) {
    str << "Preprocessing Parameters:" << std::endl;
    str << "  Use Community Detection:            " << std::boolalpha << params.use_community_detection << std::endl;
    str << "  Disable C. D. for Mesh Graphs:      " << std::boolalpha << params.disable_community_detection_for_mesh_graphs << std::endl;
    if (params.use_community_detection) {
      str << std::endl << params.community_detection;
    }
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const RatingParameters& params) {
    str << "  Rating Parameters:" << std::endl;
    str << "    Rating Function:                  " << params.rating_function << std::endl;
    str << "    Heavy Node Penalty:               " << params.heavy_node_penalty_policy << std::endl;
    str << "    Acceptance Policy:                " << params.acceptance_policy << std::endl;
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const CoarseningParameters& params) {
    str << "Coarsening Parameters:" << std::endl;
    str << "  Algorithm:                          " << params.algorithm << std::endl;
    str << "  Use Adaptive Edge Size:             " << std::boolalpha << params.use_adaptive_edge_size << std::endl;
    str << "  Max Allowed Weight Multiplier:      " << params.max_allowed_weight_multiplier << std::endl;
    str << "  Maximum Allowed Hypernode Weight:   " << params.max_allowed_node_weight << std::endl;
    str << "  Contraction Limit Multiplier:       " << params.contraction_limit_multiplier << std::endl;
    str << "  Deep ML Contraction Limit Multi.:   " << params.deep_ml_contraction_limit_multiplier << std::endl;
    str << "  Contraction Limit:                  " << params.contraction_limit << std::endl;
    str << "  Minimum Shrink Factor:              " << params.minimum_shrink_factor << std::endl;
    str << "  Maximum Shrink Factor:              " << params.maximum_shrink_factor << std::endl;
    str << "  Vertex Degree Sampling Threshold:   " << params.vertex_degree_sampling_threshold << std::endl;
    str << "  Number of subrounds (deterministic):" << params.num_sub_rounds_deterministic << std::endl;
    str << std::endl << params.rating;
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const LabelPropagationParameters& params) {
    str << "  Label Propagation Parameters:" << std::endl;
    str << "    Algorithm:                        " << params.algorithm << std::endl;
    if ( params.algorithm != LabelPropagationAlgorithm::do_nothing ) {
      str << "    Maximum Iterations:               " << params.maximum_iterations << std::endl;
      str << "    Rebalancing:                      " << std::boolalpha << params.rebalancing << std::endl;
      str << "    HE Size Activation Threshold:     " << std::boolalpha << params.hyperedge_size_activation_threshold << std::endl;
    }
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const JetParameters& params) {
    str << "  Jet Parameters:" << std::endl;
    str << "    Algorithm:                        " << params.algorithm << std::endl;
    if ( params.algorithm != JetAlgorithm::do_nothing ) {
      str << "    Num Iterations:                   " << params.num_iterations << std::endl;
      str << "    Relative Improvement Threshold:   " << params.relative_improvement_threshold << std::endl;
      str << "    Restrict to Border Nodes:         " << std::boolalpha << params.restrict_to_border_nodes << std::endl;
      str << "    Vertex Locking:                   " << std::boolalpha << params.vertex_locking << std::endl;
      str << "    Negative Gain Factor (Coarse):    " << params.negative_gain_factor_coarse << std::endl;
      str << "    Negative Gain Factor (Fine):      " << params.negative_gain_factor_fine << std::endl;
    }
    return str;
  }

  std::ostream& operator<<(std::ostream& out, const FMParameters& params) {
    out << "  FM Parameters: \n";
    out << "    Algorithm:                        " << params.algorithm << std::endl;
    if ( params.algorithm != FMAlgorithm::do_nothing && params.algorithm != FMAlgorithm::kway_fm ) {
      out << "    Rollback Strategy:                " << params.rollback_strategy << std::endl;
    }
    if ( params.algorithm != FMAlgorithm::do_nothing ) {
      out << "    Multitry Rounds:                  " << params.multitry_rounds << std::endl;
      out << "    Perform Moves Globally:           " << std::boolalpha << params.perform_moves_global << std::endl;
      out << "    Parallel Global Rollbacks:        " << std::boolalpha << params.rollback_parallel << std::endl;
      out << "    Rollback Bal. Violation Factor:   " << params.rollback_balance_violation_factor << std::endl;
      out << "    Num Seed Nodes:                   " << params.num_seed_nodes << std::endl;
      out << "    Enable Random Shuffle:            " << std::boolalpha << params.shuffle << std::endl;
      out << "    Obey Minimal Parallelism:         " << std::boolalpha << params.obey_minimal_parallelism << std::endl;
      out << "    Minimum Improvement Factor:       " << params.min_improvement << std::endl;
      out << "    Release Nodes:                    " << std::boolalpha << params.release_nodes << std::endl;
      out << "    Time Limit Factor:                " << params.time_limit_factor << std::endl;
    }
    if ( params.algorithm != FMAlgorithm::do_nothing && params.algorithm != FMAlgorithm::kway_fm ) {
      out << "    Penalize Moved Rebalancing Nodes: " << std::boolalpha << params.penalty_for_moved_rebalancing_nodes << std::endl;
      out << "    Bal. Viol. Factor Rebalancing:    " << std::boolalpha << params.rebalancing_use_violation_factor << std::endl;
      out << "    Use Moved Nodes:                  " << std::boolalpha << params.rebalancing_use_moved_nodes << std::endl;
      out << "    Merge Move at Rebalancing Pos.:   " << std::boolalpha << params.insert_merged_move_at_rebalancing_position << std::endl;
    }
    out << std::flush;
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const NLevelGlobalFMParameters& params) {
    if ( params.use_global_fm ) {
      out << "  Boundary FM Parameters: \n";
      out << "    Refine Until No Improvement:      " << std::boolalpha << params.refine_until_no_improvement << std::endl;
      out << "    Num Seed Nodes:                   " << params.num_seed_nodes << std::endl;
      out << "    Obey Minimal Parallelism:         " << std::boolalpha << params.obey_minimal_parallelism << std::endl;
    }
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const FlowParameters& params) {
    out << "  Flow Parameters: \n";
    out << "    Algorithm:                        " << params.algorithm << std::endl;
    if ( params.algorithm != FlowAlgorithm::do_nothing ) {
      out << "    Flow Scaling:                     " << params.alpha << std::endl;
      out << "    Maximum Number of Pins:           " << params.max_num_pins << std::endl;
      out << "    Find Most Balanced Cut:           " << std::boolalpha << params.find_most_balanced_cut << std::endl;
      out << "    Determine Distance From Cut:      " << std::boolalpha << params.determine_distance_from_cut << std::endl;
      out << "    Parallel Searches Multiplier:     " << params.parallel_searches_multiplier << std::endl;
      out << "    Number of Parallel Searches:      " << params.num_parallel_searches << std::endl;
      out << "    Maximum BFS Distance:             " << params.max_bfs_distance << std::endl;
      out << "    Min Rel. Improvement Per Round:   " << params.min_relative_improvement_per_round << std::endl;
      out << "    Time Limit Factor:                " << params.time_limit_factor << std::endl;
      out << "    Skip Small Cuts:                  " << std::boolalpha << params.skip_small_cuts << std::endl;
      out << "    Skip Unpromising Blocks:          " << std::boolalpha << params.skip_unpromising_blocks << std::endl;
      out << "    Pierce in Bulk:                   " << std::boolalpha << params.pierce_in_bulk << std::endl;
      out << std::flush;
    }
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const JetRebalancingParameters& params) {
    out << "  Jet Rebalancing Parameters: \n";
    out << "    Weak Iterations:                  " << params.num_weak_iterations << std::endl;
    out << "    Strong Iterations:                " << params.num_strong_iterations << std::endl;
    out << "    Heavy Vertex Exclusion Factor:    " << params.heavy_vertex_exclusion_factor << std::endl;
    out << "    Relative Deadzone Size:           " << params.relative_deadzone_size << std::endl;
    out << "    Use Greedy Balanced:              " << std::boolalpha << params.use_greedy_balanced_instead_of_strong_iteration << std::endl;
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const DeterministicRefinementParameters& params) {
    out << "    Number of sub-rounds for Sync LP:  " << params.num_sub_rounds_sync_lp << std::endl;
    out << "    Use active node set:               " << std::boolalpha << params.use_active_node_set << std::endl;
    out << "    recalculate gains on second apply: " << std::boolalpha
        << params.recalculate_gains_on_second_apply << std::endl;
    return out;
  }

  std::ostream & operator<< (std::ostream& str, const RefinementParameters& params) {
    str << "Refinement Parameters:" << std::endl;
    str << "  Rebalancing Algorithm:              " << params.rebalancer << std::endl;
    str << "  Refine Until No Improvement:        " << std::boolalpha << params.refine_until_no_improvement << std::endl;
    str << "  Relative Improvement Threshold:     " << params.relative_improvement_threshold << std::endl;
    str << "  Maximum Batch Size:                 " << params.max_batch_size << std::endl;
    str << "  Min Border Vertices Per Thread:     " << params.min_border_vertices_per_thread << std::endl;
    if (params.rebalancer == RebalancingAlgorithm::jet_rebalancer) {
      str << "\n" << params.jet_rebalancing;
    }
    str << "\n" << params.label_propagation;
    str << "\n" << params.jet;
    str << "\n" << params.fm;
    if ( params.global_fm.use_global_fm ) {
      str << "\n" << params.global_fm;
    }
    str << "\n" << params.flows;
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const InitialPartitioningParameters& params) {
    str << "Initial Partitioning Parameters:" << std::endl;
    str << "  Initial Partitioning Mode:          " << params.mode << std::endl;
    str << "  Number of Runs:                     " << params.runs << std::endl;
    str << "  Use Adaptive IP Runs:               " << std::boolalpha << params.use_adaptive_ip_runs << std::endl;
    if ( params.use_adaptive_ip_runs ) {
      str << "  Min Adaptive IP Runs:               " << params.min_adaptive_ip_runs << std::endl;
    }
    str << "  Perform Refinement On Best:         " << std::boolalpha << params.perform_refinement_on_best_partitions << std::endl;
    str << "  Fm Refinement Rounds:               " << params.fm_refinment_rounds << std::endl;
    str << "  Remove Degree-Zero HNs Before IP:   " << std::boolalpha << params.remove_degree_zero_hns_before_ip << std::endl;
    str << "  Maximum Iterations of LP IP:        " << params.lp_maximum_iterations << std::endl;
    str << "  Initial Block Size of LP IP:        " << params.lp_initial_block_size << std::endl;
    str << "\nInitial Partitioning ";
    str << params.refinement << std::endl;
    return str;
  }

  std::ostream & operator<< (std::ostream& str, const SharedMemoryParameters& params) {
    str << "Shared Memory Parameters:             " << std::endl;
    str << "  Number of Threads:                  " << params.num_threads << std::endl;
    str << "  Number of used NUMA nodes:          " << TBBInitializer::instance().num_used_numa_nodes() << std::endl;
    str << "  Use Localized Random Shuffle:       " << std::boolalpha << params.use_localized_random_shuffle << std::endl;
    str << "  Random Shuffle Block Size:          " << params.shuffle_block_size << std::endl;
    return str;
  }

  bool Context::isNLevelPartitioning() const {
    return partition.partition_type == N_LEVEL_GRAPH_PARTITIONING ||
      partition.partition_type == N_LEVEL_HYPERGRAPH_PARTITIONING;
  }

  bool Context::forceGainCacheUpdates() const {
    return isNLevelPartitioning() ||
      partition.mode == Mode::deep_multilevel ||
      refinement.refine_until_no_improvement;
  }

  void Context::setupPartWeights(const HypernodeWeight total_hypergraph_weight) {
    if (partition.use_individual_part_weights) {
      ASSERT(static_cast<size_t>(partition.k) == partition.max_part_weights.size());
      const HypernodeWeight max_part_weights_sum = std::accumulate(partition.max_part_weights.cbegin(),
                                                                   partition.max_part_weights.cend(), 0);
      double weight_fraction = total_hypergraph_weight / static_cast<double>(max_part_weights_sum);
      HypernodeWeight perfect_part_weights_sum = 0;
      partition.perfect_balance_part_weights.clear();
      for (const HyperedgeWeight& part_weight : partition.max_part_weights) {
        const HypernodeWeight perfect_weight = ceil(weight_fraction * part_weight);
        partition.perfect_balance_part_weights.push_back(perfect_weight);
        perfect_part_weights_sum += perfect_weight;
      }

      if (max_part_weights_sum < total_hypergraph_weight) {
        ERR("Sum of individual part weights is less than the total hypergraph weight. Finding a valid partition is impossible.\n"
                << "Total hypergraph weight: " << total_hypergraph_weight << "\n"
                << "Sum of part weights:     " << max_part_weights_sum);
      } else {
        // To avoid rounding issues, epsilon should be calculated using the sum of the perfect part weights instead of
        // the total hypergraph weight. See also recursive_bipartitioning_initial_partitioner
        partition.epsilon = std::min(0.99, max_part_weights_sum / static_cast<double>(std::max(perfect_part_weights_sum, 1)) - 1);
      }
    } else {
      partition.perfect_balance_part_weights.clear();
      partition.perfect_balance_part_weights.push_back(ceil(
              total_hypergraph_weight
              / static_cast<double>(partition.k)));
      for (PartitionID part = 1; part != partition.k; ++part) {
        partition.perfect_balance_part_weights.push_back(
                partition.perfect_balance_part_weights[0]);
      }
      partition.max_part_weights.clear();
      partition.max_part_weights.push_back((1 + partition.epsilon)
                                          * partition.perfect_balance_part_weights[0]);
      for (PartitionID part = 1; part != partition.k; ++part) {
        partition.max_part_weights.push_back(partition.max_part_weights[0]);
      }
    }
  }

  void Context::setupContractionLimit(const HypernodeWeight total_hypergraph_weight) {
    // Setup contraction limit
    if (initial_partitioning.mode == Mode::deep_multilevel) {
      coarsening.contraction_limit =
        std::max(2 * shared_memory.num_threads, static_cast<size_t>(partition.k)) *
          coarsening.contraction_limit_multiplier;
    } else {
      coarsening.contraction_limit =
              coarsening.contraction_limit_multiplier * partition.k;
    }

    // Setup maximum allowed vertex and high-degree vertex weight
    setupMaximumAllowedNodeWeight(total_hypergraph_weight);
  }

  void Context::setupMaximumAllowedNodeWeight(const HypernodeWeight total_hypergraph_weight) {
    HypernodeWeight min_block_weight = std::numeric_limits<HypernodeWeight>::max();
    for ( PartitionID part_id = 0; part_id < partition.k; ++part_id ) {
      min_block_weight = std::min(min_block_weight, partition.max_part_weights[part_id]);
    }

    double hypernode_weight_fraction =
            coarsening.max_allowed_weight_multiplier
            / coarsening.contraction_limit;
    coarsening.max_allowed_node_weight =
            std::ceil(hypernode_weight_fraction * total_hypergraph_weight);
    coarsening.max_allowed_node_weight =
            std::min(coarsening.max_allowed_node_weight, min_block_weight);
  }

  void Context::sanityCheck() {
    if ( isNLevelPartitioning() && coarsening.algorithm == CoarseningAlgorithm::multilevel_coarsener ) {
        ALGO_SWITCH("Coarsening algorithm" << coarsening.algorithm << "is only supported in multilevel mode."
                                           << "Do you want to use the n-level version instead (Y/N)?",
                    "Partitioning with" << coarsening.algorithm
                                        << "coarsener in n-level mode is not supported!",
                    coarsening.algorithm,
                    CoarseningAlgorithm::nlevel_coarsener);
    } else if ( !isNLevelPartitioning() && coarsening.algorithm == CoarseningAlgorithm::nlevel_coarsener ) {
        ALGO_SWITCH("Coarsening algorithm" << coarsening.algorithm << "is only supported in n-Level mode."
                                           << "Do you want to use the multilevel version instead (Y/N)?",
                    "Partitioning with" << coarsening.algorithm
                                        << "coarsener in multilevel mode is not supported!",
                    coarsening.algorithm,
                    CoarseningAlgorithm::multilevel_coarsener);
    }

    ASSERT(partition.use_individual_part_weights != partition.max_part_weights.empty());
    if (partition.use_individual_part_weights && static_cast<size_t>(partition.k) != partition.max_part_weights.size()) {
      ALGO_SWITCH("Individual part weights specified, but number of parts doesn't match k."
                          << "Do you want to use k =" << partition.max_part_weights.size() << "instead (Y/N)?",
                  "Number of parts is not equal to k!",
                  partition.k,
                  partition.max_part_weights.size());
    }


    shared_memory.static_balancing_work_packages = std::clamp(shared_memory.static_balancing_work_packages, UL(4), UL(256));

    if ( partition.deterministic ) {
      coarsening.algorithm = CoarseningAlgorithm::deterministic_multilevel_coarsener;

      // disable FM until we have a deterministic version
      refinement.fm.algorithm = FMAlgorithm::do_nothing;
      initial_partitioning.refinement.fm.algorithm = FMAlgorithm::do_nothing;

      // disable adaptive IP
      initial_partitioning.use_adaptive_ip_runs = false;


      // switch silently
      auto lp_algo = refinement.label_propagation.algorithm;
      if ( lp_algo != LabelPropagationAlgorithm::do_nothing && lp_algo != LabelPropagationAlgorithm::deterministic ) {
        refinement.label_propagation.algorithm = LabelPropagationAlgorithm::deterministic;
      }

      lp_algo = initial_partitioning.refinement.label_propagation.algorithm;
      if ( lp_algo != LabelPropagationAlgorithm::do_nothing && lp_algo != LabelPropagationAlgorithm::deterministic ) {
        initial_partitioning.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::deterministic;
      }
    }

    // Set correct gain policy type
    if ( partition.instance_type == InstanceType::hypergraph ) {
      switch ( partition.objective ) {
        case Objective::km1: partition.gain_policy = GainPolicy::km1; break;
        case Objective::cut: partition.gain_policy = GainPolicy::cut; break;
        case Objective::soed: partition.gain_policy = GainPolicy::soed; break;
        case Objective::UNDEFINED: partition.gain_policy = GainPolicy::none; break;
      }
    } else if ( partition.instance_type == InstanceType::graph ) {
      if ( partition.objective != Objective::cut ) {
        partition.objective = Objective::cut;
        INFO("All supported objective functions are equivalent for graphs. Objective function is set to edge cut metric.");
      }
      partition.gain_policy = GainPolicy::cut_for_graphs;
    }

    if ( partition.preset_type == PresetType::large_k ) {
      // Silently switch to deep multilevel scheme for large k partitioning
      partition.mode = Mode::deep_multilevel;
    }
  }

  void Context::setupThreadsPerFlowSearch() {
    if ( refinement.flows.algorithm == FlowAlgorithm::flow_cutter ) {
      // = min(t, min(tau * k, k * (k - 1) / 2))
      // t = number of threads
      // k * (k - 1) / 2 = maximum number of edges in the quotient graph
      refinement.flows.num_parallel_searches = partition.k == 2 ? 1 :
        std::min(shared_memory.num_threads, std::min(std::max(UL(1), static_cast<size_t>(
          refinement.flows.parallel_searches_multiplier * partition.k)),
            static_cast<size_t>((partition.k * (partition.k - 1)) / 2) ));
    }
  }

  void Context::load_default_preset() {
    // General
    partition.preset_type = PresetType::default_preset;
    partition.mode = Mode::direct;
    partition.large_hyperedge_size_threshold_factor = 0.01;
    partition.smallest_large_he_size_threshold = 50000;
    partition.ignore_hyperedge_size_threshold = 1000;
    partition.num_vcycles = 0;

    // shared_memory
    shared_memory.use_localized_random_shuffle = false;
    shared_memory.static_balancing_work_packages = 128;

    // preprocessing
    preprocessing.use_community_detection = true;
    preprocessing.disable_community_detection_for_mesh_graphs = true;
    preprocessing.community_detection.edge_weight_function = LouvainEdgeWeight::hybrid;
    preprocessing.community_detection.max_pass_iterations = 5;
    preprocessing.community_detection.min_vertex_move_fraction = 0.01;
    preprocessing.community_detection.vertex_degree_sampling_threshold = 200000;

    // coarsening
    coarsening.algorithm = CoarseningAlgorithm::multilevel_coarsener;
    coarsening.use_adaptive_edge_size= true;
    coarsening.minimum_shrink_factor = 1.01;
    coarsening.maximum_shrink_factor = 2.5;
    coarsening.max_allowed_weight_multiplier = 1.0;
    coarsening.contraction_limit_multiplier = 160;
    coarsening.vertex_degree_sampling_threshold = 200000;

    // coarsening -> rating
    coarsening.rating.rating_function = RatingFunction::heavy_edge;
    coarsening.rating.heavy_node_penalty_policy = HeavyNodePenaltyPolicy::no_penalty;
    coarsening.rating.acceptance_policy = AcceptancePolicy::best_prefer_unmatched;

    // initial partitioning
    initial_partitioning.mode = Mode::recursive_bipartitioning;
    initial_partitioning.runs = 20;
    initial_partitioning.use_adaptive_ip_runs = true;
    initial_partitioning.min_adaptive_ip_runs = 5;
    initial_partitioning.perform_refinement_on_best_partitions = true;
    initial_partitioning.fm_refinment_rounds = 1;
    initial_partitioning.lp_maximum_iterations = 20;
    initial_partitioning.lp_initial_block_size = 5;
    initial_partitioning.remove_degree_zero_hns_before_ip = true;

    // initial partitioning -> refinement
    initial_partitioning.refinement.refine_until_no_improvement = false;

    // initial partitioning -> refinement -> label propagation
    initial_partitioning.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::label_propagation;
    initial_partitioning.refinement.label_propagation.maximum_iterations = 5;
    initial_partitioning.refinement.label_propagation.rebalancing = true;
    initial_partitioning.refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // TODO(maas): jet??

    // initial partitioning -> refinement -> fm
    initial_partitioning.refinement.fm.algorithm = FMAlgorithm::kway_fm;
    initial_partitioning.refinement.fm.multitry_rounds = 5;
    initial_partitioning.refinement.fm.perform_moves_global = false;
    initial_partitioning.refinement.fm.rollback_parallel = true;
    initial_partitioning.refinement.fm.rollback_balance_violation_factor = 1;
    initial_partitioning.refinement.fm.num_seed_nodes = 25;
    initial_partitioning.refinement.fm.obey_minimal_parallelism = false;
    initial_partitioning.refinement.fm.release_nodes = true;
    initial_partitioning.refinement.fm.time_limit_factor = 0.25;
    initial_partitioning.refinement.fm.iter_moves_on_recalc = true;

    // initial partitioning -> refinement -> flows
    initial_partitioning.refinement.flows.algorithm = FlowAlgorithm::do_nothing;

    // refinement
    refinement.rebalancer = RebalancingAlgorithm::simple_rebalancer;
    refinement.refine_until_no_improvement = false;

    // refinement -> label propagation
    refinement.label_propagation.algorithm = LabelPropagationAlgorithm::label_propagation;
    refinement.label_propagation.maximum_iterations = 5;
    refinement.label_propagation.rebalancing = true;
    refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // refinement -> fm
    refinement.fm.algorithm = FMAlgorithm::kway_fm;
    refinement.fm.multitry_rounds = 10;
    refinement.fm.perform_moves_global = false;
    refinement.fm.rollback_parallel = true;
    refinement.fm.rollback_balance_violation_factor = 1.25;
    refinement.fm.num_seed_nodes = 25;
    refinement.fm.obey_minimal_parallelism = true;
    refinement.fm.release_nodes = true;
    refinement.fm.time_limit_factor = 0.25;
    refinement.fm.min_improvement = -1;
    refinement.fm.iter_moves_on_recalc = true;

    // refinement -> flows
    refinement.flows.algorithm = FlowAlgorithm::do_nothing;
  }

  void Context::load_default_flow_preset() {
    load_default_preset();

    // General
    partition.preset_type = PresetType::default_flows;

    // refinement
    refinement.refine_until_no_improvement = true;
    refinement.relative_improvement_threshold = 0.0025;

    // refinement -> flows;
    refinement.flows.algorithm = FlowAlgorithm::flow_cutter;
    refinement.flows.alpha = 16;
    refinement.flows.max_num_pins = 4294967295;
    refinement.flows.find_most_balanced_cut = true;
    refinement.flows.determine_distance_from_cut = true;
    refinement.flows.parallel_searches_multiplier = 1.0;
    refinement.flows.max_bfs_distance = 2;
    refinement.flows.time_limit_factor = 8;
    refinement.flows.skip_small_cuts = true;
    refinement.flows.skip_unpromising_blocks = true;
    refinement.flows.pierce_in_bulk = true;
    refinement.flows.min_relative_improvement_per_round = 0.001;
  }

  void Context::load_deterministic_preset() {
    // General
    partition.preset_type = PresetType::deterministic;
    partition.mode = Mode::direct;
    partition.deterministic = true;
    partition.large_hyperedge_size_threshold_factor = 0.01;
    partition.smallest_large_he_size_threshold = 50000;
    partition.ignore_hyperedge_size_threshold = 1000;
    partition.num_vcycles = 0;

    // shared_memory
    shared_memory.use_localized_random_shuffle = false;
    shared_memory.static_balancing_work_packages = 128;

    // preprocessing
    preprocessing.use_community_detection = true;
    preprocessing.disable_community_detection_for_mesh_graphs = true;
    preprocessing.stable_construction_of_incident_edges = true;
    preprocessing.community_detection.edge_weight_function = LouvainEdgeWeight::hybrid;
    preprocessing.community_detection.max_pass_iterations = 5;
    preprocessing.community_detection.min_vertex_move_fraction = 0.01;
    preprocessing.community_detection.vertex_degree_sampling_threshold = 200000;
    preprocessing.community_detection.low_memory_contraction = true;
    preprocessing.community_detection.num_sub_rounds_deterministic = 16;

    // coarsening
    coarsening.algorithm = CoarseningAlgorithm::deterministic_multilevel_coarsener;
    coarsening.use_adaptive_edge_size= true;
    coarsening.minimum_shrink_factor = 1.01;
    coarsening.maximum_shrink_factor = 2.5;
    coarsening.max_allowed_weight_multiplier = 1.0;
    coarsening.contraction_limit_multiplier = 160;
    coarsening.vertex_degree_sampling_threshold = 200000;
    coarsening.num_sub_rounds_deterministic = 3;

    // coarsening -> rating
    coarsening.rating.rating_function = RatingFunction::heavy_edge;
    coarsening.rating.heavy_node_penalty_policy = HeavyNodePenaltyPolicy::no_penalty;
    coarsening.rating.acceptance_policy = AcceptancePolicy::best_prefer_unmatched;

    // initial partitioning
    initial_partitioning.mode = Mode::recursive_bipartitioning;
    initial_partitioning.runs = 20;
    initial_partitioning.use_adaptive_ip_runs = false;
    initial_partitioning.perform_refinement_on_best_partitions = false;
    initial_partitioning.fm_refinment_rounds = 3;
    initial_partitioning.lp_maximum_iterations = 20;
    initial_partitioning.lp_initial_block_size = 5;
    initial_partitioning.population_size = 64;
    initial_partitioning.remove_degree_zero_hns_before_ip = true;

    // initial partitioning -> refinement
    initial_partitioning.refinement.refine_until_no_improvement = false;

    // initial partitioning -> refinement -> label propagation
    initial_partitioning.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::deterministic;
    initial_partitioning.refinement.label_propagation.maximum_iterations = 5;
    initial_partitioning.refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // initial partitioning -> refinement -> deterministic
    initial_partitioning.refinement.deterministic_refinement.num_sub_rounds_sync_lp = 1;
    initial_partitioning.refinement.deterministic_refinement.use_active_node_set = true;

    // initial partitioning -> refinement -> fm
    initial_partitioning.refinement.fm.algorithm = FMAlgorithm::do_nothing;

    // initial partitioning -> refinement -> flows
    initial_partitioning.refinement.flows.algorithm = FlowAlgorithm::do_nothing;

    // refinement
    refinement.rebalancer = RebalancingAlgorithm::simple_rebalancer;
    refinement.refine_until_no_improvement = false;

    // refinement -> label propagation
    refinement.label_propagation.algorithm = LabelPropagationAlgorithm::deterministic;
    refinement.label_propagation.maximum_iterations = 5;
    refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // refinement -> deterministic
    refinement.deterministic_refinement.num_sub_rounds_sync_lp = 1;
    refinement.deterministic_refinement.use_active_node_set = true;

    // refinement -> fm
    refinement.fm.algorithm = FMAlgorithm::do_nothing;

    // refinement -> flows
    refinement.flows.algorithm = FlowAlgorithm::do_nothing;
  }

  void Context::load_quality_preset() {
    // General
    partition.preset_type = PresetType::quality_preset;
    partition.mode = Mode::direct;
    partition.large_hyperedge_size_threshold_factor = 0.01;
    partition.smallest_large_he_size_threshold = 50000;
    partition.ignore_hyperedge_size_threshold = 1000;
    partition.num_vcycles = 0;

    // shared_memory
    shared_memory.use_localized_random_shuffle = false;
    shared_memory.static_balancing_work_packages = 128;

    // preprocessing
    preprocessing.use_community_detection = true;
    preprocessing.disable_community_detection_for_mesh_graphs = true;
    preprocessing.community_detection.edge_weight_function = LouvainEdgeWeight::hybrid;
    preprocessing.community_detection.max_pass_iterations = 5;
    preprocessing.community_detection.min_vertex_move_fraction = 0.01;
    preprocessing.community_detection.vertex_degree_sampling_threshold = 200000;

    // coarsening
    coarsening.algorithm = CoarseningAlgorithm::nlevel_coarsener;
    coarsening.use_adaptive_edge_size = true;
    coarsening.minimum_shrink_factor = 1.01;
    coarsening.maximum_shrink_factor = 100.0;
    coarsening.max_allowed_weight_multiplier = 1.0;
    coarsening.contraction_limit_multiplier = 160;
    coarsening.vertex_degree_sampling_threshold = 200000;

    // coarsening -> rating
    coarsening.rating.rating_function = RatingFunction::heavy_edge;
    coarsening.rating.heavy_node_penalty_policy = HeavyNodePenaltyPolicy::no_penalty;
    coarsening.rating.acceptance_policy = AcceptancePolicy::best_prefer_unmatched;

    // initial partitioning
    initial_partitioning.mode = Mode::recursive_bipartitioning;
    initial_partitioning.runs = 20;
    initial_partitioning.use_adaptive_ip_runs = true;
    initial_partitioning.min_adaptive_ip_runs = 5;
    initial_partitioning.perform_refinement_on_best_partitions = true;
    initial_partitioning.fm_refinment_rounds = 2147483647;
    initial_partitioning.lp_maximum_iterations = 20;
    initial_partitioning.lp_initial_block_size = 5;
    initial_partitioning.remove_degree_zero_hns_before_ip = true;

    // initial partitioning -> refinement
    initial_partitioning.refinement.refine_until_no_improvement = true;
    initial_partitioning.refinement.max_batch_size = 1000;
    initial_partitioning.refinement.min_border_vertices_per_thread = 0;

    // initial partitioning -> refinement -> label propagation
    initial_partitioning.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::label_propagation;
    initial_partitioning.refinement.label_propagation.maximum_iterations = 5;
    initial_partitioning.refinement.label_propagation.rebalancing = true;
    initial_partitioning.refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // initial partitioning -> refinement -> fm
    initial_partitioning.refinement.fm.algorithm = FMAlgorithm::kway_fm;
    initial_partitioning.refinement.fm.multitry_rounds = 5;
    initial_partitioning.refinement.fm.perform_moves_global = false;
    initial_partitioning.refinement.fm.rollback_parallel = false;
    initial_partitioning.refinement.fm.rollback_balance_violation_factor = 1;
    initial_partitioning.refinement.fm.num_seed_nodes = 5;
    initial_partitioning.refinement.fm.obey_minimal_parallelism = false;
    initial_partitioning.refinement.fm.release_nodes = true;
    initial_partitioning.refinement.fm.time_limit_factor = 0.25;
    initial_partitioning.refinement.fm.iter_moves_on_recalc = false;

    // initial partitioning -> refinement -> flows
    initial_partitioning.refinement.flows.algorithm = FlowAlgorithm::do_nothing;

    // initial partitioning -> refinement -> global fm
    initial_partitioning.refinement.global_fm.use_global_fm = false;

    // refinement
    refinement.rebalancer = RebalancingAlgorithm::simple_rebalancer;
    refinement.refine_until_no_improvement = true;
    refinement.max_batch_size = 1000;
    refinement.min_border_vertices_per_thread = 50;

    // refinement -> label propagation
    refinement.label_propagation.algorithm = LabelPropagationAlgorithm::label_propagation;
    refinement.label_propagation.maximum_iterations = 5;
    refinement.label_propagation.rebalancing = true;
    refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // refinement -> fm
    refinement.fm.algorithm = FMAlgorithm::kway_fm;
    refinement.fm.multitry_rounds = 10;
    refinement.fm.perform_moves_global = false;
    refinement.fm.rollback_parallel = false;
    refinement.fm.rollback_balance_violation_factor = 1.25;
    refinement.fm.num_seed_nodes = 5;
    refinement.fm.obey_minimal_parallelism = false;
    refinement.fm.release_nodes = true;
    refinement.fm.time_limit_factor = 0.25;
    refinement.fm.min_improvement = -1;
    refinement.fm.iter_moves_on_recalc = true;

    // refinement -> flows
    refinement.flows.algorithm = FlowAlgorithm::do_nothing;

    // refinement -> global fm
    refinement.global_fm.use_global_fm = true;
    refinement.global_fm.refine_until_no_improvement = false;
    refinement.global_fm.num_seed_nodes = 5;
    refinement.global_fm.obey_minimal_parallelism = true;
  }

  void Context::load_quality_flow_preset() {
    load_quality_preset();

    // General
    partition.preset_type = PresetType::quality_flows;

    // refinement
    refinement.relative_improvement_threshold = 0.0025;

    // refinement -> fm
    refinement.fm.iter_moves_on_recalc = false;

    // refinement -> flows;
    refinement.flows.algorithm = FlowAlgorithm::flow_cutter;
    refinement.flows.alpha = 16;
    refinement.flows.max_num_pins = 4294967295;
    refinement.flows.find_most_balanced_cut = true;
    refinement.flows.determine_distance_from_cut = true;
    refinement.flows.parallel_searches_multiplier = 1.0;
    refinement.flows.max_bfs_distance = 2;
    refinement.flows.time_limit_factor = 8;
    refinement.flows.skip_small_cuts = true;
    refinement.flows.skip_unpromising_blocks = true;
    refinement.flows.pierce_in_bulk = true;
    refinement.flows.min_relative_improvement_per_round = 0.001;

    // refinement -> global fm
    refinement.global_fm.refine_until_no_improvement = true;
  }

  void Context::load_large_k_preset() {
    // General
    partition.preset_type = PresetType::large_k;
    partition.mode = Mode::deep_multilevel;
    partition.large_hyperedge_size_threshold_factor = 0.01;
    partition.smallest_large_he_size_threshold = 50000;
    partition.ignore_hyperedge_size_threshold = 1000;
    partition.num_vcycles = 0;

    // shared_memory
    shared_memory.use_localized_random_shuffle = false;
    shared_memory.static_balancing_work_packages = 128;

    // preprocessing
    preprocessing.use_community_detection = true;
    preprocessing.disable_community_detection_for_mesh_graphs = true;
    preprocessing.community_detection.edge_weight_function = LouvainEdgeWeight::hybrid;
    preprocessing.community_detection.max_pass_iterations = 5;
    preprocessing.community_detection.min_vertex_move_fraction = 0.01;
    preprocessing.community_detection.vertex_degree_sampling_threshold = 200000;

    // coarsening
    coarsening.algorithm = CoarseningAlgorithm::multilevel_coarsener;
    coarsening.use_adaptive_edge_size= true;
    coarsening.minimum_shrink_factor = 1.01;
    coarsening.maximum_shrink_factor = 2.5;
    coarsening.max_allowed_weight_multiplier = 1.0;
    coarsening.contraction_limit_multiplier = 500;
    coarsening.deep_ml_contraction_limit_multiplier = 160;
    coarsening.vertex_degree_sampling_threshold = 200000;

    // coarsening -> rating
    coarsening.rating.rating_function = RatingFunction::heavy_edge;
    coarsening.rating.heavy_node_penalty_policy = HeavyNodePenaltyPolicy::no_penalty;
    coarsening.rating.acceptance_policy = AcceptancePolicy::best_prefer_unmatched;

    // initial partitioning
    initial_partitioning.mode = Mode::direct;
    initial_partitioning.runs = 5;
    initial_partitioning.use_adaptive_ip_runs = true;
    initial_partitioning.min_adaptive_ip_runs = 3;
    initial_partitioning.perform_refinement_on_best_partitions = true;
    initial_partitioning.fm_refinment_rounds = 1;
    initial_partitioning.lp_maximum_iterations = 20;
    initial_partitioning.lp_initial_block_size = 5;
    initial_partitioning.enabled_ip_algos = {1, 1, 0, 1, 1, 0, 1, 0, 1};
    initial_partitioning.remove_degree_zero_hns_before_ip = true;

    // initial partitioning -> refinement
    initial_partitioning.refinement.refine_until_no_improvement = false;

    // initial partitioning -> refinement -> label propagation
    initial_partitioning.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::label_propagation;
    initial_partitioning.refinement.label_propagation.maximum_iterations = 5;
    initial_partitioning.refinement.label_propagation.rebalancing = true;
    initial_partitioning.refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // initial partitioning -> refinement -> fm
    initial_partitioning.refinement.fm.algorithm = FMAlgorithm::do_nothing;

    // initial partitioning -> refinement -> flows
    initial_partitioning.refinement.flows.algorithm = FlowAlgorithm::do_nothing;

    // refinement
    refinement.rebalancer = RebalancingAlgorithm::simple_rebalancer;
    refinement.refine_until_no_improvement = false;

    // refinement -> label propagation
    refinement.label_propagation.algorithm = LabelPropagationAlgorithm::label_propagation;
    refinement.label_propagation.maximum_iterations = 5;
    refinement.label_propagation.rebalancing = true;
    refinement.label_propagation.hyperedge_size_activation_threshold = 100;

    // refinement -> fm
    refinement.fm.algorithm = FMAlgorithm::do_nothing;

    // refinement -> flows
    refinement.flows.algorithm = FlowAlgorithm::do_nothing;
  }

  std::ostream & operator<< (std::ostream& str, const Context& context) {
    str << "*******************************************************************************\n"
        << "*                            Partitioning Context                             *\n"
        << "*******************************************************************************\n"
        << context.partition
        << "-------------------------------------------------------------------------------\n"
        << context.preprocessing
        << "-------------------------------------------------------------------------------\n"
        << context.coarsening
        << "-------------------------------------------------------------------------------\n"
        << context.initial_partitioning
        << "-------------------------------------------------------------------------------\n"
        << context.refinement
        << "-------------------------------------------------------------------------------\n"
        << context.shared_memory
        << "-------------------------------------------------------------------------------";
    return str;
  }
}
