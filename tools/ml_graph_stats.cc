/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Tobias Heuer <tobias.heuer@kit.edu>
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

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <type_traits>

#include "tbb/parallel_sort.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/parallel_reduce.h"

#include "mt-kahypar/macros.h"
#include "mt-kahypar/datastructures/static_graph.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/delete.h"
#include "mt-kahypar/utils/hypergraph_statistics.h"

#include "kahypar-resources/utils/math.h"

#include "neighborhood_computation.h"

using namespace mt_kahypar;
namespace po = boost::program_options;

using Graph = ds::StaticGraph;


enum class FeatureType {
  floatingpoint,
  integer,
  boolean,
};

struct Feature {
  double value;
  FeatureType type;

  Feature(double value, FeatureType type): value(value), type(type) {
    ALWAYS_ASSERT(type == FeatureType::floatingpoint);
  }
  Feature(uint64_t value, FeatureType type): value(static_cast<double>(value)), type(type) {
    ALWAYS_ASSERT(type != FeatureType::floatingpoint);
  }
};


void printHeader(std::ostream& os, const std::vector<std::string>& header) {
  for (size_t i = 0; i < header.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << header[i];
  }
  os << std::endl;
}

void printFeatures(std::ostream& os, const std::vector<Feature>& features) {
  for (size_t i = 0; i < features.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    if (features[i].type != FeatureType::floatingpoint) {
      os << static_cast<uint64_t>(features[i].value);
    } else {
      os << std::setprecision(4) << features[i].value;
    }
  }
  os << std::endl;
}


// ################ Feature Definitions ################

struct Invalid {};

template<typename T = Invalid>
struct Statistic {
  static constexpr uint64_t num_entries = 7;

  double avg = 0.0;
  double sd = 0.0;
  T min = 0;
  T q1 = 0;
  T med = 0;
  T q3 = 0;
  T max = 0;
  // TODO: skew
  // TODO: chi^2 ?

  std::vector<Feature> featureList() const {
    FeatureType type = std::is_floating_point_v<T> ? FeatureType::floatingpoint : FeatureType::integer;
    std::vector<Feature> result {
      {avg, FeatureType::floatingpoint},
      {sd, FeatureType::floatingpoint},
      {min, type},
      {q1, type},
      {med, type},
      {q3, type},
      {max, type},
    };
    ALWAYS_ASSERT(result.size() == num_entries, "header info was not properly updated");
    return result;
  }

  static std::vector<std::string> header(const char* suffix) {
    std::vector<std::string> result {
      std::string("avg_") + suffix,
      std::string("sd_") + suffix,
      std::string("min_") + suffix,
      std::string("q1_") + suffix,
      std::string("med_") + suffix,
      std::string("q3_") + suffix,
      std::string("max_") + suffix,
    };
    ALWAYS_ASSERT(result.size() == num_entries, "header info was not properly updated");
    return result;
  }
};

struct GlobalFeatures {
  static constexpr uint64_t num_entries = 2 * Statistic<>::num_entries + 10;

  uint64_t n = 0;
  uint64_t m = 0;
  double irregularity = 0.0;
  uint64_t exp_median_degree = 0;
  Statistic<uint64_t> degree_stats;  // over nodes
  Statistic<double> locality_stats;  // over edges
  uint64_t n_communities_0 = 0;
  uint64_t n_communities_1 = 0;
  uint64_t n_communities_2 = 0;
  double modularity_0 = 0.0;
  double modularity_1 = 0.0;
  double modularity_2 = 0.0;

  std::vector<Feature> featureList() const {
    std::vector<Feature> result_1 {
      {n, FeatureType::integer},
      {m, FeatureType::integer},
      {irregularity, FeatureType::floatingpoint},
      {exp_median_degree, FeatureType::integer},
    };
    std::vector<Feature> degree_features = degree_stats.featureList();
    std::vector<Feature> locality_features = locality_stats.featureList();
    result_1.insert(result_1.end(), degree_features.begin(), degree_features.end());
    result_1.insert(result_1.end(), locality_features.begin(), locality_features.end());
    std::vector<Feature> result_2 {
      {n_communities_0, FeatureType::integer},
      {n_communities_1, FeatureType::integer},
      {n_communities_2, FeatureType::integer},
      {modularity_0, FeatureType::floatingpoint},
      {modularity_1, FeatureType::floatingpoint},
      {modularity_2, FeatureType::floatingpoint},
    };
    result_1.insert(result_1.end(), result_2.begin(), result_2.end());
    ALWAYS_ASSERT(result_1.size() == num_entries, "header info was not properly updated");
    return result_1;
  }

  static std::vector<std::string> header() {
    std::vector<std::string> result_1 {"n", "m", "irregularity", "exp_median_degree"};
    std::vector<std::string> degree_header = Statistic<>::header("degree");
    std::vector<std::string> locality_header = Statistic<>::header("locality");
    result_1.insert(result_1.end(), degree_header.begin(), degree_header.end());
    result_1.insert(result_1.end(), locality_header.begin(), locality_header.end());
    std::vector<std::string> result_2 {
      "n_communities_0", "n_communities_1", "n_communities_2",
      "modularity_0", "modularity_1", "modularity_2",
    };
    result_1.insert(result_1.end(), result_2.begin(), result_2.end());
    ALWAYS_ASSERT(result_1.size() == num_entries, "header info was not properly updated");
    return result_1;
  }
};

struct N1Features {
  static constexpr uint64_t num_entries = 2 * Statistic<>::num_entries + 10;

  uint64_t degree = 0;
  double degree_quantile = 0;
  Statistic<uint64_t> degree_stats;  // over nodes
  Statistic<double> locality_stats;  // over nodes
  uint64_t to_n1_edges = 0;
  uint64_t to_n2_edges = 0;
  uint64_t d1_nodes = 0;
  double modularity = 0;
  double max_modularity = 0;
  uint64_t max_modularity_size = 0;
  double min_contracted_degree = 0;
  uint64_t min_contracted_degree_size = 0;
  // TODO: community overlap?

  std::vector<Feature> featureList() const {
    std::vector<Feature> result_1 {
      {degree, FeatureType::integer},
      {degree_quantile, FeatureType::floatingpoint},
    };
    std::vector<Feature> degree_features = degree_stats.featureList();
    std::vector<Feature> locality_features = locality_stats.featureList();
    result_1.insert(result_1.end(), degree_features.begin(), degree_features.end());
    result_1.insert(result_1.end(), locality_features.begin(), locality_features.end());
    std::vector<Feature> result_2 {
      {to_n1_edges, FeatureType::integer},
      {to_n2_edges, FeatureType::integer},
      {d1_nodes, FeatureType::integer},
      {modularity, FeatureType::floatingpoint},
      {max_modularity, FeatureType::floatingpoint},
      {max_modularity_size, FeatureType::integer},
      {min_contracted_degree, FeatureType::floatingpoint},
      {min_contracted_degree_size, FeatureType::integer},
    };
    result_1.insert(result_1.end(), result_2.begin(), result_2.end());
    ALWAYS_ASSERT(result_1.size() == num_entries, "header info was not properly updated");
    return result_1;
  }

  static std::vector<std::string> header() {
    std::vector<std::string> result_1 {"degree", "degree_quantile"};
    std::vector<std::string> degree_header = Statistic<>::header("n1_degree");
    std::vector<std::string> locality_header = Statistic<>::header("n1_locality");
    result_1.insert(result_1.end(), degree_header.begin(), degree_header.end());
    result_1.insert(result_1.end(), locality_header.begin(), locality_header.end());
    std::vector<std::string> result_2 {
      "n1_to_n1_edges", "n1_to_n2_edges", "n1_d1_nodes", "n1_modularity", "n1_max_modularity",
      "n1_max_modularity_size", "n1_min_contracted_degree", "n1_min_contracted_degree_size",
    };
    result_1.insert(result_1.end(), result_2.begin(), result_2.end());
    ALWAYS_ASSERT(result_1.size() == num_entries, "header info was not properly updated");
    return result_1;
  }
};

struct N2Features {
  static constexpr uint64_t num_entries = 2 * Statistic<>::num_entries + 4;

  uint64_t size = 0;
  Statistic<uint64_t> degree_stats;  // over nodes
  Statistic<double> locality_stats;  // over nodes
  uint64_t to_n1n2_edges = 0;
  uint64_t out_edges = 0;
  double modularity = 0;
  // TODO: community overlap?

  std::vector<Feature> featureList() const {
    std::vector<Feature> result_1 { {size, FeatureType::integer} };
    std::vector<Feature> degree_features = degree_stats.featureList();
    std::vector<Feature> locality_features = locality_stats.featureList();
    result_1.insert(result_1.end(), degree_features.begin(), degree_features.end());
    result_1.insert(result_1.end(), locality_features.begin(), locality_features.end());
    std::vector<Feature> result_2 {
      {to_n1n2_edges, FeatureType::integer},
      {out_edges, FeatureType::integer},
      {modularity, FeatureType::floatingpoint},
    };
    result_1.insert(result_1.end(), result_2.begin(), result_2.end());
    ALWAYS_ASSERT(result_1.size() == num_entries, "header info was not properly updated");
    return result_1;
  }

  static std::vector<std::string> header() {
    std::vector<std::string> result_1 {"n2_size"};
    std::vector<std::string> degree_header = Statistic<>::header("n2_degree");
    std::vector<std::string> locality_header = Statistic<>::header("n2_locality");
    result_1.insert(result_1.end(), degree_header.begin(), degree_header.end());
    result_1.insert(result_1.end(), locality_header.begin(), locality_header.end());
    std::vector<std::string> result_2 {"n2_to_n1n2_edges", "n2_out_edges", "n2_modularity"};
    result_1.insert(result_1.end(), result_2.begin(), result_2.end());
    ALWAYS_ASSERT(result_1.size() == num_entries, "header info was not properly updated");
    return result_1;
  }
};



// ################ Feature Computation ################


template <typename T>
Statistic<T> createStats(std::vector<T>& vec, bool parallel) {
  Statistic<T> stats;
  if (!vec.empty()) {
    double avg = 0;
    double stdev = 0;
    if (parallel) {
      tbb::parallel_sort(vec.begin(), vec.end());
      avg = utils::parallel_avg(vec, vec.size());
      stdev = utils::parallel_stdev(vec, avg, vec.size());
    } else {
      std::sort(vec.begin(), vec.end());
      for (auto val: vec) {
        avg += val;
      }
      avg = avg / static_cast<double>(vec.size());
      for (auto val: vec) {
        stdev += (val - avg) * (val - avg);
      }
      stdev = std::sqrt(stdev / static_cast<double>(vec.size()));
    }
    const auto quartiles = kahypar::math::firstAndThirdQuartile(vec);
    stats.min = vec.front();
    stats.q1 = quartiles.first;
    stats.med = kahypar::math::median(vec);
    stats.q3 = quartiles.second;
    stats.max = vec.back();
    stats.avg = avg;
    stats.sd = stdev;
  }
  return stats;
}


bool float_eq(double left, double right) {
  return left <= 1.001 * right && right <= 1.001 * left;
}


std::pair<GlobalFeatures, std::vector<uint64_t>> computeGlobalFeatures(const Graph& graph) {
  GlobalFeatures features;

  std::vector<uint64_t> hn_degrees;
  hn_degrees.resize(graph.initialNumNodes());
  graph.doParallelForAllNodes([&](const HypernodeID& node) {
    hn_degrees[node] = graph.nodeDegree(node);
    ALWAYS_ASSERT(graph.nodeWeight(node) == 1);
  });

  HypernodeID num_nodes = graph.initialNumNodes();
  HyperedgeID num_edges = Graph::is_graph ? graph.initialNumEdges() / 2 : graph.initialNumEdges();
  graph.doParallelForAllEdges([&](const HyperedgeID& edge) {
    ALWAYS_ASSERT(graph.edgeSize(edge) == 2);
    ALWAYS_ASSERT(graph.edgeWeight(edge) == 1);
    // TODO: locality, only directed edges
  });

  Statistic degree_stats = createStats(hn_degrees, true);
  // ASSERT([&]{
  //   Statistic alt_stats = createStats(hn_degrees, false);
  //   if (!float_eq(degree_stats.avg, alt_stats.avg)
  //     || !float_eq(degree_stats.med, alt_stats.med)
  //     || !float_eq(degree_stats.sd, alt_stats.sd)) {
  //       return false;
  //   }
  //   return true;
  // }());

  features.n = num_nodes;
  features.m = num_edges;
  features.degree_stats = degree_stats;
  features.irregularity = degree_stats.sd / degree_stats.avg;

  // compute exp_median_degree via suffix sum
  uint64_t pins = graph.initialNumPins();
  uint64_t count = 0;
  for (size_t i = num_nodes; i > 0; --i) {
    count += hn_degrees[i-1];
    if (count >= pins / 2) {
      features.exp_median_degree = hn_degrees[i-1];
      break;
    }
  }

  // TODO: modularity
  return {features, hn_degrees};
}

N1Features n1FeaturesFromNeighborhood(const Graph& graph, const std::vector<uint64_t>& global_degrees, const NeighborhoodResult& data) {
  N1Features result;
  HypernodeID num_nodes = data.n1_list.size();
  result.degree = num_nodes;
  size_t lower = std::lower_bound(global_degrees.begin(), global_degrees.end(), result.degree) - global_degrees.begin();  // always < n
  size_t upper = std::upper_bound(global_degrees.begin(), global_degrees.end(), result.degree) - global_degrees.begin() - 1;  // always >= 1
  result.degree_quantile = (static_cast<double>(lower) + static_cast<double>(upper)) / (2 * static_cast<double>(global_degrees.size() - 1));

  // compute degree stats
  std::vector<uint64_t> degrees;
  degrees.reserve(num_nodes);
  for (HypernodeID node: data.n1_list) {
    degrees.push_back(graph.nodeDegree(node));
  }
  result.degree_stats = createStats(degrees, degrees.size() >= 20000);

  // TODO: edges between neighbors are ignored in the same way as in
  // the coarsening heuristic. Is this what we want?
  HyperedgeWeight out_edges = result.degree;
  HypernodeWeight node_weight = 1;
  for (HyperedgeID d: degrees) {
    if ((static_cast<HyperedgeWeight>(d) - 2) * node_weight < out_edges) {
      out_edges += d - 2;
      node_weight++;
    } 
  }
  result.min_contracted_degree = static_cast<double>(out_edges) / static_cast<double>(node_weight);
  result.min_contracted_degree_size = node_weight;

  // compute locality stats and related values
  std::vector<double> locality_values;
  locality_values.reserve(num_nodes);
  for (HypernodeID node: data.n1_list) {
    uint64_t local_n1_edges = 0;
    for (HyperedgeID edge: graph.incidentEdges(node)) {
      HypernodeID neighbor = graph.edgeTarget(edge);
      if (data.isInN1Exactly(neighbor)) {
        result.to_n1_edges++;
        local_n1_edges++;
      } else if (!data.isRoot(neighbor)) {
        result.to_n2_edges++;
      }
    }
    HypernodeID node_degree = graph.nodeDegree(node);
    uint64_t divisor = std::min(num_nodes, node_degree) - 1;
    if (divisor != 0) {
      locality_values.push_back(static_cast<double>(local_n1_edges) / static_cast<double>(divisor));
    } else if (node_degree == 1) {
      result.d1_nodes++;
    }
  }
  result.to_n1_edges /= 2;  // doubly counted
  result.locality_stats = createStats(locality_values, locality_values.size() >= 20000);
  // TODO: modularity, community overlap?
  return result;
}

N2Features n2FeaturesFromNeighborhood(const Graph& graph, const NeighborhoodResult& data) {
  ALWAYS_ASSERT(data.includes_two_hop);
  N2Features result;
  HypernodeID num_nodes = data.n2_list.size();
  result.size = num_nodes;

  // compute degree stats
  std::vector<uint64_t> degrees;
  degrees.reserve(num_nodes);
  for (HypernodeID node: data.n2_list) {
    degrees.push_back(graph.nodeDegree(node));
  }
  result.degree_stats = createStats(degrees, degrees.size() >= 20000);

  // compute locality stats and related values
  std::vector<double> locality_values;
  locality_values.reserve(num_nodes);
  for (HypernodeID node: data.n2_list) {
    uint64_t local_edges = 0;
    for (HyperedgeID edge: graph.incidentEdges(node)) {
      HypernodeID neighbor = graph.edgeTarget(edge);
      if (data.isInN2(neighbor)) {
        result.to_n1n2_edges++;
        local_edges++;
        if (!data.isInN2Exactly(neighbor)) {
          result.to_n1n2_edges++;
        }
      } else {
        result.out_edges++;
      }
    }
    HypernodeID node_degree = graph.nodeDegree(node);
    uint64_t divisor = std::min(num_nodes + static_cast<HypernodeID>(data.n1_list.size()), node_degree);
    ALWAYS_ASSERT(divisor != 0);
    locality_values.push_back(static_cast<double>(local_edges) / static_cast<double>(divisor));
  }
  result.to_n1n2_edges /= 2;  // doubly counted
  result.locality_stats = createStats(locality_values, locality_values.size() >= 20000);
  // TODO: modularity, community overlap?
  return result;
}


std::vector<std::tuple<HypernodeID, N1Features, N2Features>>  computeNodeFeatures(const Graph& graph, const std::vector<uint64_t>& global_degrees) {
  std::vector<std::tuple<HypernodeID, N1Features, N2Features>> result;
  result.resize(graph.initialNumNodes());

  tbb::enumerable_thread_specific<NeighborhoodComputation> computations(graph.initialNumNodes());
  graph.doParallelForAllNodes([&](HypernodeID node) {
    NeighborhoodComputation& local_compute = computations.local();
    local_compute.reset();
    NeighborhoodResult neighborhood = local_compute.computeNeighborhood(graph, std::array{node}, true);
    N1Features n1_features = n1FeaturesFromNeighborhood(graph, global_degrees, neighborhood);
    N2Features n2_features = n2FeaturesFromNeighborhood(graph, neighborhood);
    result[node] = {node, n1_features, n2_features};
  });

  return result;
}


// ################    Main   ################


int main(int argc, char* argv[]) {
  Context context;
  std::string global_out;
  std::string nodes_out;

  po::options_description options("Options");
  options.add_options()
          ("help", "show help message")
          ("hypergraph,h",
           po::value<std::string>(&context.partition.graph_filename)->value_name("<string>")->required(),
           "Graph Filename")
          ("global,g",
           po::value<std::string>(&global_out)->value_name("<string>")->required(),
           "Output file for global features")
          ("nodes,n",
           po::value<std::string>(&nodes_out)->value_name("<string>")->required(),
           "Output file for node features")
          ("input-file-format",
            po::value<std::string>()->value_name("<string>")->notifier([&](const std::string& s) {
              if (s == "hmetis") {
                context.partition.file_format = FileFormat::hMetis;
              } else if (s == "metis") {
                context.partition.file_format = FileFormat::Metis;
              }
            })->default_value("metis"),
            "Input file format: \n"
            " - hmetis : hMETIS hypergraph file format \n"
            " - metis : METIS graph file format");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  if (cmd_vm.count("help") != 0 || argc == 1) {
    LOG << options;
    exit(0);
  }
  po::notify(cmd_vm);

  std::ofstream global(global_out);
  std::ofstream nodes(nodes_out);

  // Read Hypergraph
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar::io::readInputFile(
      context.partition.graph_filename, PresetType::default_preset,
      InstanceType::graph, context.partition.file_format, true);
  Graph& graph = utils::cast<Graph>(hypergraph);

  auto [global_features, degrees] = computeGlobalFeatures(graph);
  tbb::parallel_invoke([&]{
    auto header = global_features.header();
    auto features = global_features.featureList();
    ALWAYS_ASSERT(header.size() == features.size());
    printHeader(global, header);
    printFeatures(global, features);
  }, [&]{
    auto node_features = computeNodeFeatures(graph, degrees);
    std::vector<std::string> header {"node_id"};
    std::vector<std::string> n1_header = N1Features::header();
    header.insert(header.end(), n1_header.begin(), n1_header.end());
    std::vector<std::string> n2_header = N2Features::header();
    header.insert(header.end(), n2_header.begin(), n2_header.end());
    printHeader(nodes, header);

    std::vector<Feature> features;
    for (const auto& [id, n1_features, n2_features]: node_features) {
      features.clear();
      features.emplace_back(static_cast<uint64_t>(id), FeatureType::integer);  // this is not actually a feature
      std::vector<Feature> n1 = n1_features.featureList();
      features.insert(features.end(), n1.begin(), n1.end());
      std::vector<Feature> n2 = n2_features.featureList();
      features.insert(features.end(), n2.begin(), n2.end());
      ALWAYS_ASSERT(header.size() == features.size());
      printFeatures(nodes, features);
    }
  });
  // std::string graph_name = context.partition.graph_filename.substr(
  //   context.partition.graph_filename.find_last_of("/") + 1);

  utils::delete_hypergraph(hypergraph);

  return 0;
}
