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
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>

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

struct Statistic {
  static constexpr uint64_t num_entries = 7;

  double avg = 0.0;
  double sd = 0.0;
  uint64_t min = 0;
  uint64_t q1 = 0;
  uint64_t med = 0;
  uint64_t q3 = 0;
  uint64_t max = 0;
  // TODO: skew
  // TODO: chi^2 ?

  std::vector<Feature> featureList() {
    std::vector<Feature> result {
      {avg, FeatureType::floatingpoint},
      {sd, FeatureType::floatingpoint},
      {min, FeatureType::integer},
      {q1, FeatureType::integer},
      {med, FeatureType::integer},
      {q3, FeatureType::integer},
      {max, FeatureType::integer},
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
  static constexpr uint64_t num_entries = 2 * Statistic::num_entries + 10;

  uint64_t n = 0;
  uint64_t m = 0;
  double irregularity = 0.0;
  uint64_t exp_median_degree = 0;
  Statistic degree_stats;  // over nodes
  Statistic locality_stats;  // over edges
  uint64_t n_communities_0 = 0;
  uint64_t n_communities_1 = 0;
  uint64_t n_communities_2 = 0;
  double modularity_0 = 0.0;
  double modularity_1 = 0.0;
  double modularity_2 = 0.0;

  std::vector<Feature> featureList() {
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
    std::vector<std::string> degree_header = Statistic::header("degree");
    std::vector<std::string> locality_header = Statistic::header("locality");
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
  static constexpr uint64_t num_entries = 2 * Statistic::num_entries + 8;

  uint64_t degree = 0;
  double degree_quantile = 0;
  Statistic degree_stats;  // over nodes
  Statistic locality_stats;  // over nodes
  uint64_t to_n1_edges = 0;
  double modularity = 0;
  double max_modularity = 0;
  uint64_t max_modularity_size = 0;
  double min_contracted_degree = 0;
  uint64_t min_contracted_degree_size = 0;

  std::vector<Feature> featureList() {
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
    std::vector<std::string> degree_header = Statistic::header("n1_degree");
    std::vector<std::string> locality_header = Statistic::header("n1_locality");
    result_1.insert(result_1.end(), degree_header.begin(), degree_header.end());
    result_1.insert(result_1.end(), locality_header.begin(), locality_header.end());
    std::vector<std::string> result_2 {
      "n1_to_n1_edges", "n1_modularity", "n1_max_modularity",
      "n1_max_modularity_size", "n1_min_contracted_degree", "n1_min_contracted_degree_size",
    };
    result_1.insert(result_1.end(), result_2.begin(), result_2.end());
    ALWAYS_ASSERT(result_1.size() == num_entries, "header info was not properly updated");
    return result_1;
  }
};

struct N2Features {
  static constexpr uint64_t num_entries = 2 * Statistic::num_entries + 4;

  uint64_t size = 0;
  Statistic degree_stats;  // over nodes
  Statistic locality_stats;  // over nodes
  uint64_t to_n1n2_edges = 0;
  uint64_t out_edges = 0;
  double modularity = 0;

  std::vector<Feature> featureList() {
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
    std::vector<std::string> degree_header = Statistic::header("n2_degree");
    std::vector<std::string> locality_header = Statistic::header("n2_locality");
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
Statistic createStats(std::vector<T>& vec, bool parallel) {
  Statistic stats;
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


GlobalFeatures computeGlobalFeatures(const Graph& graph) {
  GlobalFeatures features;

  std::vector<HyperedgeID> hn_degrees;
  hn_degrees.resize(graph.initialNumNodes());
  graph.doParallelForAllNodes([&](const HypernodeID& node) {
    hn_degrees[node] = graph.nodeDegree(node);
    ALWAYS_ASSERT(graph.nodeWeight(node) == 1);
  });

  HypernodeID num_nodes = graph.initialNumNodes();
  HyperedgeID num_edges = Graph::is_graph ? graph.initialNumEdges() / 2 : graph.initialNumEdges();
  graph.doParallelForAllEdges([&](const HyperedgeID& edge) {
    // TODO: only directed edges
    ALWAYS_ASSERT(graph.edgeSize(edge) == 2);
    ALWAYS_ASSERT(graph.edgeWeight(edge) == 1);
    // TODO: locality
  });

  Statistic degree_stats = createStats(hn_degrees, true);
  ASSERT([&]{
    Statistic alt_stats = createStats(hn_degrees, false);
    if (!float_eq(degree_stats.avg, alt_stats.avg)
      || !float_eq(degree_stats.med, alt_stats.med)
      || !float_eq(degree_stats.sd, alt_stats.sd)) {
        return false;
    }
    return true;
  }());

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
  return features;
}




// ################    Main   ################


int main(int argc, char* argv[]) {
  Context context;

  po::options_description options("Options");
  options.add_options()
          ("help", "show help message")
          ("hypergraph,h",
           po::value<std::string>(&context.partition.graph_filename)->value_name("<string>")->required(),
           "Graph Filename")
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

  // Read Hypergraph
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar::io::readInputFile(
      context.partition.graph_filename, PresetType::default_preset,
      InstanceType::graph, context.partition.file_format, true);
  Graph& graph = utils::cast<Graph>(hypergraph);

  GlobalFeatures global_features = computeGlobalFeatures(graph);

  auto header = global_features.header();
  auto features = global_features.featureList();
  ALWAYS_ASSERT(header.size() == features.size());
  printHeader(std::cout, header);
  printFeatures(std::cout, features);
  // std::string graph_name = context.partition.graph_filename.substr(
  //   context.partition.graph_filename.find_last_of("/") + 1);

  utils::delete_hypergraph(hypergraph);

  return 0;
}
