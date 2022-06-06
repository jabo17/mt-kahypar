/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2019 Sebastian Schlag <tobias.heuer@kit.edu>
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

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "kahypar/definitions.h"
#include "kahypar/io/hypergraph_io.h"

#include "mt-kahypar/macros.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/partition/factories.h"

using namespace mt_kahypar;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  Context context;

  po::options_description options("Options");
  options.add_options()
          ("hypergraph,h",
           po::value<std::string>(&context.partition.graph_filename)->value_name("<string>")->required(),
           "Hypergraph Filename")
          ("seed",
          po::value<int>(&context.partition.seed)->value_name("<int>")->required(),
          "Seed value");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  context.partition.file_format = FileFormat::hMetis;
  context.coarsening.algorithm = CoarseningAlgorithm::nlevel_coarsener;
  context.partition.k = 2;
  context.partition.epsilon = 0.03;
  context.coarsening.minimum_shrink_factor = 1.01;
  context.coarsening.maximum_shrink_factor = 100.0;
  context.coarsening.contraction_limit_multiplier = 160;
  context.coarsening.max_allowed_weight_multiplier = 1;
  context.coarsening.vertex_degree_sampling_threshold = 200000;
  context.coarsening.rating.rating_function = RatingFunction::heavy_edge;
  context.coarsening.rating.heavy_node_penalty_policy = HeavyNodePenaltyPolicy::no_penalty;
  context.coarsening.rating.acceptance_policy = AcceptancePolicy::best_prefer_unmatched;

  mt_kahypar::TBBInitializer::instance(std::thread::hardware_concurrency());
  mt_kahypar::utils::Randomize::instance().setSeed(context.partition.seed);

  vec<Memento> contractions;
  {
    // Read Hypergraph
    Hypergraph hg = mt_kahypar::io::readInputFile(
      context.partition.graph_filename, context.partition.file_format, true, false);
    context.setupPartWeights(hg.totalWeight());
    context.setupContractionLimit(hg.totalWeight());
    context.setupMaximumAllowedNodeWeight(hg.totalWeight());
    UncoarseningData uncoarseningData(true, hg, context);
    std::unique_ptr<ICoarsener> coarsener = CoarsenerFactory::getInstance().createObject(
            context.coarsening.algorithm, hg, context, uncoarseningData);
    coarsener->coarsen();
    vec<vec<vec<Memento>>> batches = hg.createBatchUncontractionHierarchy(1000);
    for ( const vec<vec<Memento>>& b_1 : batches  ) {
      for ( const vec<Memento>& b_2 : b_1 ) {
        for ( const Memento& c : b_2 ) {
          contractions.push_back(c);
        }
      }
    }
  }

  double time_mt_kahypar = 0.0;
  {
    // Read Hypergraph
    Hypergraph hg = mt_kahypar::io::readInputFile(
      context.partition.graph_filename, context.partition.file_format, true, false);
    HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
    for ( const Memento& m : contractions ) {
      hg.registerContraction(m.u, m.v);
      hg.contract(m.v, context.coarsening.max_allowed_node_weight);
    }
    HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
    time_mt_kahypar = std::chrono::duration<double>(end - start).count();
  }

  double time_mt_kahypar_opt = 0.0;
  {
    // Read Hypergraph
    Hypergraph hg = mt_kahypar::io::readInputFile(
      context.partition.graph_filename, context.partition.file_format, true, false);
    HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
    for ( const Memento& m : contractions ) {
      hg.contract(m.u, m.v);
    }
    HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
    time_mt_kahypar_opt = std::chrono::duration<double>(end - start).count();
  }

  double time_kahypar = 0.0;
  {
    // Read Hypergraph
    kahypar::Hypergraph hg(
      kahypar::io::createHypergraphFromFile(context.partition.graph_filename,
                                            context.partition.k));
    HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
    for ( const Memento& m : contractions ) {
      hg.contract(m.u, m.v);
    }
    HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
    time_kahypar = std::chrono::duration<double>(end - start).count();
  }

  std::string graph_name = context.partition.graph_filename.substr(
    context.partition.graph_filename.find_last_of("/") + 1);
  std::cout  << "RESULT graph=" << graph_name
             << " seed=" << context.partition.seed
             << " mt_kahypar_time=" << time_mt_kahypar
             << " mt_kahypar_time_opt=" << time_mt_kahypar_opt
             << " kahypar_time=" << time_kahypar
             << std::endl;


  return 0;
}
