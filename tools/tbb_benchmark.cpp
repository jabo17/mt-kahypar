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
#include <vector>
#include <string>
#include <chrono>

#include "tbb/parallel_sort.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_scan.h"

#include "mt-kahypar/macros.h"
#include "mt-kahypar/utils/randomize.h"
#include "mt-kahypar/parallel/tbb_initializer.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/parallel/parallel_prefix_sum.h"

namespace po = boost::program_options;

using HighResClockTimepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

static double sort_benchmark(size_t N) {
  // Generate random permutation of the numbers from 0 to N
  mt_kahypar::vec<uint32_t> a(N, 0);
  tbb::parallel_for(0UL, N, [&](size_t i) {
    a[i] = i;
  });
  mt_kahypar::utils::Randomize::instance().shuffleVector(a);

  HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
  tbb::parallel_sort(a.begin(), a.end());
  HighResClockTimepoint end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double>(end - start).count();
}

static double reduce_benchmark(size_t N) {
  // Generate bitset of size N
  mt_kahypar::vec<uint32_t> a(N, 0);
  tbb::parallel_for(0UL, N, [&](size_t i) {
    a[i] = mt_kahypar::utils::Randomize::instance().flipCoin(sched_getcpu());
  });

  HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
  size_t total_ones = tbb::parallel_reduce(tbb::blocked_range<size_t>(0UL, N), 0UL,
    [&](const tbb::blocked_range<size_t>& range, size_t init) {
      size_t ones = init;
      for (size_t i = range.begin(); i < range.end(); ++i) {
        ones += a[i];
      }
      return ones;
    }, std::plus<size_t>());
  HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
  unused(total_ones);

  return std::chrono::duration<double>(end - start).count();
}

static double prefix_sum_benchmark(size_t N) {
  // Generate bitset of size N
  mt_kahypar::vec<uint32_t> a(N, 0);
  tbb::parallel_for(0UL, N, [&](size_t i) {
    a[i] = mt_kahypar::utils::Randomize::instance().flipCoin(sched_getcpu());
  });

  HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
  mt_kahypar::parallel::TBBPrefixSum<uint32_t> prefix_sum(a);
  tbb::parallel_scan(tbb::blocked_range<size_t>(0UL, N + 1), prefix_sum);
  HighResClockTimepoint end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double>(end - start).count();
}

static double random_shuffle_benchmark(size_t N) {
  // Generate vector with the numbers from 0 to N
  mt_kahypar::vec<uint32_t> a(N, 0);
  tbb::parallel_for(0UL, N, [&](size_t i) {
    a[i] = i;
  });

  HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
  mt_kahypar::utils::Randomize::instance().parallelShuffleVector(a, 0, a.size());
  HighResClockTimepoint end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char* argv[]) {
  size_t N;
  std::string type;
  int seed;
  int threads;

  po::options_description options("Options");
  options.add_options()
    ("n",
    po::value<size_t>(&N)->value_name("<uint64_t>")->required(),
    "Number of elements")
    ("type",
    po::value<std::string>(&type)->value_name("<string>")->required(),
    "Benchmark Type")
    ("seed",
    po::value<int>(&seed)->value_name("<int>")->required(),
    "Seed value")
    ("num-threads",
    po::value<int>(&threads)->value_name("<int>")->required(),
    "Number of threads");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  mt_kahypar::utils::Randomize::instance().setSeed(seed);
  mt_kahypar::TBBInitializer::instance(threads);


  double time = 0.0;
  if ( type == "sort" ) {
    time = sort_benchmark(N);
  } else if ( type == "reduce" ) {
    time = reduce_benchmark(N);
  } else if ( type == "prefix_sum" ) {
    time = prefix_sum_benchmark(N);
  } else if ( type == "random_shuffle" ) {
    time = random_shuffle_benchmark(N);
  }

  std::cout << "RESULT"
            << " N=" << N
            << " type=" << type
            << " seed=" << seed
            << " threads=" << threads
            << " time=" << time
            << std::endl;

  return 0;
}
