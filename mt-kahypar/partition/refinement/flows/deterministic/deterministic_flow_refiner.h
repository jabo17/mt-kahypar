/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Tobias Heuer <tobias.heuer@kit.edu>
 * Copyright (C) 2021 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#pragma once

#include "algorithm/hyperflowcutter.h"
#include "algorithm/sequential_push_relabel.h"
#include "algorithm/parallel_push_relabel.h"

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/flows/sequential_construction.h"
#include "mt-kahypar/partition/refinement/flows/parallel_construction.h"
#include "mt-kahypar/partition/refinement/flows/deterministic/deterministic_quotient_graph.h"
#include "mt-kahypar/partition/refinement/flows/deterministic/deterministic_problem_construction.h"
#include "mt-kahypar/utils/cast.h"

#include "external_tools/WHFC/io/hmetis_io.h"
#include "external_tools/WHFC/io/whfc_io.h"


namespace mt_kahypar {

template<typename GraphAndGainTypes>
class DeterministicFlowRefiner {

    static constexpr bool debug = false;
    static constexpr bool sequential = false;

    using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
    using TypeTraits = typename GraphAndGainTypes::TypeTraits;
public:
    explicit DeterministicFlowRefiner(const HypernodeID num_hypernodes, const HyperedgeID num_hyperedges,
        const Context& context) :
        _context(context),
        _flow_hg(),
        _sequential_hfc(_flow_hg, context.partition.seed),
        _parallel_hfc(_flow_hg, context.partition.seed, true),
        _sequential_construction(num_hyperedges, _flow_hg, _sequential_hfc, context),
        _parallel_construction(num_hyperedges, _flow_hg, _parallel_hfc, context),
        _problem_construction(num_hypernodes, num_hyperedges, context),
        _whfc_to_node() {
        _sequential_hfc.find_most_balanced = _context.refinement.flows.find_most_balanced_cut;
        _sequential_hfc.timer.active = false;
        _sequential_hfc.forceSequential(true);
        _sequential_hfc.setBulkPiercing(context.refinement.flows.pierce_in_bulk);

        _parallel_hfc.find_most_balanced = _context.refinement.flows.find_most_balanced_cut;
        _parallel_hfc.timer.active = false;
        _parallel_hfc.forceSequential(false);
        _parallel_hfc.setBulkPiercing(context.refinement.flows.pierce_in_bulk);
    }


    DeterministicFlowRefiner(const DeterministicFlowRefiner&) = delete;
    DeterministicFlowRefiner(DeterministicFlowRefiner&&) = delete;
    DeterministicFlowRefiner& operator= (const DeterministicFlowRefiner&) = delete;
    DeterministicFlowRefiner& operator= (DeterministicFlowRefiner&&) = delete;

    virtual ~DeterministicFlowRefiner() = default;

    void initialize(PartitionedHypergraph&) {
        _flow_hg.clear();
        _whfc_to_node.clear();
    }

    MoveSequence refine(PartitionedHypergraph& phg, DeterministicQuotientGraph<TypeTraits>& quotientGraph, const PartitionID block0, const PartitionID block1, const size_t seed) {
        return refineImpl(phg, quotientGraph, block0, block1, seed);
    }


private:

    MoveSequence refineImpl(PartitionedHypergraph& phg,
        DeterministicQuotientGraph<TypeTraits>& quotientGraph,
        const PartitionID block0,
        const PartitionID block1,
        const size_t seed) {
        //utils::Timer& timer = utils::Utilities::instance().getTimer(_context.utility_id);
        MoveSequence sequence{ { }, 0 };
        //timer.start_timer("problem_construction", "Problem Construction", true);
        Subhypergraph sub_hg = _problem_construction.construct(phg, quotientGraph, block0, block1);
        //timer.stop_timer("problem_construction");
        // if (block0 == 48 && block1 == 51 && seed == 3563114499) {
        //     std::cout << sub_hg.block_0 << std::endl;
        //     std::cout << sub_hg.block_1 << std::endl;
        //     for (auto a : sub_hg.nodes_of_block_0) {
        //         std::cout << a << ",";
        //     }
        //     std::cout << std::endl;
        //     for (auto a : sub_hg.nodes_of_block_1) {
        //         std::cout << a << ",";
        //     }
        //     std::cout << std::endl;
        //     std::cout << sub_hg.weight_of_block_0 << std::endl;
        //     std::cout << sub_hg.weight_of_block_1 << std::endl;
        //     for (auto a : sub_hg.hes) {
        //         std::cout << a << ",";
        //     }
        //     std::cout << std::endl;
        //     std::cout << sub_hg.num_pins << std::endl;
        // }
        if (sub_hg.numNodes() > 0) {

            // TODO: Decide wether to use sequential or parallel problem construction?
            //timer.start_timer("construct_flow_hypergraph", "Flow Hypergraph Construction", true);
            FlowProblem flow_problem;
            if (sequential) {
                flow_problem = _sequential_construction.constructFlowHypergraph(phg, sub_hg, block0, block1, _whfc_to_node);
            } else {
                flow_problem = _parallel_construction.constructFlowHypergraph(phg, sub_hg, block0, block1, _whfc_to_node, true /*deterministic*/);
                // FlowHypergraphBuilder fg = _flow_hg;
                // vec<HypernodeID> whfc2 = _whfc_to_node;
                // std::vector<int32_t> distances = _parallel_hfc.cs.border_nodes.distance;
                // for (int i = 0; i < 10 && block0 == 37 && block1 == 39 && seed == 682769175; ++i) {
                //     _flow_hg.clear();
                //     _whfc_to_node.clear();
                //     FlowProblem flow_problem2 = _parallel_construction.constructFlowHypergraph(phg, sub_hg, block0, block1, _whfc_to_node, true /*deterministic*/);
                //     if (!compareFlowProblems(flow_problem, flow_problem2)) {
                //         std::cout << "FLOW-PROBLEM" << std::endl;
                //         exit(-1);
                //     }
                //     if (!compareFlowHypergraphs(_flow_hg, fg)) {
                //         std::cout << "Flow-Hypergraph" << std::endl;
                //         std::cout << "?????????????????????????????????????????????????????????????????" << std::endl;
                //         _flow_hg.printHypergraph(std::cout);
                //         std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                //         fg.printHypergraph(std::cout);
                //         exit(-1);
                //     }
                //     if (_whfc_to_node != whfc2) {
                //         std::cout << "WHFC " << std::endl;
                //         exit(-1);
                //     }
                //     if (_parallel_hfc.cs.border_nodes.distance != distances) {
                //         std::cout << "Flow-Hypergraph" << std::endl;
                //         exit(-1);
                //     }
                // }
            }
            if (flow_problem.total_cut - flow_problem.non_removable_cut > 0) {
                //timer.stop_timer("construct_flow_hypergraph");
                DBG << V(block0) << V(block1) << V(flow_problem.non_removable_cut) << ", " << V(flow_problem.sink) << ", " << V(flow_problem.source) << ", " << V(flow_problem.total_cut) << ", " << V(flow_problem.weight_of_block_0) << ", " << V(flow_problem.weight_of_block_1);
                //timer.start_timer("run_flow_cutter", "Run Flow Cutter", true);
                bool flowcutter_succeeded = runFlowCutter(flow_problem, block0, block1, seed);
                //timer.stop_timer("run_flow_cutter");
                DBG << V(flowcutter_succeeded) << V(block0) << V(block1);
                if (flowcutter_succeeded) {
                    //timer.start_timer("extract_move_sequence", "Extract Move Sequence", true);
                    extractMoveSequence(phg, flow_problem, sequence, block0, block1);
                    //timer.stop_timer("extract_move_sequence");
                }
            }
        }
        return sequence;

    }

    bool runFlowCutter(FlowProblem& flow_problem, const PartitionID block0, const PartitionID block1, const size_t seed) {
        whfc::Node s = flow_problem.source;
        whfc::Node t = flow_problem.sink;
        if (sequential) {
            _sequential_hfc.reset();

            _sequential_hfc.cs.setMaxBlockWeight(0, std::max(
                flow_problem.weight_of_block_0, _context.partition.max_part_weights[block0]));
            _sequential_hfc.cs.setMaxBlockWeight(1, std::max(
                flow_problem.weight_of_block_1, _context.partition.max_part_weights[block1]));

            _sequential_hfc.setSeed(seed);
            _sequential_hfc.setFlowBound(flow_problem.total_cut - flow_problem.non_removable_cut);
            return _sequential_hfc.enumerateCutsUntilBalancedOrFlowBoundExceeded(s, t);
        } else {
            _parallel_hfc.reset();

            _parallel_hfc.cs.setMaxBlockWeight(0, std::max(
                flow_problem.weight_of_block_0, _context.partition.max_part_weights[block0]));
            _parallel_hfc.cs.setMaxBlockWeight(1, std::max(
                flow_problem.weight_of_block_1, _context.partition.max_part_weights[block1]));

            _parallel_hfc.setSeed(seed);
            _parallel_hfc.setFlowBound(flow_problem.total_cut - flow_problem.non_removable_cut);
            if constexpr (debug) {

                if (block0 == 37 && block1 == 39 && seed == 682769175) {
                    print(_flow_hg);

                    for (auto a : _parallel_hfc.cs.border_nodes.distance) {
                        std::cout << a << ", ";

                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                    // whfc::WHFC_IO::WHFCInformation infos;
                    // infos.maxBlockWeight = { std::max(
                    // flow_problem.weight_of_block_0, _context.partition.max_part_weights[block0]), std::max(
                    // flow_problem.weight_of_block_1, _context.partition.max_part_weights[block1]) };
                    // infos.s = whfc::Node(block0);
                    // infos.t = whfc::Node(block1);
                    // infos.upperFlowBound = (flow_problem.total_cut - flow_problem.non_removable_cut);
                    std::string pathtohg = "/home/robert/KIT/Master/mt-kahypar/external_tools/WHFC/build/hypergraph";

                    //whfc::WHFC_IO::writeAdditionalInformation(pathtohg, infos, _parallel_hfc.cs.rng);

                    // std::string name = std::to_string(block0) + "-" + std::to_string(block1) + "-" + std::to_string(std::max(flow_problem.weight_of_block_0, _context.partition.max_part_weights[block0])) + "-" + std::to_string(std::max(
                    //     flow_problem.weight_of_block_1, _context.partition.max_part_weights[block1])) + "-" + std::to_string(flow_problem.total_cut - flow_problem.non_removable_cut) + "-" + std::to_string(seed);
                    // whfc::HMetisIO::writeFlowHypergraph(_flow_hg, name);
                    buildFromFlowHypergraph(whfc::HMetisIO::readFlowHypergraph(pathtohg));
                    auto result = _parallel_hfc.enumerateCutsUntilBalancedOrFlowBoundExceeded(s, t);
                    _sequential_hfc.reset();
                    _sequential_hfc.cs.border_nodes.distance = _parallel_hfc.cs.border_nodes.distance;
                    _sequential_hfc.cs.setMaxBlockWeight(0, std::max(
                        flow_problem.weight_of_block_0, _context.partition.max_part_weights[block0]));
                    _sequential_hfc.cs.setMaxBlockWeight(1, std::max(
                        flow_problem.weight_of_block_1, _context.partition.max_part_weights[block1]));

                    _sequential_hfc.setSeed(seed);
                    _sequential_hfc.setFlowBound(flow_problem.total_cut - flow_problem.non_removable_cut);
                    _sequential_hfc.enumerateCutsUntilBalancedOrFlowBoundExceeded(s, t);
                    std::cout << _sequential_hfc.cs.toString() << std::endl;
                    std::cout << _parallel_hfc.cs.toString() << std::endl;
                    exit(-1);
                }
            }
            // std::cout << name << std::endl;
            return _parallel_hfc.enumerateCutsUntilBalancedOrFlowBoundExceeded(s, t);
        }
    }

    void extractMoveSequence(const PartitionedHypergraph& phg, const FlowProblem& flow_problem, MoveSequence& sequence, const PartitionID block0, const PartitionID block1) {
        // We apply the solution if it either improves the cut or the balance of
        // the bipartition induced by the two blocks
        DBG << V(flow_problem.non_removable_cut) << ", " << V(flow_problem.total_cut) << ", " << V(flow_problem.weight_of_block_0) << ", " << V(flow_problem.weight_of_block_1);
        HyperedgeWeight new_cut = flow_problem.non_removable_cut;
        HypernodeWeight max_part_weight;
        if (sequential) {
            new_cut += _sequential_hfc.cs.flow_algo.flow_value;
            max_part_weight = std::max(_sequential_hfc.cs.source_weight, _sequential_hfc.cs.target_weight);
        } else {
            new_cut += _parallel_hfc.cs.flow_algo.flow_value;
            max_part_weight = std::max(_parallel_hfc.cs.source_weight, _parallel_hfc.cs.target_weight);
        }
        const bool improved_solution = new_cut < flow_problem.total_cut ||
            (new_cut == flow_problem.total_cut && max_part_weight < std::max(flow_problem.weight_of_block_0, flow_problem.weight_of_block_1));

        // Extract move sequence
        if (improved_solution) {
            sequence.expected_improvement = flow_problem.total_cut - new_cut;
            for (const whfc::Node& u : _flow_hg.nodeIDs()) {
                const HypernodeID hn = _whfc_to_node[u];
                if (hn != kInvalidHypernode) {
                    const PartitionID from = phg.partID(hn);
                    PartitionID to;
                    if (sequential) {
                        to = _sequential_hfc.cs.flow_algo.isSource(u) ? block0 : block1;
                    } else {
                        to = _parallel_hfc.cs.flow_algo.isSource(u) ? block0 : block1;
                    }

                    if (from != to) {
                        sequence.moves.push_back(Move{ from, to, hn, kInvalidGain });
                    }
                }
            }
        }
    }

    void print(FlowHypergraphBuilder a) {
        std::cout << a._finalized << std::endl;
        std::cout << a._numPinsAtHyperedgeStart << std::endl;
        for (auto i : a._inc_he_pos) {
            std::cout << "(" << i << "),";
        }
        std::cout << std::endl;
        for (auto i : a.nodes) {
            std::cout << "(" << i.first_out << "," << i.weight << "),";
        }
        std::cout << std::endl;
        for (auto i : a.hyperedges) {
            std::cout << "(" << i.first_out << "," << i.capacity << "),";
        }
        std::cout << std::endl;

        for (auto i : a.pins) {
            std::cout << "(" << i.pin << "," << i.he_inc_iter << "),";
        }
        std::cout << std::endl;

        for (auto i : a.incident_hyperedges) {
            std::cout << "(" << i.e << "," << i.pin_iter << "),";
        }
        std::cout << std::endl;
        std::cout << V(a.total_node_weight) << std::endl;
        std::cout << V(a.maxHyperedgeCapacity) << std::endl;
    }
    bool compareSubHypergraphs(const Subhypergraph& a, const Subhypergraph& b) {
        if (a.block_0 != b.block_0) {
            return false;
        }
        if (a.block_1 != b.block_1) {
            return false;
        }
        if (a.nodes_of_block_0 != b.nodes_of_block_0) {
            return false;
        }
        if (a.nodes_of_block_1 != b.nodes_of_block_1) {
            return false;
        }
        if (a.weight_of_block_0 != b.weight_of_block_0) {
            return false;
        }
        if (a.weight_of_block_1 != b.weight_of_block_1) {
            return false;
        }
        if (a.hes != b.hes) {
            return false;
        }
        if (a.num_pins != b.num_pins) {
            return false;
        }
        return true;
    }

    bool compareFlowProblems(const FlowProblem& a, const FlowProblem& b) {
        if (a.source != b.source) {
            return false;
        }
        if (a.sink != b.sink) {
            return false;
        }
        if (a.total_cut != b.total_cut) {
            return false;
        }
        if (a.non_removable_cut != b.non_removable_cut) {
            return false;
        }
        if (a.weight_of_block_0 != b.weight_of_block_0) {
            return false;
        }
        if (a.weight_of_block_1 != b.weight_of_block_1) {
            return false;
        }
        return true;
    }

    bool compareFlowHypergraphs(const FlowHypergraphBuilder& a, const FlowHypergraphBuilder& b) {
        if (a._finalized != b._finalized) {
            return false;
        }
        if (a._numPinsAtHyperedgeStart != b._numPinsAtHyperedgeStart) {
            return false;
        }
        if (a._inc_he_pos != b._inc_he_pos) {
            return false;
        }
        if (a._tmp_csr_buckets.size() == b._tmp_csr_buckets.size()) {
            for (size_t i = 0; i < a._tmp_csr_buckets.size(); ++i) {
                if (!comapreCSRBucket(a._tmp_csr_buckets[i], b._tmp_csr_buckets[i])) {
                    return false;
                }
            }
        } else {
            return false;
        }
        if (a.nodes.size() != b.nodes.size()) {
            return false;
        }
        for (size_t i = 0; i < a.nodes.size(); ++i) {
            if (a.nodes[i].first_out != b.nodes[i].first_out || a.nodes[i].weight != b.nodes[i].weight) {
                return false;
            }
        }
        if (a.hyperedges.size() != b.hyperedges.size()) {
            return false;
        }
        for (size_t i = 0; i < a.hyperedges.size(); ++i) {
            if (a.hyperedges[i].first_out != b.hyperedges[i].first_out || a.hyperedges[i].capacity != b.hyperedges[i].capacity) {
                return false;
            }
        }
        if (a.pins != b.pins) {
            return false;
        }
        if (a.incident_hyperedges != b.incident_hyperedges) {
            return false;
        }
        if (a.total_node_weight != b.total_node_weight) {
            return false;
        }

        if (a.maxHyperedgeCapacity != b.maxHyperedgeCapacity) {
            return false;
        }
        return true;
    }

    bool comapreCSRBucket(const FlowHypergraphBuilder::TmpCSRBucket& a, const FlowHypergraphBuilder::TmpCSRBucket& b) {
        for (size_t i = 0; i < a._hes.size(); ++i) {
            if (a._hes[i].capacity != b._hes[i].capacity || a._hes[i].first_out != b._hes[i].first_out) {
                return false;
            }
        }
        if (a._pins != b._pins) {
            return false;
        }
        if (a._num_hes != b._num_hes) {
            return false;
        }
        if (a._global_start_he != b._global_start_he) {
            return false;
        }
        if (a._num_pins != b._num_pins) {
            return false;
        }
        if (a._global_start_pin_idx != b._global_start_pin_idx) {
            return false;
        }
        return true;
    }

    void buildFromFlowHypergraph(const whfc::FlowHypergraph& fhg) {
        _flow_hg.clear();
        _flow_hg.nodes.resize(fhg.nodes.size());
        for (size_t i = 0; i < fhg.nodes.size(); ++i) {
            _flow_hg.nodes[i] = fhg.nodes[i];
        }
        _flow_hg.hyperedges.resize(fhg.hyperedges.size());
        for (size_t i = 0; i < fhg.hyperedges.size(); ++i) {
            _flow_hg.hyperedges[i] = fhg.hyperedges[i];
        }
        _flow_hg.pins.resize(fhg.pins.size());
        for (size_t i = 0; i < fhg.pins.size(); ++i) {
            _flow_hg.pins[i] = fhg.pins[i];
        }
        _flow_hg.incident_hyperedges.resize(fhg.incident_hyperedges.size());
        for (size_t i = 0; i < fhg.incident_hyperedges.size(); ++i) {
            _flow_hg.incident_hyperedges[i] = fhg.incident_hyperedges[i];
        }
        _flow_hg.total_node_weight = fhg.total_node_weight;
    }

    FlowHypergraphBuilder getFlowHg() {
        return _flow_hg;
    }

    void setFlowHg(FlowHypergraphBuilder fhg) {
        _flow_hg = fhg;
    }

    FlowProblem getFlowProblem() {
        return _fp;
    }

    void setFlowProblem(FlowProblem fp) {
        _fp = fp;
    }


    const Context& _context;
    FlowHypergraphBuilder _flow_hg; // reset

    whfc::HyperFlowCutter<whfc::SequentialPushRelabel> _sequential_hfc;
    whfc::HyperFlowCutter<whfc::ParallelPushRelabel> _parallel_hfc;

    SequentialConstruction<GraphAndGainTypes> _sequential_construction; // reset
    ParallelConstruction<GraphAndGainTypes> _parallel_construction;

    DeterministicProblemConstruction<TypeTraits> _problem_construction; // reset

    vec<HypernodeID> _whfc_to_node; // reset
    FlowProblem _fp;
};
}  // namespace mt_kahypar
