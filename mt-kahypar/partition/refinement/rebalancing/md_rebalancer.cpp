/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
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

#include "mt-kahypar/partition/refinement/rebalancing/md_rebalancer.h"

#include <stdlib.h>

#include <chrono>

#include <tbb/parallel_sort.h>
#include <ranges>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <list>

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/cast.h"

#include "mt-kahypar/parallel/parallel_prefix_sum.h"

#include <queue>

namespace mt_kahypar{




  template <typename PriorityType> struct NodePQ{
    virtual bool isEmpty() = 0;
    
    virtual void changeKey(HypernodeID hn, PriorityType prio) = 0;

    virtual void move(PartitionID from, PartitionID to) = 0;

    virtual std::pair<HypernodeID,PriorityType> getMax() = 0;

    virtual std::pair<HypernodeID,PriorityType> deleteMax() = 0;

    virtual std::vector<HypernodeID> updateRequired() = 0;

    virtual void update(HypernodeID hn, PriorityType value) = 0;

    virtual void invalidate(HypernodeID hn) = 0;

    virtual PriorityType get_entry(HypernodeID hn) = 0;

    virtual bool isValid(HypernodeID hn) = 0;

    virtual bool firstInPQ(HypernodeID hn) = 0;

    virtual uint64_t num_swaps() = 0;
  };


  template <typename GraphAndGainTypes> struct lazyPQComputer{
    using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
    using GainCache = typename GraphAndGainTypes::GainCache;
    PartitionedHypergraph* phg;
    GainCache* _gain_cache;
    const Context* _context;
    const vec<bool> *L;
    std::vector<std::vector<AddressablePQ<HypernodeID,double>>> queues;
    std::vector<std::vector<bool>> is_initialized;
    std::vector<HypernodeID> id_to_index;
    std::vector<std::vector<HypernodeID>> index_to_id;
    std::vector<std::pair<HypernodeID,HypernodeID>> id_ed;
    std::vector<bool> is_extracted;

    lazyPQComputer(PartitionedHypergraph *hg, GainCache *gc, const Context *ct, const vec<bool> *l){
      phg = hg;
      _gain_cache = gc;
      _context = ct;
      L = l;
      queues.resize(phg->k());
      is_extracted.resize(phg->initialNumNodes(), false);
      is_initialized.resize(phg->k());
      for(PartitionID p = 0; p < phg->k(); p++){
        queues[p].resize(dimension);
        is_initialized[p].resize(dimension, false);
      }
      id_ed.resize(phg->initialNumNodes());
      for(HypernodeID hn : phg->nodes()){
        id_ed[hn].first = _gain_cache->penaltyTerm(hn, phg->partID(hn));
        id_ed[hn].second = 0;
        for(PartitionID p = 0; p < phg->k(); p++){
          if(p == phg->partID(hn)) continue;
          id_ed[hn].second += _gain_cache->benefitTerm(hn, p);
        }
      }
      id_to_index.resize(phg->initialNumNodes());
      index_to_id.resize(dimension * phg->k());
      for(HypernodeID hn : phg->nodes()){
        if((*L)[hn]) continue;
        if(totalweight(phg->nodeWeight(hn)) > _context->partition.fallback_large_node_threshold) continue;
        id_to_index[hn] = index_to_id[phg->partID(hn)].size();
        index_to_id[phg->partID(hn)].push_back(hn);
      }
    }

    double totalweight(HypernodeWeight hn){
      double res = 0.0;
      for(int d = 0; d < dimension; d++){
        res += hn.weights[d] * _context->partition.max_part_weights_inv[0][d];
      }
      return res;
    }

    AddressablePQ<HypernodeID,double> *get_pq(int k, int d){
      if(!is_initialized[k][d]){
        queues[k][d].setSize(index_to_id[k].size());
        for(HypernodeID h = 0; h < index_to_id[k].size(); h++){
          HypernodeID hn = index_to_id[k][h];
          
          queues[k][d].changeKey({h, get_prio(hn, k, d)});
        }
        is_initialized[k][d] = true;
      }      
      return &queues[k][d];
    }

    double get_prio(HypernodeID hn, int k, int d){
      double gain = id_ed[hn].first - (id_ed[hn].second) / (phg->k() - 1);
      ASSERT(gain >= 0);
      double weight = phg->nodeWeight(hn).weights[d] * _context->partition.max_part_weights_inv[k][d];
      double other_weight = 0.0;
      for(int d1 = 0; d1 < dimension; d1++){
        if(d1 == d) continue;
        other_weight += phg->nodeWeight(hn).weights[d1] * _context->partition.max_part_weights_inv[k][d1];
      }
      other_weight /= phg->k() - 1;
      if(weight <= 0.0){
        return std::numeric_limits<double>::max();
      }
      weight = weight - other_weight;
      return weight >= 0 ? -weight / gain : -weight * gain;
      //return gain >= 0.0 ? gain / weight : gain * weight;
    }

    void extract(HypernodeID hn){
      is_extracted[hn] = true;
      for(HyperedgeID he : phg->incidentEdges(hn)){
        HypernodeID et = phg->edgeTarget(he);
        if(phg->partID(et) == phg->partID(hn)){
          id_ed[et].first -= phg->edgeWeight(he);
        }
        else{
          id_ed[et].second -= phg->edgeWeight(he);
        }
        for(int d = 0; d < dimension; d++){
          if(is_initialized[phg->partID(et)][d]){
            std::pair<HypernodeID,double> val;
            val.first = id_to_index[et];
            val.second = get_prio(et, phg->partID(et), d);
            queues[phg->partID(et)][d].changeKey(val);
          }
        }
      }
    }

    HypernodeID deleteMax(PartitionID p, int dim){
      get_pq(p, dim);
      std::pair<HypernodeID, double> max_pair = queues[p][dim].deleteMax();
      std::cout << "new gain: " << max_pair.second << "\n";
      HypernodeID max = index_to_id[p][max_pair.first];
      while(is_extracted[max]){
        max = index_to_id[p][queues[p][dim].deleteMax().first];
      }
      extract(max);
      return max;
    }

  };


  template <typename GraphAndGainTypes> struct Binpacker{
    using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;

    PartitionedHypergraph *phg;
    std::vector<HypernodeID> *nodes;
    const Context *_context;
    HypernodeID current_idx = 0;
    HypernodeID last_success = 0;
    std::vector<bool> used;
    std::vector<bool> available;
    std::vector<std::vector<std::pair<HypernodeID, int32_t>>> ns_1d;
    std::vector<std::pair<HypernodeID,double>> ns_total_weight;
    std::vector<std::pair<HypernodeID,double>> high_low_diff;

    std::vector<PartitionID> partitioning;

    Binpacker(std::vector<HypernodeID> *n, PartitionedHypergraph *hg, const Context *c){
      phg = hg;
      nodes = n;
      available.resize(phg->initialNumNodes(), false);
      used.resize(phg->initialNumNodes(), false);
      partitioning.resize(phg->initialNumNodes(), -1);
      ns_1d.resize(dimension);
      _context = c;
    }

    void update_sorted_lists(){      
      for(int d = 0; d < dimension; d++){
        auto ns_1d_prio = [this, d](HypernodeID hn){
          return phg->nodeWeight(hn).weights[d] * 
            _context->partition.max_part_weights_inv[0][d];
        };
        update_sorted_list(&ns_1d[d], ns_1d_prio);
      }
      auto hld_prio = [&](HypernodeID hn){
        double high = 0.0;
        double low = std::numeric_limits<double>::max();
        for(int d = 0; d < dimension; d++){
          double w = phg->nodeWeight(hn).weights[d] * 
            _context->partition.max_part_weights_inv[0][d];
          high = std::max(high, w); 
          low = std::min(low, w);
        }
        return high - low;
      };
      update_sorted_list(&high_low_diff, hld_prio);
      auto get_normalized_weight = [&](HypernodeID hn){
        double res = 0.0;
        for(int d = 0; d < dimension; d++){
          res += phg->nodeWeight(hn).weights[d] * _context->partition.max_part_weights_inv[0][d];
        }
        return res;
      };
      update_sorted_list(&ns_total_weight, get_normalized_weight);
    }

    template<typename T>
    void update_sorted_list(std::vector<std::pair<HypernodeID,T>> *nodes_sorted, auto get_priority){
      HypernodeID num_nodes_before = nodes_sorted->size();
      for(HypernodeID idx = num_nodes_before; idx < nodes->size(); idx++){
          nodes_sorted->push_back({(*nodes)[idx], get_priority((*nodes)[idx])});
      }
      std::sort(nodes_sorted->begin() + num_nodes_before, nodes_sorted->end(), [](std::pair<HypernodeID,T> a,
       std::pair<HypernodeID,T> b){
        return a.second > b.second;
       });
      std::vector<std::pair<HypernodeID, T>> tmp_nodes(nodes_sorted->size());
      HypernodeID idx_1 = 0;
      HypernodeID idx_2 = num_nodes_before;
      HypernodeID counter = 0;
      while(counter < nodes_sorted->size()){
        HypernodeID next;
        if(idx_1 == num_nodes_before){
          next = idx_2;
          idx_2++;
        }
        else if(idx_2 == nodes_sorted->size()){
          next = idx_1;
          idx_1++;
        }
        else if((*nodes_sorted)[idx_2].second > (*nodes_sorted)[idx_1].second){
          next = idx_2;
          idx_2++;
        }
        else{
          next = idx_1;
          idx_1++;
        }
        tmp_nodes[counter] = (*nodes_sorted)[next];
        counter++;
      }
      for(HypernodeID hn = 0; hn < nodes->size(); hn++){
        (*nodes_sorted)[hn] = tmp_nodes[hn];
        if(hn < nodes->size() - 1){
          ASSERT(tmp_nodes[hn].second >= tmp_nodes[hn + 1].second);
        }
      }
    }

    std::vector<HypernodeID> next_nodes(std::vector<HypernodeID>& ns_1d_idx, HypernodeID& tw_idx, HypernodeID& hld_idx){
      std::vector<HypernodeID> res;
      for(int d = 0; d < ns_1d_idx.size(); d++){
        //std::cout << "size: " << ns_1d[d].size() << "\n";
        HypernodeID idx = ns_1d_idx[d];
        //std::cout << "idx:" << idx << "\n";
        //std::cout << "a " << available[ns_1d[d][idx].first] << " " << used[ns_1d[d][idx].first] << "\n";
        //std::cout << "node " << ns_1d[d][idx].first << "\n";
        while(idx < ns_1d[d].size() && (!available[ns_1d[d][idx].first] || used[ns_1d[d][idx].first])) idx++;
        ns_1d_idx[d] = idx;
        if(idx < ns_1d[d].size()) res.push_back(ns_1d[d][idx].first);
      }
      while(tw_idx < ns_total_weight.size() && (!available[ns_total_weight[tw_idx].first] || used[ns_total_weight[tw_idx].first])) tw_idx++;
      if(tw_idx < ns_total_weight.size()) res.push_back(ns_total_weight[tw_idx].first);
      while(hld_idx < high_low_diff.size() && (!available[high_low_diff[hld_idx].first] || used[high_low_diff[hld_idx].first])) hld_idx++;
      if(hld_idx < high_low_diff.size()) res.push_back(high_low_diff[hld_idx].first);
      return res;
    }

    bool binpack(HypernodeID new_num, std::vector<HypernodeWeight> virtual_weight, auto penalty){
      ASSERT(new_num > 0);
      ASSERT(nodes->size() >= new_num);
      ASSERT(nodes->size() <= phg->initialNumNodes());
      HypernodeID old_index = current_idx;
      current_idx = new_num;
      for(int p = 0; p < phg->k(); p++){
        virtual_weight[p] = _context->partition.max_part_weights[p] - virtual_weight[p];
      }
      for(HypernodeID idx = 0; idx < new_num; idx++){
        used[(*nodes)[idx]] = false;
      }
      if(new_num > old_index){
        for(HypernodeID idx = old_index; idx < new_num; idx++){
          available[(*nodes)[idx]] = true;
        }
      }
      else{
        for(HypernodeID idx = new_num; idx < old_index; idx++){
          available[(*nodes)[idx]] = false;
        }
      }
      for(int i = 0; i < new_num; i++){
        ASSERT(available[(*nodes)[i]]);
      }
      for(int i = new_num; i < nodes->size(); i++){
        ASSERT(!available[(*nodes)[i]]);
      }
      if(new_num > ns_total_weight.size()){
        //std::cout << "higher " << ns_total_weight.size() << " " << new_num << " " << nodes->size() << "\n";
        
        update_sorted_lists();
        for(int d = 0; d < dimension; d++){
          ASSERT(ns_1d[d].size() == ns_total_weight.size());
        }
        ASSERT(ns_total_weight.size() == high_low_diff.size());
        ASSERT(ns_total_weight.size() == nodes->size());
      }
      HypernodeID nodesleft = new_num;

      std::vector<HypernodeID> ns_1d_idx(dimension, 0);
      HypernodeID tw_idx = 0;
      HypernodeID hld_idx = 0;
      std::vector<std::pair<HypernodeID,PartitionID>> tmp_partitioning;
      while(nodesleft > 0){
        HypernodeID chosen_node = 0;
        PartitionID chosen_p;
        std::vector<HypernodeID> next = next_nodes(ns_1d_idx, tw_idx, hld_idx);
        double max_penn_diff = -1.0;
        ASSERT(next.size() > 0);
        for(HypernodeID hn : next){
          ASSERT(!used[hn] && available[hn]);
          double best_pen = std::numeric_limits<double>::max();
          double second_best_pen = std::numeric_limits<double>::max();
          PartitionID best_p = -1;
          for(PartitionID p = 0; p < phg->k(); p++){
            HypernodeWeight nw = virtual_weight[p] - phg->nodeWeight(hn);
            if(!(nw >= 0)) continue;
            double pen = penalty(nw) - penalty(virtual_weight[p]);
            if(pen < best_pen){
              best_p = p;
              second_best_pen = best_pen;
              best_pen = pen;
            } 
          }
          if(best_p == -1) return false;
          if(second_best_pen - best_pen > max_penn_diff){
            chosen_node = hn;
            chosen_p = best_p;
          }
        }
        virtual_weight[chosen_p] -= phg->nodeWeight(chosen_node);
        tmp_partitioning.push_back({chosen_node, chosen_p});
        used[chosen_node] = true;        
        nodesleft--;
      }
      ASSERT(tmp_partitioning.size() == new_num);
      for(auto pair : tmp_partitioning){
        partitioning[pair.first] = pair.second;
      }
      for(PartitionID p = 0; p < phg->k(); p++){
        std::cout << "weights:\n";
        for(int d = 0; d < dimension; d++){
          std::cout << virtual_weight[p].weights[d] << " ";
        }
        std::cout << "\n";
      }
      for(HypernodeID hn = 0; hn < new_num; hn++){
        ASSERT(available[(*nodes)[hn]] && used[(*nodes)[hn]]);
        ASSERT(partitioning[(*nodes)[hn]] != -1);
      }

      for(HypernodeID idx = new_num; idx < last_success; idx++){
        partitioning[(*nodes)[idx]] = -1;
      } 
      last_success = new_num;  
      for(HypernodeID hn = new_num; hn < nodes->size(); hn++){
        ASSERT(partitioning[(*nodes)[hn]] == -1);
      }   
      return true;
    }
  };





  template <typename PriorityType> struct SimplePQ : NodePQ<PriorityType>{
    AddressablePQ<HypernodeID,PriorityType> queue;
    SimplePQ(size_t size){
      queue.setSize(size);
    }

    void changeKey(HypernodeID hn, PriorityType prio){
      queue.changeKey({hn, prio});
    }

    
    std::pair<HypernodeID,PriorityType> getMax(){
      return queue.getMax();
    }

    std::pair<HypernodeID,PriorityType> deleteMax(){
      return queue.deleteMax();
    }

    bool isEmpty(){
      return queue.isEmpty();
    }

    void move(PartitionID from, PartitionID to){}

    void invalidate(HypernodeID hn){
      queue.invalidate(hn);
    }

    std::vector<HypernodeID> updateRequired(){
      std::vector<HypernodeID> res;
      return res;
    }

    void update(HypernodeID hn, PriorityType value){}

    PriorityType get_entry(HypernodeID hn){
      return queue.items[hn];
    }

    bool isValid(HypernodeID hn){
      return queue.in_use[hn];
    }

    bool firstInPQ(HypernodeID hn){
      return queue.references[hn] == 0 && queue.in_use[hn] == true;
    }

    uint64_t num_swaps(){
      return queue.num_swaps;
    } 
  };

  template <typename PartitionedHypergraph, typename PriorityType> struct pd_PQ : NodePQ<PriorityType>{
    private:
      using PQID=uint16_t;
    /*PartitionedHypergraph& phg;*/
    /*mt_kahypar_partitioned_hypergraph_t& hg;*/
    std::vector<AddressablePQ<HypernodeID,PriorityType>> queue;
    AddressablePQ<PQID, PriorityType> top_queues;
    std::vector<HypernodeID> id_to_index;
    std::vector<std::pair<PartitionID,int>> part_max_dims;
    std::vector<std::vector<HypernodeID>> index_to_id;
    bool has_moved;
    PQID last_extracted = 0;

    
    public:
    bool isEmpty(){
      return top_queues.isEmpty();
    }

    pd_PQ(mt_kahypar_partitioned_hypergraph_t& hypergraph, const Context& _context){
      PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
      queue.resize(dimension * phg.k());
      id_to_index.resize(phg.initialNumNodes());
      index_to_id.resize(dimension * phg.k());
      part_max_dims.resize(phg.initialNumNodes());
      top_queues.setSize(dimension * phg.k());
      has_moved = false;
      for(HypernodeID hn : phg.nodes()){
        part_max_dims[hn].first = phg.partID(hn);
        int dim = get_max_dim(hn, phg, _context);
        part_max_dims[hn].second = dim;
        id_to_index[hn] = index_to_id[get_pq_id(phg.partID(hn),dim)].size();
        index_to_id[get_pq_id(phg.partID(hn),dim)].push_back(hn);
      }
      for(PartitionID p = 0; p < phg.k(); p++){
        for(int d = 0; d < dimension; d++){
          queue[get_pq_id(p, d)].setSize(index_to_id[get_pq_id(p, d)].size());
        }
      }
    }

    PQID get_pq_id(PartitionID part, int max_dim){
      return part * dimension + max_dim;
    }
    PQID get_pq_id(HypernodeID hn){
      return part_max_dims[hn].first * dimension + part_max_dims[hn].second;
    }

    AddressablePQ<HypernodeID,PriorityType>& get_queue(HypernodeID hn){
      return queue[get_pq_id(part_max_dims[hn].first, part_max_dims[hn].second)];
    }

    void invalidate(HypernodeID hn){
      PQID idx = get_pq_id(hn);
      queue[idx].invalidate(id_to_index[hn]);
      if(queue[idx].isEmpty()){
        top_queues.invalidate(idx);
      }
      else{
        top_queues.changeKey({idx, queue[idx].getMax().second});
      }
    }

    int get_max_dim(HypernodeID hn, PartitionedHypergraph& phg, const Context& context){
      int dim = 0;
      double mw = 0.0;
      for(int d = 0; d < dimension; d++){
        if(context.partition.max_part_weights_inv[phg.partID(hn)][d] * phg.nodeWeight(hn).weights[d] > mw){
          mw = context.partition.max_part_weights_inv[phg.partID(hn)][d] * phg.nodeWeight(hn).weights[d];
          dim = d;
        }
      }
      return dim;
    }


    void changeKey(HypernodeID hn, PriorityType prio){
      PQID idx = get_pq_id(hn);
      double queue_max = queue[idx].isEmpty() ? std::numeric_limits<double>::max() : queue[idx].getMax().second;
      queue[idx].changeKey({id_to_index[hn], prio});
      if(queue[idx].getMax().second != queue_max){
        top_queues.changeKey({idx, queue[idx].getMax().second});
      }
    }

    void move(PartitionID from, PartitionID to){
      has_moved = true;
    }

    std::pair<HypernodeID,PriorityType> getMax(){
      std::pair<HypernodeID,PriorityType> maximum = queue[top_queues.getMax().first].getMax();
      return {index_to_id[top_queues.getMax().first][maximum.first], maximum.second};
    }

    std::pair<HypernodeID,PriorityType> deleteMax(){
      last_extracted = top_queues.getMax().first;
      std::pair<HypernodeID, PriorityType> tmp = getMax();
      queue[top_queues.getMax().first].deleteMax();
      if(queue[top_queues.getMax().first].isEmpty()){
        top_queues.invalidate(top_queues.getMax().first);
      }
      else{
        top_queues.changeKey({top_queues.getMax().first, queue[top_queues.getMax().first].getMax().second});
      }
      return tmp;
    }

    std::vector<HypernodeID> updateRequired(){
      std::vector<HypernodeID> res;
      if(has_moved){
        for(PQID id = 0; id < queue.size(); id++){
          if(!queue[id].isEmpty()){
            res.push_back(index_to_id[id][queue[id].getMax().first]);
          }
          else{
            top_queues.invalidate(id);
          }          
        }
      }
      else{
        if(!queue[last_extracted].isEmpty()){
          res.push_back(index_to_id[last_extracted][queue[last_extracted].getMax().first]);
        }
        else{
          top_queues.invalidate(last_extracted);
        } 
      }
      has_moved = false;
      return res;
    }

    void update(HypernodeID hn, PriorityType value){
      PQID idx = get_pq_id(hn);
      if(!queue[idx].in_use[id_to_index[hn]] || queue[idx].items[id_to_index[hn]] > value){
        queue[idx].changeKey({id_to_index[hn], value});
        top_queues.changeKey({get_pq_id(hn), value});
      }
    }

    PriorityType get_entry(HypernodeID hn){
      PQID idx = get_pq_id(hn);
      return queue[idx].items[id_to_index[hn]];
    }

    bool isValid(HypernodeID hn){
      return queue[get_pq_id(hn)].in_use[id_to_index[hn]];
    }

    bool firstInPQ(HypernodeID hn){
      PQID idx = get_pq_id(hn);
      HypernodeID h = id_to_index[hn];
      return queue[idx].references[h] == 0;
    }

    uint64_t num_swaps(){
      uint64_t res = 0;
      for(PQID idx = 0; idx < queue.size(); idx++){
        res += queue[idx].num_swaps;
      }
      return res;
    }
 

  };

  template<typename GraphAndGainTypes, typename PriorityType> struct PriorityComputer{
    using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
    using GainCache = typename GraphAndGainTypes::GainCache;
    public:
      PartitionedHypergraph* phg;
      GainCache* gain_cache;
      NodePQ<PriorityType>* pq;
      int round = 1;
      std::vector<int> moved;
      std::vector<bool> is_in_boundary;
      std::vector<HypernodeID> boundary;
      std::vector<std::pair<HypernodeID,HypernodeID>> id_ed;

    NodePQ<PriorityType>* get_pq(){
      return pq;
    }

    PartitionedHypergraph* get_phg(){
      return phg;
    }

    void insert_into_boundary(HypernodeID hn){
      if(!is_in_boundary[hn]){
        is_in_boundary[hn] = true;
        boundary.push_back(hn);
      }
    }

    void initialize(){
      moved.resize(phg->initialNumNodes(), 0);
      is_in_boundary.resize(phg->initialNumNodes(), false);
      initializationImpl();
      for(HypernodeID hn : phg->nodes()){
        adjustNode(hn);
      }
    }

    void reinitialize(){
      HypernodeID virtual_boundary_size = 0;
      utils::Randomize::instance().shuffleVector(boundary);
      for(HypernodeID& hn : boundary){
        std::pair<bool,PriorityType> prio = computePriority(hn);
        if(prio.first){
          pq->changeKey(hn, -prio.second);
          boundary[virtual_boundary_size] = hn;
          virtual_boundary_size++;
        }
        else{
          is_in_boundary[hn] = false;
        }
      }
      boundary.resize(virtual_boundary_size);
      round++;
    }


    void performMove(Move move){
      moved[move.node] = round;
      registerMove(move);
    }
    void registerMove(Move move){
      if(updateNodeAfterOwnMove(move)){
        insert_into_boundary(move.node);
      }
      for(HyperedgeID he : phg->incidentEdges(move.node)){
        HypernodeID h = phg->edgeTarget(he);
        updateNodeAfterOtherMove(h, he, move);
        if(moved[h] != round){
          adjustNode(h);
        }        
      }
    }

    void initialize_id_ed(){
      if(id_ed.size() == 0){
        id_ed.resize(phg->initialNumNodes(), {0,0});
        for(HypernodeID hn : phg->nodes()){
          for(HyperedgeID he : phg->incidentEdges(hn)){
            for(HypernodeID h : phg->pins(he)){
              if(h == hn) continue;
              if(this->phg->partID(h) == this->phg->partID(hn)){
                id_ed[hn].first += this->phg->edgeWeight(he);
              }
              else{
                id_ed[hn].second += this->phg->edgeWeight(he);
              }
            }
          }
        }
      }
    }

    private: 
      void adjustNode(HypernodeID hn){
        std::pair<bool,PriorityType> prio = computePriority(hn);
        if(moved[hn] < round){
          if(prio.first){
            pq->changeKey(hn, -prio.second);
            insert_into_boundary(hn);
          }
          else{
            pq->invalidate(hn);
          }
        }          
      }
      virtual void initializationImpl() = 0;
      virtual std::pair<bool,PriorityType> computePriority(HypernodeID hn) = 0;
      virtual bool updateNodeAfterOwnMove(Move move) = 0;
      virtual void updateNodeAfterOtherMove(HypernodeID hn, HyperedgeID he, Move move) = 0;
  };

  template<typename GraphAndGainTypes> struct GainPriorityComputer : PriorityComputer<GraphAndGainTypes,Gain>{
    using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
    using GainCache = typename GraphAndGainTypes::GainCache;
    GainPriorityComputer(PartitionedHypergraph* p, GainCache* gc, NodePQ<Gain>* _pq){
      this->phg = p;
      this->gain_cache = gc;
      this->pq = _pq;      
    }

    std::pair<bool,Gain> computePriority(HypernodeID hn){
      Gain gain = -1;
      for(PartitionID p = 0; p < this->phg->k(); p++){
        if(p != this->phg->partID(hn) && this->gain_cache->gain(hn, this->phg->partID(hn), p) > gain){
          gain = this->gain_cache->gain(hn, this->phg->partID(hn), p);
        }
      }
      return {gain >= 0, gain};
    }

    void initializationImpl(){}
    bool updateNodeAfterOwnMove(Move move){
      return computePriority(move.node).first;
    }
    void updateNodeAfterOtherMove(HypernodeID hn, HyperedgeID he, Move move){}

  };

  template<typename GraphAndGainTypes> struct MetisMetricPriorityComputer : PriorityComputer<GraphAndGainTypes,double>{
    using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
    using GainCache = typename GraphAndGainTypes::GainCache;
    std::vector<std::vector<HypernodeID>> num_neighbors;
    std::vector<PartitionID> nnp;

    MetisMetricPriorityComputer(PartitionedHypergraph* p, GainCache* gc, NodePQ<double>* _pq){
      this->phg = p;
      this->gain_cache = gc;
      this->pq = _pq;      
    }

    void initializationImpl(){
      this->id_ed.resize(this->phg->initialNumNodes(), {0,0});
      num_neighbors.resize(this->phg->initialNumNodes());
      nnp.resize(this->phg->initialNumNodes(), 0);
      for(HypernodeID hn : this->phg->nodes()){
        num_neighbors[hn].resize(this->phg->k(), 0);
        for(HyperedgeID he : this->phg->incidentEdges(hn)){
          for(HypernodeID h : this->phg->pins(he)){
            if(h == hn) continue;
            if(this->phg->partID(h) == this->phg->partID(hn)){
              this->id_ed[hn].first += this->phg->edgeWeight(he);
            }
            else{
              this->id_ed[hn].second += this->phg->edgeWeight(he);
              if(num_neighbors[hn][this->phg->partID(h)]== 0){
                nnp[hn]++;
              }
            }
            num_neighbors[hn][this->phg->partID(h)] += this->phg->edgeWeight(he);
          }
        }
      }
    }

    std::pair<bool,double> computePriority(HypernodeID hn){
      if(this->id_ed[hn].second >= this->id_ed[hn].first){
        return {true, (nnp[hn] > 0 ? (this->id_ed[hn].second / sqrt(nnp[hn])) : 0.0) - this->id_ed[hn].first};
      }
      else{
        return {false, 0.0};
      }
    }

    bool updateNodeAfterOwnMove(Move move){
      if(num_neighbors[move.node][move.from] > 0) nnp[move.node]++;
      if(num_neighbors[move.node][move.to] > 0) nnp[move.node]--;
      this->id_ed[move.node].first = num_neighbors[move.node][move.to];
      this->id_ed[move.node].second = this->id_ed[move.node].second + num_neighbors[move.node][move.from] - num_neighbors[move.node][move.to];
      return put_into_pq(move.node);
    }

    void updateNodeAfterOtherMove(HypernodeID h, HyperedgeID he, Move move){
      if(num_neighbors[h][move.to] == 0 && this->phg->partID(h) != move.to){
        nnp[h]++;
      }
      num_neighbors[h][move.from] -= this->phg->edgeWeight(he);
      num_neighbors[h][move.to] += this->phg->edgeWeight(he);
      if(num_neighbors[h][move.from] == 0 && this->phg->partID(h) != move.from){
        nnp[h]--;
      }
      if(this->phg->partID(h) == move.from){
        this->id_ed[h].first -= this->phg->edgeWeight(he);
        this->id_ed[h].second += this->phg->edgeWeight(he);
      }
      if(this->phg->partID(h) == move.to){
        this->id_ed[h].second -= this->phg->edgeWeight(he);
        this->id_ed[h].first += this->phg->edgeWeight(he);
      }
    }

    bool put_into_pq(HypernodeID hn){
      return this->id_ed[hn].second >= this->id_ed[hn].first;
    }

  };


  template<typename PriorityType> struct MoveCache{
    virtual void insert(HypernodeID hn, PartitionID to, PriorityType value) = 0;
    virtual void movePerformed(PartitionID from, PartitionID to) = 0;
    virtual PriorityType getEntry(HypernodeID hn, PartitionID p) = 0;
    virtual bool could_moveout_have_improved(HypernodeID hn) = 0;
    virtual bool could_moveout_have_changed(HypernodeID hn) = 0;
    virtual bool could_movein_have_improved(HypernodeID hn, PartitionID p) = 0;
    virtual bool could_movein_have_changed(HypernodeID hn, PartitionID p) = 0;
    virtual PartitionID get_part(HypernodeID hn) = 0;
  };

  template<typename PriorityType> struct doNothingCache: MoveCache<PriorityType>{
    void insert(HypernodeID hn, PartitionID to, PriorityType value){}
    void movePerformed(PartitionID from, PartitionID to){}
    PriorityType getEntry(HypernodeID hn, PartitionID p){return 0;}
    virtual bool could_moveout_have_improved(HypernodeID hn){return true;}
    virtual bool could_moveout_have_changed(HypernodeID hn){return true;}
    virtual bool could_movein_have_improved(HypernodeID hn, PartitionID p){return true;}
    virtual bool could_movein_have_changed(HypernodeID hn, PartitionID p){return true;}
    PartitionID get_part(HypernodeID hn){return -1;}
  };

  template<typename PartitionedHypergraph, typename PriorityType> struct trackMovesCache : MoveCache<PriorityType>{
    std::vector<std::pair<MoveID,std::vector<PriorityType>>> computed_values;
    MoveID current_move = 0;
    std::vector<MoveID> last_increased;
    std::vector<MoveID> last_decreased;
    trackMovesCache(mt_kahypar_partitioned_hypergraph_t& hypergraph){
      PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
      computed_values.resize(phg.initialNumNodes());
      last_increased.resize(phg.k());
      last_decreased.resize(phg.k());
      for(HypernodeID idx = 0; idx < phg.initialNumNodes(); idx++){
        computed_values[idx].second.resize(phg.k() + 1);
        computed_values[idx].second[phg.k()] = phg.partID(idx);
      }
    }

    void insert(HypernodeID hn, PartitionID to, PriorityType value){
      computed_values[hn].first = current_move;
      computed_values[hn].second[to] = value;
    }

    void movePerformed(PartitionID from, PartitionID to){
      current_move++;
      last_decreased[from] = current_move;
      last_increased[to] = current_move;
    }

    bool could_moveout_have_improved(HypernodeID hn){
      PartitionID from = computed_values[hn].second[computed_values[hn].second.size() - 1];
      return last_increased[from] > computed_values[hn].first;
    }
    bool could_moveout_have_changed(HypernodeID hn){
      PartitionID from = computed_values[hn].second[computed_values[hn].second.size() - 1];
      return std::max(last_increased[from],last_decreased[from]) > computed_values[hn].first;
    }
    bool could_movein_have_improved(HypernodeID hn, PartitionID p){
      return last_decreased[p] > computed_values[hn].first;
    }
    bool could_movein_have_changed(HypernodeID hn, PartitionID p){
      return std::max(last_increased[p], last_decreased[p]) > computed_values[hn].first;
    }
    PartitionID get_part(HypernodeID hn){
      return computed_values[hn].second[computed_values[hn].second.size() - 1];
    }
    PriorityType getEntry(HypernodeID hn, PartitionID p){
      return computed_values[hn].second[p];
      }
  };



  struct interval{
    HypernodeID lower_index;
    HypernodeID upper_index;
    double lower_search_value;
    double upper_search_value;
  };

  struct pointer_list{
    std::vector<std::pair<int,HypernodeID>> next_per_level;
  };


  struct hypernodes_ordered{
    std::vector<std::vector<double>> nodes_normalized;
    std::vector<std::pair<double,HypernodeID>> nodes_normalized_total_weight;
    std::vector<int> imbalanced_dimensions;
    std::vector<std::vector<std::pair<HypernodeID, double>>> nodes_prio;
    std::vector<std::vector<int>> levels;
    /*std::vector<std::vector<pointer_list>> pointerList;*/
    std::vector<HypernodeID> current_indices;


    //dimensions has to be sorted
    void initialize(std::vector<std::vector<double>> nn, std::vector<std::pair<double,HypernodeID>> nntw, std::vector<bool> could_be_imbalanced){
      nodes_normalized = nn;
      nodes_normalized_total_weight = nntw;
      for(int i = 0; i < could_be_imbalanced.size(); i++){
        if(could_be_imbalanced[i]){
          imbalanced_dimensions.push_back(i);
        }
      }
      nodes_prio.resize(std::pow(2, imbalanced_dimensions.size()));
      current_indices.resize(std::pow(2, imbalanced_dimensions.size()), 0);
      /*pointerList.resize(std::pow(2, imbalanced_dimensions.size()));*/
      levels.resize(std::pow(2, imbalanced_dimensions.size()));
    }

    void initialize(std::vector<std::vector<double>> nn, std::vector<std::pair<double,HypernodeID>> nntw){
      nodes_normalized = nn;
      nodes_normalized_total_weight = nntw;
      for(int i = 0; i < dimension; i++){
        imbalanced_dimensions.push_back(i);
      }
      nodes_prio.resize(std::pow(2, imbalanced_dimensions.size()));
      /*pointerList.resize(std::pow(2, imbalanced_dimensions.size()));*/
      current_indices.resize(std::pow(2, imbalanced_dimensions.size()), 0);
      levels.resize(std::pow(2, imbalanced_dimensions.size()));
    }

    size_t get_index(std::vector<bool> imbalanced){
      size_t index = 0;
      for(int i = 0; i < imbalanced_dimensions.size(); i++){
        if(imbalanced[imbalanced_dimensions[i]]){
          index += std::pow(2,i);
        }
      }
      return index;
    }

    //imbalanced has to be sorted
    std::vector<std::pair<HypernodeID, double>> nodes_priorised(std::vector<bool> imbalanced){
      size_t index = get_index(imbalanced);
      /*pointerList[index].resize(nodes_normalized.size() + 1);*/
      std::vector<size_t> imbalanced_list;
      for(int i = 0; i < imbalanced_dimensions.size(); i++){
        if(imbalanced[imbalanced_dimensions[i]]){
          imbalanced_list.push_back(imbalanced_dimensions[i]);
        }
      }
      if(nodes_prio[index].size() != 0){
        return nodes_prio[index];
      }
      nodes_prio[index].resize(nodes_normalized.size());
      if(nodes_prio[nodes_prio.size() - 1 - index].size() != 0){
        for(HypernodeID hn = 0; hn < nodes_normalized.size(); hn++){
          nodes_prio[index][hn] = nodes_prio[nodes_prio.size() - 1 - index][nodes_normalized.size() - 1 - hn];
        }
      }
      else{
        for(HypernodeID hn = 0; hn < nodes_normalized.size(); hn++){
          nodes_prio[index][hn] = {hn, 0.0};
          for(size_t i = 0; i < imbalanced_list.size(); i++){
            nodes_prio[index][hn].second += nodes_normalized[hn][imbalanced_list[i]];
          }
          nodes_prio[index][hn].second /= nodes_normalized_total_weight[hn].first;
        }
        std::sort(nodes_prio[index].begin(), nodes_prio[index].end(), [&](std::pair<HypernodeID, double> a, std::pair<HypernodeID, double> b){
          return a.second > b.second;
        });
      }

      levels[index].resize(nodes_prio[index].size());
      for(HypernodeID hn = nodes_prio[index].size() - 1; hn-- > 0;){
        HypernodeID node = nodes_prio[index][hn].first;
        int level_number = 0;
        double ignored_percentage = 0.5;
        while(std::ceil(ignored_percentage * nodes_normalized.size()) < nodes_normalized_total_weight[node].second + 1){
          level_number++;
          ignored_percentage = (ignored_percentage + 1.0) / 2.0;
        }
        levels[index][hn] = level_number;
      }



      /*
      std::list<std::pair<int,HypernodeID>> current_last;
      levels[index].resize(nodes_prio[index].size());
      for(HypernodeID hn = nodes_prio[index].size() - 1; hn-- > 0;){
        HypernodeID node = nodes_prio[index][hn].first;
        int level_number = 0;
        double ignored_percentage = 0.5;
        while(std::ceil(ignored_percentage * nodes_normalized.size()) < nodes_normalized_total_weight[node].second + 1){
          level_number++;
          ignored_percentage = (ignored_percentage + 1.0) / 2.0;
        }
        levels[index][hn] = level_number;
        int idx_in_current = 0;
        HypernodeID pointer;
        bool exists_smaller = false;
        for(auto x : current_last){
          if(x.first < level_number){
            exists_smaller =true;
            pointer = x.second;
            continue;
          }
          pointerList[index][hn + 1].next_per_level.push_back(x);
        }
        if(exists_smaller && (pointerList[index][hn + 1].next_per_level.size() == 0 || pointerList[index][hn + 1].next_per_level[0].first != level_number)){
          pointerList[index][hn + 1].next_per_level.changeKey(pointerList[index][hn + 1].next_per_level.begin(), {level_number, pointer});
        }       
        
        auto it = current_last.begin();
        while(it != current_last.end() && (*it).first < level_number){
          std::next(it, 1);
        }

        current_last.erase(it, current_last.end());
        
        current_last.push_back({level_number, hn + 1});

        
              
      }
      
      
      for(auto x : current_last){
        pointerList[index][0].next_per_level.push_back(x);
      }


      for(HypernodeID hn = 0; hn < nodes_prio[index].size() + 1; hn++){
        for(int i = 0; i < pointerList[index][hn].next_per_level.size(); i++){
          std::cout << pointerList[index][hn].next_per_level[i].first << " " << pointerList[index][hn].next_per_level[i].second << "\n";
        }
        std::cout << "end\n";
        
      }*/

      

      /*std::vector<pointer_list> &p_l = pointerList[index];
      double frac = 0.5;
      HypernodeID level_index;
      int level_number = 0;
      std::vector<bool> nodeInserted(nodes_prio[index].size(), false);
      do{
        level_index = std::ceil(frac * nodes_normalized.size());
        HypernodeID lastIndex = 0;
        for(HypernodeID hn = 0; hn < nodes_prio[index].size(); hn++){
          if(nodes_normalized_total_weight[nodes_prio[index][hn].first].second <= level_index){
            if(!nodeInserted[hn]){
              p_l[lastIndex].next_per_level.push_back({level_number, hn});
            }
            lastIndex = hn + 1;
            nodeInserted[hn] = true;
          }
        }
        frac = (1.0 + frac) / 2.0;
        level_number++;
      }while(level_index != nodes_normalized.size());
      p_l[p_l.size() - 1].next_per_level.push_back({0, -1});*/
      
      return nodes_prio[index];
    }

    std::pair<HypernodeID,int> getNext(std::vector<bool> imbalanced, int level){
      nodes_priorised(imbalanced);
      size_t index = get_index(imbalanced);
      int max_level = level;
      for(HypernodeID hn = current_indices[index]; hn < nodes_prio[index].size(); hn++){
        max_level = std::max(max_level, levels[index][hn]);
        if(levels[index][hn] <= level){
          current_indices[index] = hn + 1;
          return {nodes_prio[index][hn].first, max_level};
        }
      }
      current_indices[index] = 0;
      return {0, -1};

      /*pointer_list current = pointerList[index][current_indices[index]];    
      if(current.next_per_level.size() == 0){
        return {0, -1};
      }
      int start = 0;
      int end = current.next_per_level.size();
      while(end != start + 1 && current.next_per_level[start + 1].first < level){
        int middle = (start + end) / 2;
        if(current.next_per_level[middle].first <= level){
          start = middle;
        }
        else{
          end = middle;
        }
      }
      current_indices[index] = current.next_per_level[start].second;
      int lowest_above = start < current.next_per_level.size() - 1 ? current.next_per_level[start + 1].first : current.next_per_level[start].first;
      return {nodes_prio[index][current.next_per_level[start].second - 1].first, lowest_above};*/
    }

    /*struct pointer_list getentry(std::vector<bool> imbalanced, HypernodeID index){
      return pointerList[get_index(imbalanced)][index];
    }*/

    void reset(){
      for(HypernodeID hn = 0; hn < current_indices.size(); hn++){
        current_indices[hn] = 0;
      }
    }


  };

  /*std::vector<double> relative_weights(HypernodeWeight nw, HypernodeWeight max){
    std::vector<double> res;
    double sum = 0.0;
    for(int i = 0; i < dimension; i++){
      double fraction = static_cast<double>(nw.weights[i]) / static_cast<double>(max.weights[i]);
      res.push_back(fraction);
      sum += fraction;
    }
    for(int i = 0; i < dimension; i++){
      res[i] /= sum;
    }
    return res;
  }*/


  double vertical_imbalance(std::vector<double> weights){
    double total = 0.0;
    for(int i = 0; i < weights.size(); i++){
      total += weights[i];
    }
    total /= weights.size();
    double loss = 0.0;
    for(int i = 0; i < weights.size(); i++){
      loss += (weights[i] - total) * (weights[i] - total);
    }
    return loss;
  }

  std::vector<double> minus(std::vector<double> a, std::vector<double> b){
    std::vector<double> res;
    for(int i = 0; i < a.size(); i++){
      res.push_back(a[i] - b[i]);
    }
    return res;
  }

  std::pair<bool, std::vector<std::vector<std::pair<PartitionID, HypernodeID>>>> bin_packing(std::vector<std::vector<double>> limits, std::vector<std::pair<double, std::pair<PartitionID, HypernodeID>>> nodes, std::vector<std::vector<std::vector<double>>> weights){
    std::vector<std::vector<std::pair<PartitionID, HypernodeID>>> packing(limits.size());
    for(HypernodeID hn = 0; hn < nodes.size(); hn++){
      double max_bg = std::numeric_limits<double>::max();
      PartitionID max_p = -1;
      for(PartitionID p = 0; p < limits.size(); p++){
        std::vector<double> new_limit = minus(limits[p], weights[nodes[hn].second.first][nodes[hn].second.second]);
        bool too_big = false;
        for(int d = 0; d < dimension; d++){
          if(new_limit[d] <= 0.0){
            too_big = true;
          }
        }
        if(too_big){
          continue;
        }
        double bg = vertical_imbalance(new_limit) - vertical_imbalance(limits[p]);
        if(bg < max_bg){
          max_p = p;
          max_bg = bg;
        }
      }
      if(max_p == -1){
        return {false, packing};
      }
      limits[max_p] = minus(limits[max_p], weights[nodes[hn].second.first][nodes[hn].second.second]);
      packing[max_p].push_back(nodes[hn].second);
    }
    /*std::cout << "limits\n";
    for(PartitionID p = 0; p < limits.size(); p++){
      for(int d = 0; d < dimension; d++){
        std::cout << limits[p][d] << "\n";
        if(limits[p][d] < 0.0){
          return {false, packing};
        }
      }
    }*/
    return {true, packing};
  }

  std::pair<bool, std::vector<std::vector<std::pair<PartitionID, HypernodeID>>>> bin_packing_best_fit(std::vector<std::vector<double>> limits, std::vector<std::pair<double, std::pair<PartitionID, HypernodeID>>> nodes, std::vector<std::vector<std::vector<double>>> weights){
    std::vector<std::vector<std::pair<PartitionID, HypernodeID>>> packing(limits.size());
    for(HypernodeID hn = 0; hn < nodes.size(); hn++){
      double min_bg = std::numeric_limits<double>::max();
      PartitionID max_p = -1;
      for(PartitionID p = 0; p < limits.size(); p++){
        std::vector<double> new_limit = minus(limits[p], weights[nodes[hn].second.first][nodes[hn].second.second]);
        bool too_big = false;
        double this_bg = 0.0;
        for(int d = 0; d < dimension; d++){
          if(new_limit[d] <= 0.0){
            too_big = true;
          }
          this_bg += new_limit[d];

        }
        if(too_big){
          continue;
        }
        if(min_bg > this_bg){
          max_p = p;
          min_bg = this_bg;
        }
      }
      if(max_p == -1){
        return {false, packing};
      }
      limits[max_p] = minus(limits[max_p], weights[nodes[hn].second.first][nodes[hn].second.second]);
      packing[max_p].push_back(nodes[hn].second);
    }
    for(PartitionID p = 0; p < limits.size(); p++){
      for(int d = 0; d < dimension; d++){
        if(limits[p][d] < 0.0){
          return {false, packing};
        }
      }
    }
    return {true, packing};
  }





  


  std::pair<bool, std::vector<std::vector<std::pair<PartitionID, HypernodeID>>>> bin_packing_random(std::vector<std::vector<double>> limits, std::vector<std::pair<double, std::pair<PartitionID, HypernodeID>>> nodes, std::vector<std::vector<std::vector<double>>> weights){
    std::vector<std::vector<std::pair<PartitionID, HypernodeID>>> packing(limits.size());
    std::cout << "binpack\n";
    HypernodeID number_random_assertions = std::rand() % nodes.size();
    for(HypernodeID hn = 0; hn < number_random_assertions; hn++){
      PartitionID to = -1;
      for(PartitionID p = 0; p < limits.size(); p++){
        to = std::rand() % limits.size();
        std::vector<double> new_limit = minus(limits[p], weights[nodes[hn].second.first][nodes[hn].second.second]);
        bool too_big = false;
        for(int d = 0; d < dimension; d++){
          if(new_limit[d] <= 0.0){
            too_big = true;
          }
        }
        if(too_big){
          continue;
        }
        break;
      }
      if(to = -1){
        return {false, packing};
      }
      limits[to] = minus(limits[to], weights[nodes[hn].second.first][nodes[hn].second.second]);
      packing[to].push_back(nodes[hn].second);
    }
    for(HypernodeID hn = number_random_assertions; hn < nodes.size(); hn++){
      double max_bg = std::numeric_limits<double>::max();
      PartitionID max_p = -1;
      for(PartitionID p = 0; p < limits.size(); p++){
        std::vector<double> new_limit = minus(limits[p], weights[nodes[hn].second.first][nodes[hn].second.second]);
        bool too_big = false;
        for(int d = 0; d < dimension; d++){
          if(new_limit[d] <= 0.0){
            too_big = true;
          }
        }
        if(too_big){
          continue;
        }
        double bg = vertical_imbalance(new_limit) - vertical_imbalance(limits[p]);
        if(bg < max_bg){
          max_p = p;
          max_bg = bg;
        }
      }
      if(max_p == -1){
        return {false, packing};
      }
      limits[max_p] = minus(limits[max_p], weights[nodes[hn].second.first][nodes[hn].second.second]);
      packing[max_p].push_back(nodes[hn].second);
    }
    for(PartitionID p = 0; p < limits.size(); p++){
      for(int d = 0; d < dimension; d++){
        if(limits[p][d] < 0.0){
          return {false, packing};
        }
      }
    }
    return {true, packing};
  }


  std::pair<bool, std::vector<std::vector<std::pair<PartitionID, HypernodeID>>>> bin_packing_totally_random(std::vector<std::vector<double>> limits, std::vector<std::pair<double, std::pair<PartitionID, HypernodeID>>> nodes, std::vector<std::vector<std::vector<double>>> weights){
    std::vector<std::vector<std::pair<PartitionID, HypernodeID>>> packing(limits.size());
    std::cout << "binpack\n";
    for(HypernodeID hn = 0; hn < nodes.size(); hn++){
      PartitionID to = -1;
      for(PartitionID p = 0; p < limits.size(); p++){
        to = std::rand() % limits.size();
        std::vector<double> new_limit = minus(limits[p], weights[nodes[hn].second.first][nodes[hn].second.second]);
        bool too_big = false;
        for(int d = 0; d < dimension; d++){
          if(new_limit[d] <= 0.0){
            too_big = true;
          }
        }
        if(too_big){
          continue;
        }
        break;
      }
      if(to = -1){
        return {false, packing};
      }
      limits[to] = minus(limits[to], weights[nodes[hn].second.first][nodes[hn].second.second]);
      packing[to].push_back(nodes[hn].second);
    }
    for(PartitionID p = 0; p < limits.size(); p++){
      for(int d = 0; d < dimension; d++){
        if(limits[p][d] < 0.0){
          return {false, packing};
        }
      }
    }
    return {true, packing};
  }


  /*std::pair<bool, std::vector<std::vector<std::pair<PartitionID, HypernodeID>>>> bin_packing_random(std::vector<std::vector<double>> limits, 
    std::vector<std::pair<double, std::pair<PartitionID, HypernodeID>>> nodes, std::vector<std::vector<std::vector<double>>> weights){
      std::vector<std::vector<std::pair<PartitionID, HypernodeID>>> packing(limits.size());
    for(HypernodeID hn = 0; hn < nodes.size(); hn++){
      double max_bg = std::numeric_limits<double>::max();
      PartitionID max_p = -1;
      for(PartitionID p = 0; p < limits.size(); p++){
        std::vector<double> new_limit = minus(limits[p], weights[nodes[hn].second.first][nodes[hn].second.second]);
        bool too_big = false;
        for(int d = 0; d < dimension; d++){
          if(new_limit[d] <= 0.0){
            too_big = true;
          }
        }
        if(too_big){
          continue;
        }
        double bg = vertical_imbalance(new_limit) - vertical_imbalance(limits[p]);
        if(bg < max_bg){
          max_p = p;
          max_bg = bg;
        }
      }
      if(max_p == -1){
        return {false, packing};
      }
      limits[max_p] = minus(limits[max_p], weights[nodes[hn].second.first][nodes[hn].second.second]);
      packing[max_p].push_back(nodes[hn].second);
    }
    return {true, packing};
  }*/
  template <typename GraphAndGainTypes>
  bool MDRebalancer<GraphAndGainTypes>::greedyRefiner(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                       vec<vec<Move>>* moves_by_part,
                                                       vec<Move>* moves_linear,
                                                       Metrics& best_metrics, 
                                                       std::vector<HypernodeID> nodes, bool balance, Gain& local_attributed_gain){                                                        
                                                   
    return 0;
  }

  struct S_Calculator{
    double factor_lowerBound = 0.0;
    double factor_maxbound;
    double factor_upperBound;
    double factor = 1.0;
    double search_threshold = 0.001;
    PartitionID k;
    std::vector<std::vector<double>> max_remaining_weight;
    std::vector<int> current_level_per_partition;
    std::vector<std::vector<std::vector<double>>> normalized_weights;
    //the second part of the pair is its rank in sorted total weights
    std::vector<std::vector<std::pair<double,HypernodeID>>> totalWeight;
    std::vector<std::vector<double>> normalized_partition_weights;
    std::vector<std::vector<double>> normalized_perfect_weights;
    std::vector<std::vector<double>> virtual_weight;

    std::vector<struct hypernodes_ordered> nodes_ordered;
    bool last = false;



    void initialize(std::vector<std::vector<std::vector<double>>> &nws, 
        std::vector<std::vector<std::pair<double,HypernodeID>>> &tW,
        std::vector<std::vector<double>> &npws,
        std::vector<std::vector<double>> &nperfws, double epsilon){
      factor_maxbound = 1.0 + epsilon;
      factor_upperBound = factor_maxbound;
      normalized_weights = nws;
      totalWeight = tW;
      normalized_partition_weights = npws;
      normalized_perfect_weights = nperfws;
      k = normalized_partition_weights.size();
      nodes_ordered.resize(k);
      virtual_weight.resize(k);
      max_remaining_weight.resize(k);
      current_level_per_partition.resize(k,0);
      for(PartitionID p = 0; p < k; p++){
        nodes_ordered[p].initialize(normalized_weights[p], totalWeight[p]);
      }
      current_level_per_partition.resize(k, 0);
    }

    bool hasNext(){
      return !last && (factor_upperBound - factor_lowerBound > search_threshold || factor_lowerBound == 0.0 && factor != 0.0);
    }

    void adjustFactor(bool success){
      if(factor == 0.0){
        last = true;
      }
      if(success){
        factor_lowerBound = factor;
      }
      else{
        factor_upperBound = factor;
      }
      if(factor_lowerBound == 0.0){
        factor = 2.0 * factor_upperBound - factor_maxbound;
      }
      else{
        factor = (factor_upperBound + factor_lowerBound) / 2.0;
      }
      if(factor_lowerBound == 0.0 && factor < search_threshold){
        factor = 0.0;
      }
    }
    
    std::vector<std::vector<bool>> calculate_S(bool success){ 
      

      auto smaller_equal = [](std::vector<double> a, std::vector<double> b){
        for(size_t i = 0; i < a.size(); i++){
          if(a[i] > b[i] + 0.00000000001){
            return false;
          }
        }
        return true;
      };
      std::vector<std::vector<double>> max_remaining_weight(k);
      for(PartitionID p = 0; p < k; p++){
        max_remaining_weight[p].resize(dimension);
        for(int d = 0; d < dimension; d++){
          max_remaining_weight[p][d] = normalized_perfect_weights[p][d] * factor;
        }
      }
      std::vector<std::vector<bool>> extracted(k);
      for(PartitionID p = 0; p < k; p++){
        extracted[p].resize(normalized_weights[p].size(), false);
        if(smaller_equal(normalized_partition_weights[p], max_remaining_weight[p])){
          continue;
        }
        bool improvement = true;
        int level = current_level_per_partition[p] - 1;
        double level_quality = std::numeric_limits<double>::max();
        while(improvement){
          std::vector<bool> imbalanced(dimension);
          std::vector<bool> used(normalized_weights[p].size(), false);
          int highest_would_use = 0;
          std::vector<double> this_run_weight;
          for(int d = 0; d < dimension; d++){
            this_run_weight.push_back(normalized_partition_weights[p][d]);
          }
          bool possible_to_balance = true;
          while(!smaller_equal(this_run_weight, max_remaining_weight[p])){
            double max = std::numeric_limits<double>::min();
            double min = std::numeric_limits<double>::max();
            for(int d = 0; d < dimension; d++){
              imbalanced[d] = this_run_weight[d] > max_remaining_weight[p][d];
              max = std::max(max, this_run_weight[d]);
              min = std::min(min, this_run_weight[d]);
            }
            for(int d = 0; d < dimension; d++){
              imbalanced[d] = this_run_weight[d] > (max + min) / 2.0;
            }
            std::pair<HypernodeID,int> next;
            do{
              next = nodes_ordered[p].getNext(imbalanced, level + 1);
            }while(next.second != -1 && used[next.first]);
            if(next.second == -1){
              possible_to_balance = false;
              break;
            }
            used[next.first] = true;
            highest_would_use = std::max(highest_would_use, next.second);
            for(int d = 0; d < dimension; d++){
              this_run_weight[d] -= normalized_weights[p][next.first][d];
            }
          }
              
          nodes_ordered[p].reset();
          if(!possible_to_balance){
            level++;
            continue;
          }
          if(smaller_equal(this_run_weight, max_remaining_weight[p])){
            double quality = 0.0;
            for(int d = 0; d < dimension; d++){
              quality += std::pow(max_remaining_weight[p][d] - this_run_weight[d], 2);
            }
            if(quality >= level_quality){
              improvement = false;
              level--;
            }
            else{
              level_quality = quality;
              extracted[p] = used;
              virtual_weight[p] = this_run_weight;
              if(highest_would_use <= level + 1){
                improvement = false;
              }
            }
          }
              
              
          level = std::max(level + 1, highest_would_use);
        }
        current_level_per_partition[p] = level;
      }
      return extracted;
    }
  

    std::vector<std::vector<double>> get_virtual_weight(){
      return virtual_weight;
    }
  };
  
  template <typename GraphAndGainTypes, typename PriorityType> struct Refiner{
    using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
    using GainCache = typename GraphAndGainTypes::GainCache;
    PartitionedHypergraph* phg;
    PriorityComputer<GraphAndGainTypes,PriorityType>* prio_computer;
    GainCache* _gain_cache;
    const Context* _context;
    //queue for rebalancing
    NodePQ<double>* queue;
    std::vector<tbb::concurrent_vector<std::pair<HypernodeID, HypernodeID>>> nodes_sorted;
    std::vector<int> moved_in_round;
    int round;

    Refiner(NodePQ<double>* q, PartitionedHypergraph* pg, PriorityComputer<GraphAndGainTypes,PriorityType>* pc,
      GainCache* gc, const Context* c){
      prio_computer = pc;
      _gain_cache = gc;
      _context = c;
      phg = pg;
      queue = q;
      moved_in_round.resize(phg->initialNumNodes(), -1);
    }

    void setup_nodes_sorted(){
      nodes_sorted.resize(mt_kahypar::dimension);
      for(HypernodeID hn : phg->nodes()){
        for(int i = 0; i < dimension; i++){
          ASSERT(hn < phg->initialNumNodes());
          nodes_sorted[i].push_back({phg->nodeWeight(hn).weights[i], hn});
        }
      }
      tbb::parallel_for(UL(0), mt_kahypar::dimension, [&](const size_t i){tbb::parallel_sort(nodes_sorted[i].begin(), nodes_sorted[i].end());});
    }

    bool refinement(Metrics& best_metrics){        
      ASSERT([&]{
        for(HypernodeID hn : phg->nodes()){
          if(phg->partID(hn) == -1) return false;
        }
        return true;
      }(), "fail");       
      setup_nodes_sorted();
      prio_computer->initialize();
      Gain quality = 0/*metrics::quality(*phg, *_context)*/;
      Gain local_attributed_gain = 0;
      //constraint_refinement(best_metrics, local_attributed_gain);
      ASSERT([&]{
        for(HypernodeID hn : phg->nodes()){
          if(phg->partID(hn) == -1) return false;
        }
        return true;
      }(), "fail");    
      unconstraint_refinement(best_metrics, local_attributed_gain);
      best_metrics.quality += local_attributed_gain;
      return true;
    }

    void rebalancing(vec<Move>* rebalance_moves, Metrics& best_metrics, Gain& local_attributed_gain){
      HypernodeID num_moves_before = rebalance_moves->size();
      ASSERT([&]{
        for(HypernodeID hn : phg->nodes()){
          if(phg->partID(hn) == -1) return false;
        }
        return true;
      }(), "fail");    
      if(!greedyRefiner(rebalance_moves, best_metrics, local_attributed_gain, _context->partition.l1_start_factor)){
        if(_context->partition.use_l1_factor_decrease){
          double factor = _context->partition.l1_start_factor;
          while(!metrics::isBalanced(*phg, *_context) && factor >= 0.97){
            std::cout << "factor: " << factor << "\n";
            double imbalance = 0.0;
            for(int d = 0; d < dimension; d++){
              double tmp_ib = 0;
              for(PartitionID p = 0; p < phg->k(); p++){
                tmp_ib = std::max(tmp_ib, (phg->partWeight(p).weights[d] - _context->partition.max_part_weights[p].weights[d]) * _context->partition.max_part_weights_inv[p][d]);
              }
              imbalance = std::max(imbalance, tmp_ib);
            }
            factor = 1.0 / std::pow((1.0 + imbalance), 1.2) * factor;
            for(size_t idx = 1; idx <= rebalance_moves->size() - num_moves_before; idx++){
              Move move = (*rebalance_moves)[rebalance_moves->size() - idx];
              ASSERT(move.from != -1);
              phg->changeNodePart(*_gain_cache, move.node, move.to, move.from, HypernodeWeight(true), []{}, [&](const SynchronizedEdgeUpdate& sync_update){
                            local_attributed_gain += (sync_update.pin_count_in_to_part_after == 1 ? sync_update.edge_weight : 0) +
                            (sync_update.pin_count_in_from_part_after == 0 ? -sync_update.edge_weight : 0);
                            
                          });
            }
            rebalance_moves->resize(num_moves_before);
            greedyRefiner(rebalance_moves, best_metrics, local_attributed_gain, factor);
          }
        }
        else if(_context->partition.use_fallback){
          HypernodeID size = rebalance_moves->size();
          std::cout << "fallback\n";
          vec<bool> L(phg->initialNumNodes(), false);
          if(!fallback(rebalance_moves, &L)) return;
          for(HypernodeID idx = size; idx < rebalance_moves->size(); idx++){
            ASSERT((*rebalance_moves)[idx].from == phg->partID((*rebalance_moves)[idx].node));
            ASSERT((*rebalance_moves)[idx].to != -1);
            phg->changeNodePart(*_gain_cache, (*rebalance_moves)[idx].node, (*rebalance_moves)[idx].from, (*rebalance_moves)[idx].to, HypernodeWeight(true), []{}, [&](const SynchronizedEdgeUpdate& sync_update){
                            local_attributed_gain += (sync_update.pin_count_in_to_part_after == 1 ? sync_update.edge_weight : 0) +
                            (sync_update.pin_count_in_from_part_after == 0 ? -sync_update.edge_weight : 0);
                            
                          });
          }
          if(L.size() > 0){
            greedyRefiner(rebalance_moves, best_metrics, local_attributed_gain, _context->partition.l1_start_factor);
          } 
          for(PartitionID p = 0; p < phg->k(); p++){
            if(!(phg->partWeight(p) <= _context->partition.max_part_weights[p])){
              std::cout << "part: " << p << "\n";
            }
          }
          ASSERT(metrics::isBalanced(*phg, *_context));
        }
      }
    }

    void constraint_refinement(Metrics& best_metrics, Gain& local_attributed_gain){
      for(int i = 0; i < 10; i++){
        Gain before_quality = local_attributed_gain;
        round++;
        if(i != 0){
          prio_computer->reinitialize();
        }
        simple_lp(NULL, best_metrics, 0.0, local_attributed_gain, false);
        if(before_quality == local_attributed_gain) break;
      }      
      if(!metrics::isBalanced(*phg, *_context)){
        vec<Move> m;
        rebalancing(&m, best_metrics, local_attributed_gain);  
        simple_lp(NULL, best_metrics, 0.0, local_attributed_gain, false);
      }
    }

    void unconstraint_refinement(Metrics& best_metrics, Gain& local_attributed_gain){
      double allowed_ib = _context->partition.allowed_imbalance_refine;
      std::cout << "before " << local_attributed_gain << "\n";
      for(int r = 0; r < 10; r++){
        vec<Move> moves;
        round++;
        HypernodeID before_moves = moves.size();
        Gain before_gain = local_attributed_gain;
        prio_computer->reinitialize();
        simple_lp(&moves, best_metrics, allowed_ib, local_attributed_gain, _context->partition.vertex_locking);
        std::cout << "lp: " << local_attributed_gain << "\n";
        rebalancing(&moves, best_metrics, local_attributed_gain);
        std::cout << "rb: " << local_attributed_gain << "\n";
        prio_computer->reinitialize();
        simple_lp(&moves, best_metrics, 0.0, local_attributed_gain, false);
        if(local_attributed_gain - before_gain > 0 || !metrics::isBalanced(*phg, *_context)){
          std::cout << "failure\n";
          for(HypernodeID i = 1; i <= moves.size() - before_moves; i++){
            HypernodeID idx = moves.size() - i;
            ASSERT(moves[idx].from != -1);
            phg->changeNodePart(*_gain_cache, moves[idx].node, moves[idx].to, moves[idx].from, HypernodeWeight(true), []{}, [&](const SynchronizedEdgeUpdate& sync_update){
                            local_attributed_gain += (sync_update.pin_count_in_to_part_after == 1 ? sync_update.edge_weight : 0) +
                            (sync_update.pin_count_in_from_part_after == 0 ? -sync_update.edge_weight : 0);                          
                          });
          }
          //allowed_ib /= 2.0;
        }
      }
    }

    void simple_lp(vec<Move>* moves_linear,Metrics& best_metrics, 
      double allowed_imbalance, Gain& local_attributed_gain, bool vertex_locking){
      std::vector<HypernodeWeight> max_part_weights(phg->k());
      for(PartitionID p = 0; p < phg->k(); p++){
        for(int d = 0; d < dimension; d++){
          max_part_weights[p].weights[d] = phg->partWeight(p).weights[d]; 
            if(max_part_weights[p].weights[d] < (std::ceil(static_cast<double>(_context->partition.max_part_weights[p].weights[d]) * (1.0 + allowed_imbalance)))){
              max_part_weights[p].weights[d] = std::ceil(static_cast<double>(_context->partition.max_part_weights[p].weights[d]) * (1.0 + allowed_imbalance));
            }
        }
      }
      Gain before_gain = local_attributed_gain;
      auto L1_balance_gain = [&](const HypernodeID node,const PartitionID to){
        PartitionID from = phg->partID(node);
        double gain = 0.0;
        for(int i = 0; i < dimension; i++){
          int32_t to_excess = std::max(0, std::min(phg->nodeWeight(node).weights[i], phg->partWeight(to).weights[i] + 
          phg->nodeWeight(node).weights[i] - _context->partition.max_part_weights[to].weights[i]));
          int32_t from_excess = std::max(0, std::min(phg->nodeWeight(node).weights[i], phg->partWeight(from).weights[i] - _context->partition.max_part_weights[from].weights[i]));
          gain += static_cast<double>(to_excess) * _context->partition.max_part_weights_inv[to][i]; 
          gain -= static_cast<double>(from_excess) * _context->partition.max_part_weights_inv[from][i]; 
        }        
        return gain;
      };
      auto helper = [&](std::pair<double,double> balance, double add) -> std::pair<double,double> {
        return {std::max(balance.first, add), balance.second + add*add};
      };
      auto betterBalanceKWay = [&](const HypernodeWeight& vwgt, int a1, PartitionID p1, int a2, PartitionID p2){ 
        std::pair<double,double> balance1, balance2; 
        for (int i = 0; i < dimension; i++) {
          balance1 = helper(balance1,(phg->partWeight(p1).weights[i] + a1 * vwgt.weights[i] - _context->partition.max_part_weights[p1].weights[i]) * _context->partition.max_part_weights_inv[p1][i]);
          balance2 = helper(balance2,(phg->partWeight(p2).weights[i] + a2 * vwgt.weights[i] - _context->partition.max_part_weights[p2].weights[i]) * _context->partition.max_part_weights_inv[p2][i]);
        }
        return balance2 < balance1;
      };      
      auto betterToMove = [&](HypernodeID h, PartitionID p){
        ASSERT(p != phg->partID(h));
        std::pair<double,double> balance1, balance2;        
        HypernodeID from = phg->partID(h);
        for (int i = 0; i < dimension; i++) {
          balance1 = helper(balance1, (phg->partWeight(from).weights[i] - _context->partition.max_part_weights[from].weights[i]) * _context->partition.max_part_weights_inv[from][i]);
          balance1 = helper(balance1, (phg->partWeight(p).weights[i] - _context->partition.max_part_weights[p].weights[i]) * _context->partition.max_part_weights_inv[p][i]);
          balance2 = helper(balance2, (phg->partWeight(from).weights[i] - phg->nodeWeight(h).weights[i] - _context->partition.max_part_weights[from].weights[i]) * _context->partition.max_part_weights_inv[from][i]);
          balance2 = helper(balance2, (phg->partWeight(p).weights[i] + phg->nodeWeight(h).weights[i] - _context->partition.max_part_weights[p].weights[i]) * _context->partition.max_part_weights_inv[p][i]);
        }
        return balance2 < balance1;
      };

      auto totalBalanceMetis = [&](){
        std::vector<double> imbalances;
        for(PartitionID p = 0; p < phg->k(); p++){
          for(int d = 0; d < dimension; d++){
            imbalances.push_back((phg->partWeight(p).weights[d] - _context->partition.max_part_weights[p].weights[d]) * _context->partition.max_part_weights_inv[p][d]);
          }
        }
        std::sort(imbalances.begin(), imbalances.end());
        return imbalances;
      };

      auto kahypar_betterToMove = [&](HypernodeID h, PartitionID p){
        return L1_balance_gain(h, p) < 0.0;
      };

      auto better_move_than_leave = [&](HypernodeID hn, PartitionID p){
        return _context->partition.refine_metis_move_criterion ? betterToMove(hn, p) : kahypar_betterToMove(hn,p);
      };
      auto refine_tiebreak = [&](HypernodeID hn, PartitionID p1, PartitionID p2){
        return _context->partition.refine_metis_tiebreak ? betterBalanceKWay(phg->nodeWeight(hn), 1, p1, 1, p2) :
          (L1_balance_gain(hn, p1) < L1_balance_gain(hn, p2));
      };
      
      if(!_context->partition.refine_random_order){
        NodePQ<PriorityType>* pq = prio_computer->get_pq();    
        HypernodeID num_loops = 0;
        Gain gain = 0;
        while(!pq->isEmpty()){
          num_loops++;
          std::pair<HypernodeID,Gain> max_node = pq->deleteMax();
          HypernodeID hn = max_node.first;
          std::pair<PartitionID,Gain> max_move = {-1, 0};
          for(PartitionID p = 0; p < phg->k(); p++){
            if(!(phg->partWeight(p) + phg->nodeWeight(hn) <= max_part_weights[p])) continue;
            if(p != phg->partID(hn) && (_gain_cache->gain(hn, phg->partID(hn), p) > max_move.second || 
              _gain_cache->gain(hn, phg->partID(hn), p) == max_move.second && 
                (max_move.first == -1 ? better_move_than_leave(hn, p) : 
                refine_tiebreak(hn, max_move.first, p)))){
              max_move = {p, _gain_cache->gain(hn, phg->partID(hn), p)};
            }
          }
          if(max_move.first != -1){
            Move move = {phg->partID(hn), max_move.first, hn, _gain_cache->gain(hn, phg->partID(hn), max_move.first)};
            gain += move.gain;
            std::vector<HyperedgeID> edges_with_gain_change;
            ASSERT(move.to != -1);
            phg->changeNodePart(*_gain_cache, move.node, move.from, move.to, HypernodeWeight(true), []{}, [&](const SynchronizedEdgeUpdate& sync_update) {
                            local_attributed_gain += (sync_update.pin_count_in_to_part_after == 1 ? sync_update.edge_weight : 0) +
                            (sync_update.pin_count_in_from_part_after == 0 ? -sync_update.edge_weight : 0);
                            if (!PartitionedHypergraph::is_graph && GainCache::triggersDeltaGainUpdate(sync_update)) {
                              edges_with_gain_change.push_back(sync_update.he);
                            }
                          });
            if(moves_linear != NULL){
              moves_linear->push_back({move.from, max_move.first, hn, local_attributed_gain});
            }
            if(vertex_locking){
              moved_in_round[hn] = round;
            }            
            prio_computer->performMove(move);                 
          }        
        }  
      }
      else{
        std::vector<HypernodeID> nodes;
        for(HypernodeID hn : phg->nodes()){
          nodes.push_back(hn);
        }
        utils::Randomize::instance().shuffleVector(nodes);
        HypernodeID num_loops = 0;
        Gain gain = 0;
        for(HypernodeID hn : nodes){            
          num_loops++;
          std::pair<PartitionID,Gain> max_move = {-1, 0};
          for(PartitionID p = 0; p < phg->k(); p++){
            if(!(phg->partWeight(p) + phg->nodeWeight(hn) <= max_part_weights[p])) continue;
            if(p != phg->partID(hn) && (_gain_cache->gain(hn, phg->partID(hn), p) > max_move.second || 
              _gain_cache->gain(hn, phg->partID(hn), p) == max_move.second && 
                (max_move.first == -1 ? better_move_than_leave(hn, p) : 
                refine_tiebreak(hn, max_move.first, p)))){
              max_move = {p, _gain_cache->gain(hn, phg->partID(hn), p)};
            }
          }
          if(max_move.first != -1){
            Move move = {phg->partID(hn), max_move.first, hn, _gain_cache->gain(hn, phg->partID(hn), max_move.first)};
            gain += move.gain;
            std::vector<HyperedgeID> edges_with_gain_change;
            ASSERT(move.to != -1);
            phg->changeNodePart(*_gain_cache, move.node, move.from, move.to, HypernodeWeight(true), []{}, [&](const SynchronizedEdgeUpdate& sync_update) {
                            local_attributed_gain += (sync_update.pin_count_in_to_part_after == 1 ? sync_update.edge_weight : 0) +
                            (sync_update.pin_count_in_from_part_after == 0 ? -sync_update.edge_weight : 0);
                            if (!PartitionedHypergraph::is_graph && GainCache::triggersDeltaGainUpdate(sync_update)) {
                              edges_with_gain_change.push_back(sync_update.he);
                            }
                          });
            if(moves_linear != NULL){
              moves_linear->push_back({move.from, max_move.first, hn, local_attributed_gain});
            }
            if(vertex_locking){
              moved_in_round[hn] = round;
            }                  
          }        
        }         
      }     
    }

    bool greedyRefiner(vec<Move>* moves_linear,
      Metrics& best_metrics, Gain& local_attributed_gain, double l1_factor){
      return greedyRefiner(moves_linear, best_metrics, local_attributed_gain, l1_factor, NULL);
    }

    bool greedyRefiner(vec<Move>* moves_linear,
      Metrics& best_metrics, Gain& local_attributed_gain, double l1_factor, vec<HypernodeID> *L){                                                       
      std::vector<HypernodeWeight> max_part_weights_modified(phg->k());
      for(PartitionID p = 0; p < phg->k(); p++){
        max_part_weights_modified[p] = l1_factor * _context->partition.max_part_weights[p];
      }                                                       

      auto moveout_gain = [&](HypernodeID hn, PartitionID from){
        double gain = 0.0;
        for(int i = 0; i < dimension; i++){
          int32_t from_excess = std::max(0, std::min(phg->nodeWeight(hn).weights[i], phg->partWeight(from).weights[i] - max_part_weights_modified[from].weights[i]));
          gain -= static_cast<double>(from_excess) * _context->partition.max_part_weights_inv[from][i]; 
        }        
        return gain;
      };
      auto movein_gain = [&](HypernodeID hn, PartitionID to){
        double gain = 0.0;
        for(int i = 0; i < dimension; i++){
          int32_t to_excess = std::max(0, std::min(phg->nodeWeight(hn).weights[i], phg->partWeight(to).weights[i] + 
          phg->nodeWeight(hn).weights[i] - max_part_weights_modified[to].weights[i]));
          gain += static_cast<double>(to_excess) * _context->partition.max_part_weights_inv[to][i]; 
        }        
        return gain;
      };                                                            
      auto is_positive_move = [&](Move_Internal move){
        return move.balance < 0.0 || move.balance <= 0.0 && move.gain > 0;
      };
      int counter = 1;
      auto get_max_move = [&](HypernodeID node){
        PartitionID p_max = -1;
        Move_Internal max_move = {std::numeric_limits<double>::max(), 0, 0.0};
        ASSERT(node < phg->initialNumNodes());
        PartitionID part = phg->partID(node);         
        double outgain = moveout_gain(node, part);
        for(PartitionID p = 0; p < phg->k(); p++){
          if(p == part) continue;
          double ingain = movein_gain(node,p);
          Move_Internal move = {0.0, _gain_cache->gain(node, part, p), ingain + outgain};
          move.recomputeBalance();
          if(is_positive_move(move) && (p_max == -1 || move < max_move)){
            max_move = move;
            p_max = p;
          }
        }
        return std::pair<PartitionID, Move_Internal>(p_max, max_move);
      };
      std::vector<HypernodeID> nums_per_part(phg->k(), 0);
      if(L == NULL){
        for(HypernodeID hn : phg->nodes()){   
          ASSERT(hn < phg->initialNumNodes());
          std::pair<PartitionID,Move_Internal> max_move =  get_max_move(hn);
          nums_per_part[phg->partID(hn)]++;    
          if(max_move.first != -1){
            queue->changeKey(hn, max_move.second.gain_and_balance);
          }
        }
      }
      else{
        for(HypernodeID hn : (*L)){   
          ASSERT(hn < phg->initialNumNodes());
          std::pair<PartitionID,Move_Internal> max_move =  get_max_move(hn);    
          nums_per_part[phg->partID(hn)]++;  
          if(max_move.first != -1){
            queue->changeKey(hn, max_move.second.gain_and_balance);
          }
        }
      }
      int32_t estimated_num_moves = 0;
      for(PartitionID p = 0; p < phg->k(); p++){
        for(int d = 0; d < dimension; d++){
          estimated_num_moves = std::max(estimated_num_moves, static_cast<int32_t>(std::ceil(nums_per_part[p] * 
            (std::max(0, phg->partWeight(p).weights[d] - _context->partition.max_part_weights[p].weights[d])) * 
            _context->partition.max_part_weights_inv[p][d])));
        }
      }

      std::cout << estimated_num_moves << " enm\n";

      PartitionID imbalanced = 0;
      for(PartitionID p = 0; p < phg->k(); p++){
        if(phg->partWeight(p) > _context->partition.max_part_weights[p]){
          imbalanced++;
        }
      }

      
      int other_counter = 0;
      const int UPDATE_FREQUENCY = std::ceil(_context->partition.update_frequency * estimated_num_moves);

      std::vector<HypernodeWeight> highest_part_weights;
      std::vector<HypernodeWeight> lowest_part_weights;
      for(PartitionID p = 0; p < phg->k(); p++){
        highest_part_weights.push_back(phg->partWeight(p));
        lowest_part_weights.push_back(phg->partWeight(p));
      }
      std::vector<bool> is_in_L;
      if(L != NULL){
        is_in_L.resize(phg->initialNumNodes(), false);
        for(HypernodeID hn : (*L)){
          is_in_L[hn] = true;
        }
      }
      auto completeUpdate = [&](std::vector<HypernodeID> *changed_nodes){
        for(HypernodeID hn : phg->nodes()){
          changed_nodes->push_back(hn);
        }
        /*for(int i = 0; i < dimension; i++){          
          HypernodeID min_affected_node = nodes_sorted[i].size() - 1;
          for(PartitionID p = 0; p < phg->k(); p++){
            if(phg->partWeight(p).weights[i] < _context->partition.max_part_weights[p].weights[i] && phg->partWeight(p).weights[i] < highest_part_weights[p].weights[i]){
              int prior_diff = _context->partition.max_part_weights[p].weights[i] - highest_part_weights[p].weights[i];
              while(min_affected_node >= 0 && nodes_sorted[i][min_affected_node].first > prior_diff || prior_diff < 0){
                changed_nodes->push_back(nodes_sorted[i][min_affected_node].second);
                if(min_affected_node == 0) break;
                min_affected_node--;
              }
            }
            else if(phg->partWeight(p).weights[i] > _context->partition.max_part_weights[p].weights[i] && phg->partWeight(p).weights[i] > lowest_part_weights[p].weights[i]){
              int prior_diff = lowest_part_weights[p].weights[i] -  _context->partition.max_part_weights[p].weights[i];
              while(min_affected_node >= 0 && nodes_sorted[i][min_affected_node].first > prior_diff || prior_diff < 0){
                changed_nodes->push_back(nodes_sorted[i][min_affected_node].second);
                if(min_affected_node == 0) break;
                min_affected_node--;
              }
            }
          }            
        } 
        for(PartitionID p = 0; p < phg->k(); p++){
          highest_part_weights[p] = phg->partWeight(p);
          lowest_part_weights[p] = phg->partWeight(p);
        }*/
      };

      auto update_nodes = [&](std::vector<HypernodeID> *changed_nodes){
        for(size_t i = 0; i < changed_nodes->size(); i++){
          ASSERT((*changed_nodes)[i] < phg->initialNumNodes());
          if(moved_in_round[(*changed_nodes)[i]] < round && (L == NULL || is_in_L[(*changed_nodes)[i]])){
            std::pair<PartitionID, Move_Internal> best_move = get_max_move((*changed_nodes)[i]);
            if(/*(counter % UPDATE_FREQUENCY == 0) && */best_move.first == -1){
              queue->invalidate((*changed_nodes)[i]);
            }
            else if(/*(counter % UPDATE_FREQUENCY == 0) ||*/!queue->isValid((*changed_nodes)[i]) || queue->get_entry((*changed_nodes)[i]) != best_move.second.gain_and_balance){
              queue->changeKey((*changed_nodes)[i], best_move.second.gain_and_balance);
            }
          }         
        }
      };

      while(!queue->isEmpty() && (imbalanced != 0) ){
        std::pair<PartitionID, Move_Internal> max_move;
        max_move.first = -1;
        std::pair<HypernodeID, double> max_node = queue->getMax();
        ASSERT(max_node.first < phg->initialNumNodes());
        max_move = get_max_move(max_node.first);
        if(max_move.first == -1){
          queue->deleteMax();
          other_counter++;
        }
        else if(max_node.second < max_move.second.gain_and_balance){
          if(_context->partition.rebalancer_approximate_worsened_move){

          }
          else{
            if(is_positive_move(max_move.second)){
              queue->changeKey(max_node.first, max_move.second.gain_and_balance);
            }
            else{
              queue->deleteMax();
            }
            if(!queue->firstInPQ(max_move.first)) other_counter++;
          }
        }
        else{
          ASSERT(max_node.first < phg->initialNumNodes());
          ASSERT(phg->partID(max_node.first) != -1);
          Move move = {phg->partID(max_node.first), max_move.first, max_node.first, max_move.second.gain};
          queue->deleteMax();
          counter++;      
          imbalanced += (phg->partWeight(move.from) - phg->nodeWeight(move.node) > _context->partition.max_part_weights[move.from])
            - (phg->partWeight(move.from) > _context->partition.max_part_weights[move.from])
            - (phg->partWeight(move.to) > _context->partition.max_part_weights[move.to])
            + (phg->partWeight(move.to) + phg->nodeWeight(move.node) > _context->partition.max_part_weights[move.to]);
          std::vector<HyperedgeID> edges_with_gain_change;         
          ASSERT(move.from != move.to);
          ASSERT(move.to != -1);
          phg->changeNodePart(*_gain_cache, move.node, move.from, move.to, HypernodeWeight(true), []{}, [&](const SynchronizedEdgeUpdate& sync_update) {
                          local_attributed_gain += (sync_update.pin_count_in_to_part_after == 1 ? sync_update.edge_weight : 0) +
                          (sync_update.pin_count_in_from_part_after == 0 ? -sync_update.edge_weight : 0);
                          if (!PartitionedHypergraph::is_graph && GainCache::triggersDeltaGainUpdate(sync_update)) {
                            edges_with_gain_change.push_back(sync_update.he);
                          }
                        });
          queue->move(move.from, move.to);
          moved_in_round[move.node] = round;
          for(int d = 0; d < dimension; d++){
            lowest_part_weights[move.from].weights[d] = std::min(lowest_part_weights[move.from].weights[d], phg->partWeight(move.from).weights[d]);
            highest_part_weights[move.to].weights[d] = std::max(highest_part_weights[move.to].weights[d], phg->partWeight(move.to).weights[d]);
          }
          if(moves_linear != NULL){
            move.gain = local_attributed_gain;
            moves_linear->push_back(move);
          }
          if(imbalanced == 0) break;

          std::vector<HypernodeID> changed_nodes;
          for(HyperedgeID& he : edges_with_gain_change){
            for(HypernodeID hn : phg->pins(he)){
              changed_nodes.push_back(hn);
            }
          }
          if constexpr (PartitionedHypergraph::is_graph) {
            for (const auto e : phg->incidentEdges(move.node)) {
              HypernodeID v = phg->edgeTarget(e);
              changed_nodes.push_back(v);            
            }
          }
          if(counter % UPDATE_FREQUENCY == 0){            
            completeUpdate(&changed_nodes);            
          }
          update_nodes(&changed_nodes);
          if(counter % UPDATE_FREQUENCY != 0){
            for(HypernodeID hn : queue->updateRequired()){
              ASSERT(hn < phg->initialNumNodes());
              std::pair<PartitionID,Move_Internal> max_move = get_max_move(hn);
              if(max_move.first == -1) continue;
              queue->update(hn, max_move.second.gain_and_balance);
            }
            
          }          
        }
        if(queue->isEmpty()){
          std::vector<HypernodeID> cn2;
          completeUpdate(&cn2);
          update_nodes(&cn2);                  
        }                                                 
      }      
      std::cout << counter << " " << other_counter << " " << local_attributed_gain << " " << imbalanced << " " << queue->isEmpty() << " " << queue->num_swaps() << "\n";
      if(imbalanced != 0){
        for(PartitionID p = 0; p < phg->k(); p++){
          for(int d = 0; d < dimension; d++){
            std::cout << phg->partWeight(p).weights[d] << "\n";
          }
          std::cout << "\n";
        }
        
        ASSERT([&]{
          for(HypernodeID hn : phg->nodes()){
            if(get_max_move(hn).first != -1 && (moved_in_round[hn] < round && (L == NULL || is_in_L[hn]))){
              std::cout << hn << " " << phg->partID(hn) << " " << get_max_move(hn).first << " " << get_max_move(hn).second.gain << " " << get_max_move(hn).second.balance << "\n";
              for(int d = 0; d < dimension; d++){
                std::cout << phg->nodeWeight(hn).weights[d] << " " << phg->partWeight(phg->partID(hn)).weights[d] << " " << _context->partition.max_part_weights[0].weights[d] << "\n";
              }
              return false;
            }
          }
          return true;
        }(), "didnt correct update");
      }
      return imbalanced == 0;
    }

    bool fallback(vec<Move> *fallback_moves, vec<bool> *L){
      const double threshold = _context->partition.L_threshold;
      auto get_heaviest_dim = [&](HypernodeWeight nw){
        int dim = -1;
        double weight = -1.0;
        for(int d = 0; d < dimension; d++){
          double w_n = nw.weights[d] * _context->partition.max_part_weights_inv[0][d];
          if(w_n > weight){
            weight = w_n;
            dim = d;
          }
        }
        return dim;
      };

      auto get_normalized_weight = [&](HypernodeWeight nw){
        double res = 0.0;
        for(int d = 0; d < dimension; d++){
          res += nw.weights[d] * _context->partition.max_part_weights_inv[0][d];
        }
        return res;
      };

      auto penalty = [&](HypernodeWeight weight){
        double avg = 0.0;
        std::vector<double> normalized_weights(dimension);
        for(int d = 0; d < dimension; d++){
          normalized_weights[d] = weight.weights[d] * _context->partition.max_part_weights_inv[0][d];
          avg += normalized_weights[d];
        }
        avg /= phg->k();
        double imbalance = 0.0;
        for(int d = 0; d < dimension; d++){
          imbalance += (normalized_weights[d] - avg) *  (normalized_weights[d] - avg);
        }
        return imbalance * (1.0 - avg);
      };

      std::vector<HypernodeID> S;
      std::vector<bool> is_in_S(phg->initialNumNodes(), false);

      std::vector<HypernodeWeight> virtual_weight(phg->k());

      for(PartitionID p = 0; p < phg->k(); p++){
        virtual_weight[p] = phg->partWeight(p);
      }
      for(HypernodeID hn : phg->nodes()){
        int dim = get_heaviest_dim(hn);
        ASSERT(dim >= 0);
        ASSERT(dim < dimension);
        ASSERT(phg->partID(hn) != -1, hn);
        if(phg->nodeWeight(hn).weights[dim] * _context->partition.max_part_weights_inv[phg->partID(hn)][dim] > threshold){
          (*L)[hn] = true;
          virtual_weight[phg->partID(hn)] -= phg->nodeWeight(hn);
        }
      }

      struct lazyPQComputer<GraphAndGainTypes> PQComputer = lazyPQComputer<GraphAndGainTypes>(phg, _gain_cache, _context, L); 
      Binpacker<GraphAndGainTypes> binpacker = Binpacker<GraphAndGainTypes>(&S, phg, _context);
      
      std::vector<std::vector<AddressablePQ<HypernodeID,double>>> queues(phg->k());

      std::cout << "imbalances:\n";
      for(PartitionID p = 0; p < phg->k(); p++){
        for(int d = 0; d < dimension; d++){
          std::cout << phg->partWeight(p).weights[d] - _context->partition.max_part_weights[p].weights[d] << " ";
        }
        std::cout << "\n";
      }

      auto print_parts = [&](){
        for(PartitionID p = 0; p < phg->k(); p++){
          for(int d = 0; d < dimension; d++){
            std::cout << virtual_weight[p].weights[d] * _context->partition.max_part_weights_inv[p][d] << " ";
          }
          std::cout << " | ";
        }
        std::cout << " ||| ";
      };

      double S_weight = 0.0;
      auto extract = [&](PartitionID p){
        print_parts();
        int heaviest_dim = get_heaviest_dim(virtual_weight[p]);
        HypernodeID hn = PQComputer.deleteMax(p, heaviest_dim);
        virtual_weight[p] -= phg->nodeWeight(hn);
        S.push_back(hn);
        S_weight += get_normalized_weight(phg->nodeWeight(hn));
      };
      for(PartitionID p = 0; p < phg->k(); p++){
        while(!(virtual_weight[p] <= _context->partition.max_part_weights[p])){
          extract(p);
        }
      }
      double goal = S_weight;
      double starting_goal = goal;

      HypernodeID starting_index = S.size();

      auto select_max_pen_p = [&](){
        double max_pen = 0.0;
        double max_p = -1;
        for(PartitionID p = 0; p < phg->k(); p++){
          double pen = penalty(virtual_weight[p]);
          if(pen > max_pen){
            max_pen = pen;
            max_p = p;
          }
        }
        return max_p;
      };

      auto select_heaviest_p = [&](){
        double max_p = -1;
        double max_weight = 0.0;
        for(PartitionID p = 0; p < phg->k(); p++){
          for(int d = 0; d < dimension; d++){
            if(phg->partWeight(p).weights[d] * _context->partition.max_part_weights_inv[p][d] > max_weight){
              max_weight = phg->partWeight(p).weights[d] * _context->partition.max_part_weights_inv[p][d];
              max_p = p;
            }
          }
        }
        return max_p;
      };

      while(true){
        //std::cout << "goal\n\n" << goal << " " << S_weight << "\n";
        while(S_weight < goal){
          //std::cout << "sweight: " << S_weight << " " << goal << "\n\n\n";
          PartitionID max_p = _context->partition.fallback_extract_equally ? select_heaviest_p() : select_max_pen_p();
          extract(max_p);
        }
        if(binpacker.binpack(S.size(), virtual_weight, penalty)){
          std::cout << "success\n";
          break;
        }
        else{
          std::cout << "nosucc\n";
        }
        ASSERT([&]{
          std::vector<HypernodeWeight> test;
          for(int p = 0; p < phg->k(); p++){
            test.push_back(virtual_weight[p]);
          }
          for(HypernodeID hn : S){
            test[phg->partID(hn)] += phg->nodeWeight(hn);
          }
          for(PartitionID p = 0; p < phg->k(); p++){
            if(test[p] != phg->partWeight(p)) return false;
          }
          return true;
        }(), "bwugi");
        goal *= 2.0;
        if(goal > dimension * phg->k()) return false;
        std::cout << "goal: " << goal << " ";
      }
      HypernodeID last_idx = S.size() - 1;
      HypernodeID succ_idx = S.size();
      double lower = std::max(goal / 2.0, starting_goal);
      double upper = goal;
      double current = (upper + lower) / 2.0;
      while(upper - lower > threshold){
        std::cout << "current " << current << " ";
        while(S_weight > current){
          S_weight -= get_normalized_weight(phg->nodeWeight(S[last_idx]));
          virtual_weight[phg->partID(S[last_idx])] += phg->nodeWeight(S[last_idx]);
          last_idx--;
        }
        while(S_weight < current){
          last_idx++;
          virtual_weight[phg->partID(S[last_idx])] -= phg->nodeWeight(S[last_idx]);
          S_weight += get_normalized_weight(phg->nodeWeight(S[last_idx]));          
        }
        ASSERT(last_idx + 1 >= starting_index);
        if(binpacker.binpack(last_idx + 1, virtual_weight, penalty)){
          std::cout << "succ2 ";
          succ_idx = last_idx + 1;
          upper = current;
        }
        else{
          lower = current;
        }
        int c = 0;
        for(HypernodeID hn : S){
          if(binpacker.partitioning[hn] != -1){
            c++;
          }
        } 
        std::cout << "sizes " << c << " " << succ_idx << "\n";
        ASSERT(c == succ_idx);
        ASSERT([&]{
          std::vector<HypernodeWeight> test;
          for(int p = 0; p < phg->k(); p++){
            test.push_back(virtual_weight[p]);
          }
          for(HypernodeID h = 0; h < last_idx + 1; h++){
            HypernodeID hn = S[h];
            test[phg->partID(hn)] += phg->nodeWeight(hn);
          }
          for(PartitionID p = 0; p < phg->k(); p++){
            if(test[p] != phg->partWeight(p)) return false;
          }
          return true;
        }(), "bwugi");
        current = (upper + lower) / 2.0;
      }
      int counter = 0;
      std::cout << "S: ";
      for(HypernodeID hn : S){
        for(int d = 0; d < dimension; d++){
          std::cout << phg->nodeWeight(hn).weights[d] << " ";
        }
        std::cout << "| ";
        if(binpacker.partitioning[hn] != -1){
          counter++;
        }
        if(binpacker.partitioning[hn] != -1 && binpacker.partitioning[hn] != phg->partID(hn)){
          Move move = {phg->partID(hn), binpacker.partitioning[hn], hn, 0};
          //std::cout << "move: " << move.node << " " << move.from << " " << move.to << "\n";
          ASSERT(move.from == phg->partID(hn));
          fallback_moves->push_back(move);
        }
      }
      std::cout << "sizes " << counter << " " << succ_idx << "\n";
      ASSERT(counter == succ_idx);
      return true;
    }
  };


  template <typename GraphAndGainTypes>
  bool MDRebalancer<GraphAndGainTypes>::refineInternal(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                       vec<vec<Move>>* moves_by_part,
                                                       vec<Move>* moves_linear,
                                                       Metrics& best_metrics) {                                       
    /*utils::Timer& timer = utils::Utilities::instance().getTimer(context.utility_id);
    timer.start_timer("tl_refine", "Top level refinement");*/
    vec<Move> moves;
    auto start = std::chrono::high_resolution_clock::now();                                                
    PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
    resizeDataStructuresForCurrentK();
    // This function is passed as lambda to the changeNodePart function and used
    // to calculate the "real" delta of a move (in terms of the used objective function).
    auto objective_delta = [&](const SynchronizedEdgeUpdate& sync_update) {
      _gain.computeDeltaForHyperedge(sync_update);
    };

    if(_context.partition.refine_metis_gain){
      SimplePQ<double> pq = SimplePQ<double>(phg.initialNumNodes());
      MetisMetricPriorityComputer<GraphAndGainTypes> pc = MetisMetricPriorityComputer<GraphAndGainTypes>(&phg, &_gain_cache, &pq);
      if(_context.partition.rebalancer_several_pqs){
        pd_PQ<PartitionedHypergraph,double> queue = pd_PQ<PartitionedHypergraph,double>(hypergraph, _context);
        Refiner<GraphAndGainTypes,double> refiner = Refiner<GraphAndGainTypes,double>(&queue, &phg, &pc, &_gain_cache, &_context);
        refiner.refinement(best_metrics);
      }
      else{
        SimplePQ<double> queue = SimplePQ<double>(phg.initialNumNodes());
        Refiner<GraphAndGainTypes,double> refiner = Refiner<GraphAndGainTypes,double>(&queue, &phg, &pc, &_gain_cache, &_context);
        refiner.refinement(best_metrics); 
      }     
    }
    else{
      SimplePQ<Gain> pq = SimplePQ<Gain>(phg.initialNumNodes());
      GainPriorityComputer<GraphAndGainTypes> pc = GainPriorityComputer<GraphAndGainTypes>(&phg, &_gain_cache, &pq);
      if(_context.partition.rebalancer_several_pqs){
        pd_PQ<PartitionedHypergraph,double> queue = pd_PQ<PartitionedHypergraph,double>(hypergraph, _context);
        Refiner<GraphAndGainTypes,Gain> refiner = Refiner<GraphAndGainTypes,Gain>(&queue, &phg, &pc, &_gain_cache, &_context);
        refiner.refinement(best_metrics);
      }
      else{
        SimplePQ<double> queue = SimplePQ<double>(phg.initialNumNodes());
        Refiner<GraphAndGainTypes,Gain> refiner = Refiner<GraphAndGainTypes,Gain>(&queue, &phg, &pc, &_gain_cache, &_context);
        refiner.refinement(best_metrics); 
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start; // this is in ticks
    std::chrono::milliseconds d = std::chrono::duration_cast<std::chrono::milliseconds>(diff); // ticks to time

    std::cout << "Duration: " << d.count() << "\n";
  
    return true;
  }

  template <typename GraphAndGainTypes>
  void MDRebalancer<GraphAndGainTypes>::initializeImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph) {
    auto& phg = utils::cast<PartitionedHypergraph>(hypergraph);

    if (!_gain_cache.isInitialized()) {
      _gain_cache.initializeGainCache(phg);
    }
  }

  template <typename GraphAndGainTypes>
  bool MDRebalancer<GraphAndGainTypes>::refineImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                  const vec<HypernodeID>& , Metrics& best_metrics, double) {
    return refineInternal(hypergraph, nullptr, nullptr, best_metrics);
  }

  template <typename GraphAndGainTypes>
  bool MDRebalancer<GraphAndGainTypes>::refineAndOutputMovesImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                                    const vec<HypernodeID>& ,
                                                                    vec<vec<Move>>& moves_by_part,
                                                                    Metrics& best_metrics,
                                                                    const double) {
    return refineInternal(hypergraph, &moves_by_part, nullptr, best_metrics);
  }

  template <typename GraphAndGainTypes>
  bool MDRebalancer<GraphAndGainTypes>::refineAndOutputMovesLinearImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                                          const vec<HypernodeID>& ,
                                                                          vec<Move>& moves,
                                                                          Metrics& best_metrics,
                                                                          const double) {
    return refineInternal(hypergraph, nullptr, &moves, best_metrics);
  }

  // explicitly instantiate so the compiler can generate them when compiling this cpp file
  namespace {
  #define MD_REBALANCER(X) MDRebalancer<X>
  }

  // explicitly instantiate so the compiler can generate them when compiling this cpp file
  INSTANTIATE_CLASS_WITH_VALID_TRAITS(MD_REBALANCER)




  template <typename GraphAndGainTypes>
  bool MDRebalancer<GraphAndGainTypes>::labelPropagation(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                       vec<vec<Move>>* moves_by_part,
                                                       vec<Move>* moves_linear,
                                                       Metrics& best_metrics, 
                                                       std::vector<HypernodeID> nodes, double allowed_imbalance, bool balance){

    PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
    std::vector<HypernodeWeight> max_part_weights(phg.k());
    auto get_imbalances = [&](){
      std::vector<double> ib(phg.k(), 0.0);
      for(PartitionID p = 0; p < phg.k(); p++){
        for(int i = 0; i < dimension; i++){
          ib[i] = std::max(ib[i], (phg.partWeight(p).weights[i] - _context.partition.max_part_weights[p].weights[i]) *
           _context.partition.max_part_weights_inv[p][i]);
        }
      }
      return ib;
    };      
    

    std::vector<double> imbalances = get_imbalances();
    for(PartitionID p = 0; p < phg.k(); p++){
      for(int i = 0; i < dimension; i++){
        max_part_weights[p].weights[i] = std::ceil((1.0 + allowed_imbalance + imbalances[i]) * _context.partition.max_part_weights[p].weights[i]); 
      }
    }

    if(balance){
      for(PartitionID p = 0; p < phg.k(); p++){
        for(int i = 0; i < dimension; i++){
          max_part_weights[p].weights[i] = _context.partition.max_part_weights[p].weights[i]; 
        }
      }
    }
    else{
      for(PartitionID p = 0; p < phg.k(); p++){
        for(int i = 0; i < dimension; i++){
          max_part_weights[p].weights[i] = std::max(max_part_weights[p].weights[i], _context.partition.max_part_weights[p].weights[i]); 
        }
      }
    }

    auto is_smaller = [&](int n, int scalar, HypernodeWeight x, HypernodeWeight y, HypernodeWeight z){
      for(int i = 0; i < n; i++){
        if(scalar * x.weights[i] + y.weights[i] > z.weights[i]){
          return false;
        }
      }
      return true;
    };


    auto BetterBalanceKWay = [&](HypernodeWeight vwgt, 
        int a1, PartitionID p1, 
        int a2, PartitionID p2)
    { 
      double nrm1=0.0, nrm2=0.0, max1=0.0, max2=0.0;

      for (int i = 0; i < dimension; i++) {
        double tmp = (phg.partWeight(p1).weights[i] + a1 * vwgt.weights[i] - max_part_weights[p1].weights[i]) / max_part_weights[p1].weights[i];
        //printf("BB: %d %+.4f ", (int)i, (float)tmp);
        nrm1 += tmp*tmp;
        max1 = (tmp > max1 ? tmp : max1);

        tmp = (phg.partWeight(p2).weights[i] + a2 * vwgt.weights[i] - max_part_weights[p2].weights[i]) / max_part_weights[p2].weights[i];
        //printf("%+.4f ", (float)tmp);
        nrm2 += tmp*tmp;
        max2 = (tmp > max2 ? tmp : max2);

      }

      if (max2 < max1)
        return 1;

      if (max2 == max1 && nrm2 < nrm1)
        return 1;

      return 0;
    };

    for(PartitionID p = 0; p < phg.k(); p++){
      for(int d = 0; d < dimension; d++){
        std::cout << "maxweight " << max_part_weights[p].weights[d] << " " << phg.partWeight(p).weights[d] << "\n";
      }
    }


    AddressablePQ<HypernodeID,double> pq;
    std::vector<std::pair<HypernodeID,HypernodeID>> id_ed(phg.initialNumNodes());
    std::vector<std::vector<HypernodeID>> num_neighbors(phg.initialNumNodes());
    std::vector<PartitionID> nnp(phg.initialNumNodes(), 0);
    for(HypernodeID hn : phg.nodes()){
      id_ed[hn] = std::pair<HypernodeID,HypernodeID>(0,0);
      num_neighbors[hn].resize(phg.k(), 0);
      for(HyperedgeID he : phg.incidentEdges(hn)){
        for(HypernodeID h : phg.pins(he)){
          if(h == hn) continue;
          if(phg.partID(h) == phg.partID(hn)){
            id_ed[hn].first += phg.edgeWeight(he);
          }
          else{
            id_ed[hn].second += phg.edgeWeight(he);
            if(num_neighbors[hn][phg.partID(h)]== 0){
              nnp[hn]++;
            }
          }
          num_neighbors[hn][phg.partID(h)] += phg.edgeWeight(he);
        }
      }
    }

    //std::cout << num_neighbors[14942][0] << " " << id_ed[14942].first << " thiss\n\n";
    
    std::vector<HypernodeID> idx_in_bnd(phg.initialNumNodes(), 0);
    int niter = 15;
    for(int run = 0; run < niter; run++){
      std::vector<HypernodeID> bnd;
      for(HypernodeID hn : phg.nodes()){
        if(true/*balance*/){
          if(id_ed[hn].second > 0){
            bnd.push_back(hn);
          }
        }
        else{
          if(id_ed[hn].second >= id_ed[hn].first){
            bnd.push_back(hn);
          }
        }
      }
      pq.setSize(phg.initialNumNodes());
      for(HypernodeID hn : bnd){
        double gain = (nnp[hn] > 0 ? (id_ed[hn].second / sqrt(nnp[hn])) : 0.0) - id_ed[hn].first;
        pq.changeKey({hn, -gain});
      }
      Gain quality_change = 0;
      int i = 0;
      while(!pq.isEmpty()){
        i++;
        std::cout << "move:  " << pq.getMax().second << "\n";
        HypernodeID hn = pq.deleteMax().first;
        /*if (!balance) {          
          if (id_ed[hn].first > 0 && 
              
              !ivecaxpygez(dimension, -1, vwgt+i*ncon, pwgts+from*ncon, minpwgts+from*ncon))
            continue;   
        }
        else { /* OMODE_BALANCE */
          /*if (!ivecaxpygez(ncon, -1, vwgt+i*ncon, pwgts+from*ncon, minpwgts+from*ncon)) 
            continue;   
        }*/
        PartitionID from = phg.partID(hn);
        PartitionID max_to = -1;
        Gain max_gain = 0;
        Gain highest_gain = 0; 
        if (!balance) {
                 
          for(PartitionID k = phg.k() - 1; k >=0; k--){
            Gain gain = num_neighbors[hn][k] - num_neighbors[hn][phg.partID(hn)];
            highest_gain = std::max(max_gain, gain);
            if(max_to != -1 && (num_neighbors[hn][k] > num_neighbors[hn][max_to] && is_smaller(dimension, 1, phg.nodeWeight(hn), phg.partWeight(k), max_part_weights[max_to])
              || num_neighbors[hn][k] == num_neighbors[hn][max_to] && BetterBalanceKWay(phg.nodeWeight(hn), 1, max_to, 1, k))){
              max_to = k;
              ASSERT(max_gain <= gain);
              max_gain = gain;
            }
            
            if (max_to == -1 && gain >= 0 && is_smaller(dimension, 1, phg.nodeWeight(hn), phg.partWeight(k), max_part_weights[k])){
              max_to = k;
              max_gain = gain;
            }
        
          }
          ASSERT(highest_gain == max_gain);
          if (max_to == -1)
            continue;  
          if (!(max_gain > 0 || (max_gain == 0 && BetterBalanceKWay(phg.nodeWeight(hn), -1, phg.partID(hn), +1, max_to)))){
            continue;
          }
            
            
        }
        else {  /* OMODE_BALANCE */
          /*for (k=myrinfo->nnbrs-1; k>=0; k--) {
            if (!safetos[to=mynbrs[k].pid])
              continue;
            if (ivecaxpylez(ncon, 1, vwgt+i*ncon, pwgts+to*ncon, maxpwgts+to*ncon) || 
                BetterBalanceKWay(ncon, vwgt+i*ncon, ubfactors,
                    -1, pwgts+from*ncon, pijbm+from*ncon,
                    +1, pwgts+to*ncon, pijbm+to*ncon))
              break;
          }
          if (k < 0)
            continue;  /* break out if you did not find a candidate */

          /*cto = to;
          for (j=k-1; j>=0; j--) {
            if (!safetos[to=mynbrs[j].pid])
              continue;
            if (BetterBalanceKWay(ncon, vwgt+i*ncon, ubfactors, 
                    1, pwgts+cto*ncon, pijbm+cto*ncon,
                    1, pwgts+to*ncon, pijbm+to*ncon)) {
              k   = j;
              cto = to;
            }
          }
          to = cto;

          if (mynbrs[k].ed-myrinfo->id < 0 &&
              !BetterBalanceKWay(ncon, vwgt+i*ncon, ubfactors,
                    -1, pwgts+from*ncon, pijbm+from*ncon,
                    +1, pwgts+to*ncon, pijbm+to*ncon))
            continue;*/
        }



        /*=====================================================================
        * If we got here, we can now move the vertex from 'from' to 'to' 
        *======================================================================*/
        Move move = {phg.partID(hn), max_to, hn, max_gain};
        std::cout << hn << " " << move.from << " " << move.to << " " << move.gain << "\n";
        moves_linear->push_back(move);
        ASSERT(max_to != -1);
        phg.changeNodePart(_gain_cache, hn, phg.partID(hn), max_to);
        quality_change += num_neighbors[hn][from] - num_neighbors[hn][max_to];
        //std::cout << "pc: " << quality_change << " " << hn << " ";


        for(HyperedgeID he : phg.incidentEdges(hn)){
          for(HypernodeID h : phg.pins(he)){
            if(h != hn){
              if(num_neighbors[h][max_to] == 0 && phg.partID(h) != max_to){
                nnp[h]++;
              }
              num_neighbors[h][from] -= phg.edgeWeight(he);
              num_neighbors[h][max_to] += phg.edgeWeight(he);
              if(num_neighbors[h][from] == 0 && phg.partID(h) != from){
                nnp[h]--;
              }
              if(phg.partID(h) == from){
                id_ed[h].first -= phg.edgeWeight(he);
                id_ed[h].second += phg.edgeWeight(he);
              }
              if(phg.partID(h) == max_to){
                id_ed[h].second -= phg.edgeWeight(he);
                id_ed[h].first += phg.edgeWeight(he);
              }
              if(balance){
                if(id_ed[hn].second > 0){
                  pq.changeKey({h, -((nnp[h] > 0 ? (id_ed[h].second / sqrt(nnp[h])) : 0.0) - id_ed[h].first)});
                }
                else{
                  pq.invalidate(h);
                }
              }
              else{
                if(id_ed[hn].second > 0 && id_ed[hn].second >= id_ed[hn].first){
                  pq.changeKey({h, -((nnp[h] > 0 ? (id_ed[h].second / sqrt(nnp[h])) : 0.0) - id_ed[h].first)});
                }
                else{
                  pq.invalidate(h);
                }
              }
            }
          }
        }
          std::cout << 7494 << " " << pq.items[7494] << pq.in_use[7494] << "\n";
          std::cout << 12875 << " " << pq.items[12875] << pq.in_use[12875] << "\n";
          std::cout << 10945 << " " << pq.items[10945] << pq.in_use[10945] << "\n";


        /*IFSET(ctrl->dbglvl, METIS_DBG_MOVEINFO, 
            printf("\t\tMoving %6"PRIDX" to %3"PRIDX". Gain: %4"PRIDX". Cut: %6"PRIDX"\n", 
                i, to, mynbrs[k].ed-myrinfo->id, graph->mincut));

        /* Update ID/ED and BND related information for the moved vertex */
        /*iaxpy(ncon,  1, vwgt+i*ncon, 1, pwgts+to*ncon,   1);
        iaxpy(ncon, -1, vwgt+i*ncon, 1, pwgts+from*ncon, 1);
        UpdateMovedVertexInfoAndBND(i, from, k, to, myrinfo, mynbrs, where, 
            nbnd, bndptr, bndind, bndtype);
        
        /* Update the degrees of adjacent vertices */
        /*for (j=xadj[i]; j<xadj[i+1]; j++) {
          ii = adjncy[j];
          me = where[ii];
          myrinfo = graph->ckrinfo+ii;

          oldnnbrs = myrinfo->nnbrs;

          UpdateAdjacentVertexInfoAndBND(ctrl, ii, xadj[ii+1]-xadj[ii], me, 
              from, to, myrinfo, adjwgt[j], nbnd, bndptr, bndind, bndtype);

          UpdateQueueInfo(queue, vstatus, ii, me, from, to, myrinfo, oldnnbrs, 
              nupd, updptr, updind, bndtype);

          ASSERT(myrinfo->nnbrs <= xadj[ii+1]-xadj[ii]);
        }*/
      }
      if(quality_change==0 && !balance) break;
    }
    

    /*graph->nbnd = nbnd;

    /* Reset the vstatus and associated data structures */
    /*for (i=0; i<nupd; i++) {
      ASSERT(updptr[updind[i]] != -1);
      ASSERT(vstatus[updind[i]] != VPQSTATUS_NOTPRESENT);
      vstatus[updind[i]] = VPQSTATUS_NOTPRESENT;
      updptr[updind[i]]  = -1;
    }

    if (ctrl->dbglvl&METIS_DBG_REFINE) {
       printf("\t[%6"PRIDX" %6"PRIDX"], Bal: %5.3"PRREAL", Nb: %6"PRIDX"."
              " Nmoves: %5"PRIDX", Cut: %6"PRIDX", Vol: %6"PRIDX,
              imin(nparts*ncon, pwgts,1), imax(nparts*ncon, pwgts,1), 
              ComputeLoadImbalance(graph, nparts, pijbm), 
              graph->nbnd, nmoved, graph->mincut, ComputeVolume(graph, where));
       if (ctrl->minconn) 
         printf(", Doms: [%3"PRIDX" %4"PRIDX"]", imax(nparts, nads,1), isum(nparts, nads,1));
       printf("\n");
    }

    if (nmoved == 0 || (omode == OMODE_REFINE && graph->mincut == oldcut))
      break;
  }

  rpqDestroy(queue);
    }  */ 
    return true;     
  }


template <typename GraphAndGainTypes>
  double MDRebalancer<GraphAndGainTypes>::L1_balance_gain(PartitionedHypergraph* phg,
                          const HypernodeID node,
                          const PartitionID to){
    PartitionID from = phg->partID(node);
    double gain = 0.0;
    for(int i = 0; i < dimension; i++){
      int32_t to_excess = std::max(0, std::min(phg->nodeWeight(node).weights[i], phg->partWeight(to).weights[i] + 
      phg->nodeWeight(node).weights[i] - _context.partition.max_part_weights[to].weights[i]));
      int32_t from_excess = std::max(0, std::min(phg->nodeWeight(node).weights[i], phg->partWeight(from).weights[i] - _context.partition.max_part_weights[from].weights[i]));
      gain += static_cast<double>(to_excess) * _context.partition.max_part_weights_inv[to][i]; 
      gain -= static_cast<double>(from_excess) * _context.partition.max_part_weights_inv[from][i]; 
    }        
    return gain;
  };






  /*template <typename GraphAndGainTypes, typename PriorityType>
  void MDRebalancer<GraphAndGainTypes>::simple_lp(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                       vec<vec<Move>>* moves_by_part,
                                                       vec<Move>* moves_linear,
                                                       Metrics& best_metrics, 
                                                       std::vector<HypernodeID> nodes, 
                                                       double allowed_imbalance, 
                                                       bool balance, 
                                                       Gain& local_attributed_gain,
                                                       PriorityComputer<GraphAndGainTypes,PriorityType>* prio_computer){
    
    PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
    std::vector<HypernodeWeight> max_part_weights(phg.k());
    for(PartitionID p = 0; p < phg.k(); p++){
      for(int d = 0; d < dimension; d++){
        max_part_weights[p].weights[d] = phg.partWeight(p).weights[d]; 
          if(max_part_weights[p].weights[d] < (std::ceil(static_cast<double>(_context.partition.max_part_weights[p].weights[d]) * (1.0 + allowed_imbalance)))){
            max_part_weights[p].weights[d] = std::ceil(static_cast<double>(_context.partition.max_part_weights[p].weights[d]) * (1.0 + allowed_imbalance));
          }
      }
    }
    std::cout << "startlppppp\n";
    
    auto metis_balance = [&]

    auto betterToMove = [&](HypernodeID h, PartitionID p){
      ASSERT(p != phg.partID(h));
      double nrm1=0.0, nrm2=0.0, max1=0.0, max2=0.0;
      HypernodeID from = phg.partID(h);
      for (int i = 0; i < dimension; i++) {
        double tmp = (phg.partWeight(from).weights[i] - _context.partition.max_part_weights[from].weights[i]) / _context.partition.max_part_weights[from].weights[i];
        nrm1 += tmp*tmp;
        max1 = (tmp > max1 ? tmp : max1);
        tmp = (phg.partWeight(p).weights[i] - _context.partition.max_part_weights[p].weights[i]) / _context.partition.max_part_weights[p].weights[i];
        nrm1 += tmp*tmp;
        max1 = (tmp > max1 ? tmp : max1);
        tmp = (phg.partWeight(from).weights[i] - phg.nodeWeight(h).weights[i] - _context.partition.max_part_weights[from].weights[i]) / _context.partition.max_part_weights[from].weights[i];
        nrm2 += tmp*tmp;
        max2 = (tmp > max2 ? tmp : max2);
        tmp = (phg.partWeight(p).weights[i] + phg.nodeWeight(h).weights[i] - _context.partition.max_part_weights[p].weights[i]) / _context.partition.max_part_weights[p].weights[i];
        nrm2 += tmp*tmp;
        max2 = (tmp > max2 ? tmp : max2);

      }
      if (max2 < max1)
        return 1;

      if (max2 == max1 && nrm2 < nrm1)
        return 1;

      return 0;
    };

    auto betterBalance = [&](HypernodeID hn, PartitionID p1, PartitionID p2){
      return _context.partition.refine_metis_tiebreak ? 
        betterBalanceKWay(hypergraph, phg.nodeWeight(hn), 1, p1, 1, p2) : 
        (L1_balance_gain(hypergraph, hn, p1) < L1_balance_gain(hypergraph, hn, p2));
    };
    
    NodePQ<PriorityType>* pq = prio_computer->get_pq();
    int num_rounds = 10;
    
    for(int i = 0; i < num_rounds; i++){
      if(i != 0){
        prio_computer->reinitialize();
      }
      if(pq->isEmpty()) break;
      HypernodeID num_moves = 0;
      while(!pq->isEmpty()){
        std::pair<HypernodeID,Gain> max_node = pq->deleteMax();
        HypernodeID hn = max_node.first;
        std::pair<PartitionID,Gain> max_move = {-1, 0};
        for(PartitionID p = 0; p < phg.k(); p++){
          if(!(phg.partWeight(p) + phg.nodeWeight(hn) <= max_part_weights[p])) continue;
          if(p != phg.partID(hn) && (_gain_cache.gain(hn, phg.partID(hn), p) > max_move.second || 
            _gain_cache.gain(hn, phg.partID(hn), p) == max_move.second && 
              (max_move.first == -1 ? betterToMove(hn, p) : 
              betterBalance(hn, max_move.first, p)))){
            max_move = {p, _gain_cache.gain(hn, phg.partID(hn), p)};
          }
        }
        if(max_move.first != -1){
          Move move = {phg.partID(hn), max_move.first, hn, _gain_cache.gain(hn, phg.partID(hn), p)};
          num_moves++;
          std::vector<HyperedgeID> edges_with_gain_change;
          std::vector<HypernodeID> changed_nodes;
          phg.changeNodePart(_gain_cache, move.node, move.from, move.to, HypernodeWeight(true), []{}, [&](const SynchronizedEdgeUpdate& sync_update) {
                          local_attributed_gain += (sync_update.pin_count_in_to_part_after == 1 ? sync_update.edge_weight : 0) +
                          (sync_update.pin_count_in_from_part_after == 0 ? -sync_update.edge_weight : 0);
                          if (!PartitionedHypergraph::is_graph && GainCache::triggersDeltaGainUpdate(sync_update)) {
                            edges_with_gain_change.push_back(sync_update.he);
                          }
                        });
          if(moves_linear != NULL){
            moves_linear->push_back({move.from, max_move.first, hn, local_attributed_gain});
          }
          
          prio_computer->performMove(move);


          if constexpr (!PartitionedHypergraph::is_graph) {
            for(HyperedgeID& he : edges_with_gain_change) {
              for(HypernodeID hx : phg.pins(he)){
                if(hn != hx)
                changed_nodes.push_back(hx);
              }
            }
          }                    
        }        
      }
      if(num_moves==0) break;    
    }
    std::cout << "endlp\n";*/
    /*for(HypernodeID hn : phg.nodes()){
      std::pair<PartitionID,Gain> max_move = {-1, 0};
            for(PartitionID p = 0; p < phg.k(); p++){
              if(!moved[hn] && p != phg.partID(hn) &&  (_gain_cache.gain(hn, phg.partID(hn), p) > max_move.second)){
                max_move = {p, _gain_cache.gain(hn, phg.partID(hn), p)};
              }
            }
      if(max_move.first != -1){
        std::cout << 1/0;
      }
    }*/
  /*}*/


  template<typename GraphAndGainTypes>
  void MDRebalancer<GraphAndGainTypes>::interleaveMoveSequenceWithRebalancingMoves(
                                                            mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                            const vec<HypernodeWeight>& initialPartWeights,
                                                            const std::vector<HypernodeWeight>& max_part_weights,
                                                            vec<Move>& refinement_moves,
                                                            vec<vec<Move>>& rebalancing_moves_by_part,
                                                            vec<Move>& move_order) {
    PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
    ASSERT(rebalancing_moves_by_part.size() == static_cast<size_t>(_context.partition.k));
    /*for(PartitionID p = 0; p < phg.k(); p++){
      for(Move move : rebalancing_moves_by_part[p]){
        ASSERT(move.from != -1);
        ASSERT(move.to != -1);
      }
    }*/
    /*HEAVY_REFINEMENT_ASSERT([&] {
      std::set<HypernodeID> moved_nodes;
      for (PartitionID part = 0; part < context.partition.k; ++part) {
        for (const Move& m: rebalancing_moves_by_part[part]) {
          if (m.from != part || m.to != phg.partID(m.node) || moved_nodes.count(m.node) != 0) {
            return false;
          }
          moved_nodes.changeKey(m.node);
        }
      }
      return true;
    }());*/

    std::vector<int> wasNodeMovedInThisRound(phg.initialNumNodes(), -1);
    for(int idx = 0; idx < refinement_moves.size(); idx++){
      Move move = refinement_moves[idx];
      ASSERT(wasNodeMovedInThisRound[move.node] == -1);
      wasNodeMovedInThisRound[move.node] = idx;
    }
    // Check the rebalancing moves for nodes that are moved twice. Double moves violate the precondition of the global
    // rollback, which requires that each node is moved at most once. Thus we "merge" the moves of any node
    // that is moved twice (e.g., 0 -> 2 -> 1 becomes 0 -> 1)
    for (PartitionID part = 0; part < _context.partition.k; ++part) {
      vec<Move>& moves = rebalancing_moves_by_part[part];
      tbb::parallel_for(UL(0), moves.size(), [&](const size_t i) {
        Move& r_move = moves[i];
        if (r_move.isValid() && wasNodeMovedInThisRound[r_move.node] != -1 && refinement_moves[wasNodeMovedInThisRound[r_move.node]].isValid()) {
          ASSERT(r_move.to == phg.partID(r_move.node));
          Move& first_move = refinement_moves[wasNodeMovedInThisRound[r_move.node]];
          ASSERT(r_move.node == first_move.node && r_move.from == first_move.to);
          if (first_move.from == r_move.to) {
            // if rebalancing undid the move, we simply delete it
            first_move.invalidate();
            r_move.invalidate();
          } else {
            // "merge" the moves
            r_move.from = first_move.from;
            first_move.invalidate();
          }
        }
      }, tbb::static_partitioner());
      /*for(Move m : moves){
        ASSERT(!m.isValid() || m.to == phg.partID(m.node));
      }*/
    }
    /*for(Move m : refinement_moves){
        ASSERT(!m.isValid() || m.to == phg.partID(m.node));
      }*/

    // NOTE: We re-insert invalid rebalancing moves to ensure the gain cache is updated correctly by the global rollback
    // For now we use a sequential implementation, which is probably fast enough (since this is a single scan trough
    // the move sequence). We might replace it with a parallel implementation later.
    vec<HypernodeWeight> current_part_weights = initialPartWeights;
    vec<MoveID> current_rebalancing_move_index(_context.partition.k, 0);
    MoveID next_move_index = 0;

    auto insert_moves_to_balance_part = [&](const PartitionID part) {
      if (current_part_weights[part] > max_part_weights[part]) {
        insertMovesToBalanceBlock(hypergraph, part, max_part_weights, rebalancing_moves_by_part,
                                  next_move_index, current_part_weights, current_rebalancing_move_index, move_order);
      }
    };

    // it might be possible that the initial weights are already imbalanced
    for (PartitionID part = 0; part < _context.partition.k; ++part) {
      insert_moves_to_balance_part(part);
    }

    const MoveID num_moves = refinement_moves.size();
    for (MoveID move_id = 0; move_id < num_moves; ++move_id) {
      const Move& m = refinement_moves[move_id];
      if (m.isValid()) {
        const HypernodeWeight hn_weight = phg.nodeWeight(m.node);
        current_part_weights[m.from] -= hn_weight;
        current_part_weights[m.to] += hn_weight;
        ASSERT(m.from != -1);
        ASSERT(m.to != -1);
        move_order.push_back(m);
        ++next_move_index;
        // insert rebalancing moves if necessary
        insert_moves_to_balance_part(m.to);
      } else {
        // setting moveOfNode to zero is necessary because, after replacing the move sequence,
        // wasNodeMovedInThisRound() could falsely return true otherwise
        
      }
    }

    // append any remaining rebalancing moves (rollback will decide whether to keep them)
    for (PartitionID part = 0; part < _context.partition.k; ++part) {
      while (current_rebalancing_move_index[part] < rebalancing_moves_by_part[part].size()) {
        const MoveID move_index_for_part = current_rebalancing_move_index[part];
        const Move& m = rebalancing_moves_by_part[part][move_index_for_part];
        ++current_rebalancing_move_index[part];
        if(m.from != -1){
          move_order.push_back(m);
        }
        ++next_move_index;
      }
    }
  }

  template<typename GraphAndGainTypes>
  void MDRebalancer<GraphAndGainTypes>::insertMovesToBalanceBlock(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                                                        const PartitionID block,
                                                                        const std::vector<HypernodeWeight>& max_part_weights,
                                                                        const vec<vec<Move>>& rebalancing_moves_by_part,
                                                                        MoveID& next_move_index,
                                                                        vec<HypernodeWeight>& current_part_weights,
                                                                        vec<MoveID>& current_rebalancing_move_index,
                                                                        vec<Move>& move_order) {
    PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
    while (current_part_weights[block] > max_part_weights[block]
            && current_rebalancing_move_index[block] < rebalancing_moves_by_part[block].size()) {
      const MoveID move_index_for_block = current_rebalancing_move_index[block];
      const Move& m = rebalancing_moves_by_part[block][move_index_for_block];
      ++current_rebalancing_move_index[block];
      /*ASSERT(block < _context.partition.k);
      ASSERT(m.from != -1, block);
      ASSERT(m.to != -1);*/
      
      ++next_move_index;
      if (m.isValid()) {
        move_order.push_back(m);
        const HypernodeWeight hn_weight = phg.nodeWeight(m.node);
        current_part_weights[m.from] -= hn_weight;
        current_part_weights[m.to] += hn_weight;

        if (current_part_weights[m.to] > max_part_weights[m.to]) {
          // edge case: it is possible that the rebalancing move itself causes new imbalance -> call recursively
          insertMovesToBalanceBlock(hypergraph, m.to, max_part_weights, rebalancing_moves_by_part,
                                    next_move_index, current_part_weights, current_rebalancing_move_index, move_order);
        }
      }
    }
  }
}


