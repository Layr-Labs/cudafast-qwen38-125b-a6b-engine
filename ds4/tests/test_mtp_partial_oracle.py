"""Execute production surrounding loads/stores with scalar CUB contracts.

Checks exact top-K set, tie order, padding, arbitrary compaction block order,
overflow/underflow fallback and guards. This does not execute native CUB.
"""
from pathlib import Path
import re
import argparse
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,default=root/'ds4_cuda_mtp_partial.cuh')
source=parser.parse_args().source.read_text()
def body(name):
    begin = source.index('{', source.index(name+'('))
    end, depth = begin+1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin+1:end-1]

pivot = body('mtp_partial_pivot')
p_load, p_save = pivot.split('Sort(storage).SortDescendingBlockedToStriped(items,32,64);')
p_load = re.sub(r'    using Sort = .*?;\n', '', p_load)
p_load = re.sub(r'    __shared__ typename Sort::TempStorage storage;\n', '', p_load)
p_load = p_load.replace('    uint64_t items[4];\n', '')
sort = body('mtp_partial_sort_ids')
s_load, remaining = sort.split('Keys(storage.keys).SortDescendingBlockedToStriped(items);')
s_mid, s_save = remaining.split('Ids(storage.ids).SortBlockedToStriped(original,0,id_bits);')
s_load = re.sub(r'    using (Keys|Ids) = .*?;\n', '', s_load)
s_load = re.sub(r'    __shared__ union Storage \{.*?\n    \} storage;\n', '', s_load, flags=re.S)
s_load = s_load.replace('    uint64_t items[16];\n', '')
assert s_mid.count('__syncthreads();') == 1
s_mid = s_mid.replace('    uint32_t original[8];\n', '').replace('__syncthreads();', '')
filter_body = body('mtp_partial_filter')
filter_body = re.sub(r'    using Scan = .*?;\n', '', filter_body)
filter_body = filter_body.replace('    __shared__ typename Scan::TempStorage storage;\n', '')
filter_body = filter_body.replace('    __shared__ uint32_t base;\n', '')
assert filter_body.count('__syncthreads();') == 1
filter_body = filter_body.replace('__syncthreads();', '')
filter_body = filter_body.replace('Scan(storage).ExclusiveSum(take,offset,total);',
    'scan_contract(keys,*pivot,width,blockIdx.x,threadIdx.x,take,offset,total);')

code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
constexpr uint32_t MTP_NATIVE_CAP=2048,MTP_PARTIAL_SAMPLES=1024,MTP_PARTIAL_LIMIT=4096,MTP_PARTIAL_TARGET=3072;
struct Dim{uint32_t x;};
static uint32_t atomicAdd(uint32_t *p,uint32_t n){auto old=*p;*p+=n;return old;}
static std::array<uint32_t,256> scan_prefix;static uint32_t scan_total,scan_block;
static void scan_contract(const uint64_t *keys,uint64_t pivot,uint32_t width,
        uint32_t block,uint32_t thread,uint32_t take,uint32_t &offset,uint32_t &total){
    assert(block==scan_block);
    const auto at=block*256+thread;
    assert(take==uint32_t(at<width&&keys[at]>=pivot));offset=scan_prefix[thread];total=scan_total;
}
'''
code += 'static void sample_load(const uint64_t *keys,uint32_t width,uint64_t *items,Dim threadIdx){'+p_load+'}\n'
code += 'static void sample_save(uint64_t *pivot,uint32_t width,uint64_t *items,Dim threadIdx){'+p_save+'}\n'
code += 'static void final_load(const uint64_t *selected,uint32_t count,uint64_t *items,Dim threadIdx){'+s_load+'}\n'
code += 'static void final_decode(uint64_t *items,uint32_t *original,Dim threadIdx){'+s_mid+'}\n'
code += 'static void final_save(uint32_t *ids,uint32_t *original,Dim threadIdx){'+s_save+'}\n'
code += 'static void filter_thread(uint64_t *selected,uint32_t *count,const uint64_t *keys,const uint64_t *pivot,uint32_t width,Dim blockIdx,Dim threadIdx,uint32_t &base){'+filter_body+'}\n'
code += r'''
int main(){
 std::mt19937 rng(0x19aed734);unsigned cases=0,partial=0,underflow=0,overflow=0;uint64_t checked=0;
 for(uint32_t width:{4096u,4097u,5000u,20276u,65535u,98580u,131072u,196608u,1048576u})
 for(unsigned pattern=0;pattern<9;pattern++)for(unsigned replay=0;replay<3;replay++){
  std::vector<uint64_t> guarded(width+2,0xabcdeffeef123456ull);auto *keys=guarded.data()+1;
  const uint32_t vocab=width+65537u;const unsigned bits=32-__builtin_clz(vocab-1);
  std::vector<bool> sampled(width,false);for(uint64_t i=0;i<1024;i++)sampled[i*width/1024]=true;
  for(unsigned i=0;i<width;i++){
   uint32_t high=0x00800000u+(rng()%0xff000000u);
   if(pattern==1)high=0x80000000u;
   if(pattern==2)high=0x80000000u+i;
   if(pattern==3)high=0x80000000u+width-i;
   if(pattern==4)high=sampled[i]?0x80000000u:0xc0000000u;
   if(pattern==5)high=sampled[i]?0xc0000000u:0x80000000u;
   if(pattern==6)high=i<2048?0xc1000000u:0x80000000u;
   if(pattern==7)high=(i&1)?0x00800000u:0xff7fffffu;
   if(pattern==8)high=0xc1000000u+(rng()%4);
   uint32_t id=i<width-276?i:vocab-276+i-(width-276);
   if(!id||i>=width-276)high=0xffffffffu;
   keys[i]=((uint64_t)high<<32)|(UINT32_MAX-id);
  }
  const auto saved=guarded;
  std::vector<uint64_t> sample;sample.reserve(1024);std::array<std::array<uint64_t,4>,256> sl{};
  for(unsigned t=0;t<256;t++){sample_load(keys,width,sl[t].data(),{t});sample.insert(sample.end(),sl[t].begin(),sl[t].end());}
  std::stable_sort(sample.begin(),sample.end(),[](auto a,auto b){return (a>>32)>(b>>32);});
  uint64_t pivot=0;
  for(unsigned t=0;t<256;t++){
   for(unsigned i=0;i<4;i++)sl[t][i]=sample[t+i*256];sample_save(&pivot,width,sl[t].data(),{t});
  }
  assert(pivot);
  std::array<uint64_t,4098> filtered;filtered.fill(0x5ac7e53a773655aaull);
  uint32_t count=0;std::vector<unsigned> blocks((width+255)/256);std::iota(blocks.begin(),blocks.end(),0);
  std::shuffle(blocks.begin(),blocks.end(),rng);
  for(auto block:blocks){
   scan_block=block;scan_total=0;
   for(unsigned t=0;t<256;t++){scan_prefix[t]=scan_total;unsigned at=block*256+t;scan_total+=at<width&&keys[at]>=pivot;}
   uint32_t shared_base=0;
   for(unsigned t=0;t<256;t++)filter_thread(filtered.data()+1,&count,keys,&pivot,width,{block},{t},shared_base);
  }
  unsigned truth_count=0;for(unsigned i=0;i<width;i++)truth_count+=keys[i]>=pivot;assert(count==truth_count);
  assert(filtered.front()==0x5ac7e53a773655aaull&&filtered.back()==filtered.front());
  assert(saved==guarded);
  std::vector<uint64_t> expected(keys,keys+width);std::sort(expected.begin(),expected.end(),std::greater<uint64_t>());expected.resize(2048);
  std::vector<uint32_t> ids;for(auto k:expected)ids.push_back(UINT32_MAX-(uint32_t)k);std::sort(ids.begin(),ids.end());
  if(count>=2048&&count<=4096){
   partial++;
   std::vector<uint64_t> items;std::array<std::array<uint64_t,16>,256> kl{};
   for(unsigned t=0;t<256;t++){final_load(filtered.data()+1,count,kl[t].data(),{t});items.insert(items.end(),kl[t].begin(),kl[t].end());}
   std::sort(items.begin(),items.end(),std::greater<uint64_t>());
   std::vector<uint32_t> decoded;std::array<std::array<uint32_t,8>,256> il{};
   for(unsigned t=0;t<256;t++){
    for(unsigned i=0;i<16;i++)kl[t][i]=items[t+i*256];
    final_decode(kl[t].data(),il[t].data(),{t});decoded.insert(decoded.end(),il[t].begin(),il[t].end());
   }
   const uint64_t mask=(uint64_t(1)<<bits)-1;
   std::stable_sort(decoded.begin(),decoded.end(),[mask](auto a,auto b){return (a&mask)<(b&mask);});
   std::array<uint32_t,2050> output;output.fill(0x77bb15ceu);
   for(unsigned t=0;t<256;t++){
    for(unsigned i=0;i<8;i++)il[t][i]=decoded[t+i*256];final_save(output.data()+1,il[t].data(),{t});
   }
   assert(output.front()==0x77bb15ceu&&output.back()==output.front());
   assert(std::equal(ids.begin(),ids.end(),output.begin()+1));
  }else if(count<2048)underflow++;else overflow++;
  checked+=width;cases++;
 }
 assert(partial>0&&underflow>0&&overflow>0);
 std::cout<<"PASS "<<cases<<" cases, "<<checked<<" keys, partial="<<partial<<" underflow="<<underflow<<" overflow="<<overflow<<"; exact top2048 and guards. CUB collectives are scalar contracts.\n";
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-partial-') as d:
    cpp=Path(d)/'test.cpp';exe=cpp.with_suffix('');cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=90)
