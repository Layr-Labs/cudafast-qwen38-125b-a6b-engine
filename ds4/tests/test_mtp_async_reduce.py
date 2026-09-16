#!/usr/bin/env python3
"""Run the actual CUDA reducer body with host coroutine barrier scheduling.
This validates host-executed source semantics, not CUDA execution or timing.
"""
from pathlib import Path
import json
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]

def function(source, marker):
    start = source.index(marker)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]

comparator = function((root / 'ds4_cuda.cu').read_text(),
                      '__device__ __forceinline__ static bool topk_score_better(')
comparator = comparator.replace('__device__ ', '').replace('__forceinline__ ', 'inline ')
kernel = function((root / 'ds4_cuda_mtp_native.cuh').read_text(),
                  '__global__ static void mtp_native_reduce_pending(')
assert kernel.count('__syncthreads();') == 2
kernel = kernel.replace('__global__ static void', 'static task')
kernel = kernel.replace('__shared__ ', 'static ')
kernel = kernel.replace('__syncthreads();', 'co_await std::suspend_always{};')
kernel = kernel.replace('return;', 'co_return;')
code = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
struct { unsigned x; } threadIdx;
static uint32_t __float_as_uint(float x) {uint32_t u;std::memcpy(&u,&x,4);return u;}
struct task {
 struct promise_type {
  task get_return_object(){return {std::coroutine_handle<promise_type>::from_promise(*this)};}
  std::suspend_always initial_suspend(){return {};}
  std::suspend_always final_suspend() noexcept {return {};}
  void return_void(){} void unhandled_exception(){std::terminate();}
 };
 std::coroutine_handle<promise_type> h;
};
''' + comparator + '\n' + kernel + r'''
int main() {
 constexpr uint32_t N=2048,T=1024;
 std::mt19937 random(381937);
 uint32_t cases=0;
 for(unsigned trial=0;trial<512;++trial) {
  uint32_t vocab=(trial%3==0)?UINT32_MAX:(trial%3==1?UINT32_MAX-1:4*N);
  std::vector<uint32_t> ids(N),order(N);std::vector<float> values(N);
  std::unordered_set<uint32_t> used;used.insert(0);ids[0]=0;
  for(unsigned i=1;i<N;++i){uint32_t id;do{id=random()%vocab;}while(!used.insert(id).second);ids[i]=id;}
  std::shuffle(ids.begin()+1,ids.end(),random);
  for(unsigned i=0;i<N;++i){uint32_t bits=random();std::memcpy(&values[i],&bits,4);}
  switch(trial%12) {
   case 0:std::fill(values.begin(),values.end(),-INFINITY);break;
   case 1:std::fill(values.begin(),values.end(),NAN);break;
   case 2:std::fill(values.begin(),values.end(),0.f);break;
   case 3:std::fill(values.begin(),values.end(),-0.f);values[N-1]=0.f;break;
   case 4:std::fill(values.begin(),values.end(),INFINITY);break;
   case 5:values[0]=NAN;values[N-1]=INFINITY;break;
   case 6:values[0]=-INFINITY;values[1]=NAN;break;
   case 7:values[0]=-0.f;values[1]=0.f;break;
   case 8:std::fill(values.begin(),values.end(),7.f);break;
  }
  uint32_t invalid=trial%8==7?((trial/8)%2?1u:UINT32_MAX):0u;
  if(invalid) std::rotate(ids.begin(),ids.begin()+1,ids.end());
  // A displaced zero with a set flag is a defensive raw-kernel case. The
  // real key pipeline always places mandatory zero first, even for NaNs.
  std::iota(order.begin(),order.end(),0u);
  std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return ids[a]<ids[b];});
  uint32_t expected=UINT32_MAX;
  if(!invalid) {
   expected=ids[order[0]];float best=values[order[0]];
   for(uint32_t i:order)if(values[i]>best){best=values[i];expected=ids[i];}
  }
  uint32_t actual=123;
  std::vector<task> threads;threads.reserve(T);
  for(unsigned t=0;t<T;++t)threads.push_back(mtp_native_reduce_pending(&actual,values.data(),ids.data(),N,vocab,&invalid));
  unsigned rounds=0;
  for(;;){bool active=false;
   for(unsigned t=0;t<T;++t)if(!threads[t].h.done()){threadIdx.x=t;threads[t].h.resume();active=true;}
   if(!active)break;assert(++rounds<=12);
  }
  for(auto t:threads)t.h.destroy();
  assert(actual==expected);
  if(invalid)assert(rounds==1); // Flag-first branch never reaches a barrier.
  ++cases;
 }
 std::cout<<"PASS "<<cases<<" actual-source reducer cases of 2048 IDs\n";
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-async-reduce-') as directory:
    source = Path(directory) / 'check.cpp'
    binary = Path(directory) / 'check'
    source.write_text(code)
    subprocess.run(['c++', '-O3', '-ffast-math', '-fno-finite-math-only',
                    '-std=c++20', str(source), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
print(json.dumps({'cases': 512, 'candidate_count': 2048,
                  'actual_kernel_body': True, 'gpu_executed': False,
                  'output': result.stdout.strip()}, indent=2))
