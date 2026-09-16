"""Run actual CUDA reduction bodies as cooperative host coroutines; no GPU."""
from pathlib import Path
import subprocess,tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda.cu').read_text()
def function(name):
    begin=s.index(name);begin=s.rfind('\n',0,begin)+1
    a=s.index('{',begin);b=a+1;depth=1
    while depth:
        depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[begin:b]
compare=function('__device__ __forceinline__ static bool topk_score_better').replace('__device__ __forceinline__','')
part=function('__global__ static void target_top1_partition_kernel').replace('__global__ static void','task').replace('__shared__','static').replace('__syncthreads();','co_await std::suspend_always{};').replace('return;','co_return;')
finish=function('__global__ static void target_top1_finish_kernel').replace('__global__ static void','task')
finish=finish.replace('const float other=__shfl_down_sync(0xffffffffu,value,offset);','fv[lane]=value; fi[lane]=id; co_await std::suspend_always{};\n        const float other=fv[lane+offset<32?lane+offset:lane];').replace('const uint32_t other_id=__shfl_down_sync(0xffffffffu,id,offset);','const uint32_t other_id=fi[lane+offset<32?lane+offset:lane];').replace('value=other;id=other_id;}','value=other;id=other_id;}\n        co_await std::suspend_always{};')
code=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
struct {uint32_t x,y;} blockIdx;
struct {uint32_t x;} threadIdx;
struct target_top1_pair {float value;uint32_t id;};
float fv[32];uint32_t fi[32];
struct task {
 struct promise_type {
  task get_return_object(){return {std::coroutine_handle<promise_type>::from_promise(*this)};}
  std::suspend_always initial_suspend(){return{};} std::suspend_always final_suspend() noexcept{return{};}
  void return_void(){} void unhandled_exception(){std::terminate();}
 };std::coroutine_handle<promise_type> h;
};
void execute(std::vector<task>&v){for(;;){bool active=false;for(unsigned t=0;t<v.size();t++)if(!v[t].h.done()){threadIdx.x=t;v[t].h.resume();active=true;}if(!active)break;}for(auto t:v)t.h.destroy();}
''' + compare+'\n'+part+'\n'+finish+r'''
int main(){std::mt19937 gen(42001);unsigned cases=0;
for(uint32_t width:{65536u,65537u,65543u,248320u})for(uint32_t rows:{1u,2u})for(unsigned mode=0;mode<10;mode++){
 std::vector<float>x((uint64_t)width*rows);std::vector<uint32_t>out(rows,UINT32_MAX),expected(rows);std::vector<target_top1_pair>p(rows*8);
 for(auto&f:x){uint32_t bits=gen();std::memcpy(&f,&bits,4);}
 if(mode==1)std::fill(x.begin(),x.end(),NAN);
 if(mode==2)std::fill(x.begin(),x.end(),-INFINITY);
 if(mode==3)std::fill(x.begin(),x.end(),INFINITY);
 if(mode==4)std::fill(x.begin(),x.end(),-0.f);
 if(mode==5){std::fill(x.begin(),x.end(),0.f);x[0]=NAN;x[width-1]=1.f;}
 if(mode==6){std::fill(x.begin(),x.end(),-1.f);for(unsigned i:{1023u,1024u,8191u,8192u})x[i]=7.f;}
 if(mode==7){std::fill(x.begin(),x.end(),-INFINITY);x[0]=NAN;x[width-1]=INFINITY;}
 if(mode==8)for(unsigned i=0;i<x.size();i++){uint32_t bits=i&1?0x80000001u:1u;std::memcpy(&x[i],&bits,4);}
 if(mode==9)for(unsigned i=0;i<x.size();i++){uint32_t bits=(i&1?0x80000000u:0)|((i%3)?0x7fffffu:0x800000u);std::memcpy(&x[i],&bits,4);}
 for(unsigned r=0;r<rows;r++){
  float best=-INFINITY;uint32_t id=0;
  for(uint32_t i=0;i<width;i++)if(topk_score_better(x[(uint64_t)r*width+i],i,best,id)){best=x[(uint64_t)r*width+i];id=i;}expected[r]=id;
  for(unsigned part=0;part<8;part++){blockIdx={part,r};std::vector<task>v;for(unsigned t=0;t<1024;t++)v.push_back(target_top1_partition_kernel(p.data(),x.data(),width,rows));execute(v);}
  blockIdx={r,0};std::vector<task>v;for(unsigned t=0;t<32;t++)v.push_back(target_top1_finish_kernel(out.data(),p.data()));execute(v);
 }
 assert(out==expected);cases++;
}std::cout<<"PASS "<<cases<<" actual two-stage reduction cases\n";}
'''
with tempfile.TemporaryDirectory(prefix='target-top1-host-') as d:
    source=Path(d)/'test.cpp';source.write_text(code)
    for mode,flags in [('ieee',[]),('ftz',['-ffast-math','-fno-finite-math-only'])]:
        binary=Path(d)/mode
        subprocess.run(['c++','-O2','-std=c++20',*flags,str(source),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)
