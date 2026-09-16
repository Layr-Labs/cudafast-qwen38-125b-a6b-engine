"""Compare the extracted fused fold/gate with the actual two-kernel sequence.

CUDA shuffle max and approximate reciprocal are replaced at their contracts.
This checks ordering/indexing/FTZ and output guards; it is not native execution.
"""
from pathlib import Path
import argparse,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--rescale',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
s=Path(a.source).read_text() if a.source else (root/'ds4_cuda_qwen4exp.cu').read_text()
def extract(name):
    st=s.index('__global__ static void '+name+'(');b=s.index('{',st);e=b+1;d=1
    while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
    return s[st:e].replace('__global__ ','')
code=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
struct Dim {unsigned x=0,y=0,z=0;};static Dim threadIdx,blockIdx,blockDim{256,1,1};
constexpr float QWEN4EXP_QSA_MASKED_SCORE=-3.0e38f,QWEN4EXP_QSA_MASKED_LIMIT=-1.0e30f;
constexpr float QWEN4EXP_Q8_RCP127=0x1.020408p-7f;
static float qwen4exp_q8_ftz(float x){return std::fpclassify(x)==FP_SUBNORMAL?std::copysign(0.0f,x):x;}
static float qwen4exp_q8_rcp_approx(float x){return 1.0f/x;}
static float __fmaf_rn(float a,float b,float c){return std::fma(a,b,c);}
static float lane_abs[32];static unsigned phase;
static float __shfl_xor_sync(uint32_t mask,float x,int off){
 assert(mask==UINT32_MAX&&(off==16||off==8||off==4||off==2||off==1));
 if(!phase){if(off==16)lane_abs[threadIdx.x&31]=x;return 0;}
 float v=0;for(float a:lane_abs)v=std::fmax(v,a);return v;
}
'''+extract('qwen4exp_qsa_split_fold_kernel')+'\n'+extract('qwen4exp_qsa_output_gate_doubled_quant_kernel')+'\n'+(extract('qwen4exp_qsa_split_fold_rescale_quant_kernel').replace('qwen4exp_qsa_split_fold_rescale_quant_kernel','qwen4exp_qsa_split_fold_quant_kernel') if a.rescale else extract('qwen4exp_qsa_split_fold_quant_kernel'))+r'''
static uint32_t rng=0x453287efu;
static float rnd(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return ((int)(rng%20001)-10000)/1000.0f;}
static uint64_t values;static unsigned cases;
static void run(unsigned rows,unsigned max_count,bool sparse,bool dpos,unsigned pattern){
 const unsigned heads=24,dim=256,max_tiles=(max_count+255)/256,n=rows*heads*dim;
 std::vector<float> maxima(rows*heads*max_tiles),sums(maxima.size()),ct(maxima.size()*dim),gate(n*2+16,19);
 std::vector<int32_t> counts(rows);unsigned pos=max_count-rows;
 for(unsigned r=0;r<rows;r++)counts[r]=pattern==1?0:pattern==2?1+(r*255)%(max_count):max_count-r;
 for(unsigned i=0;i<maxima.size();i++){
  maxima[i]=pattern==3?QWEN4EXP_QSA_MASKED_SCORE:rnd();sums[i]=pattern==3?0:std::fabs(rnd());
 }
 for(float &x:ct)x=pattern==3?0:rnd();
 for(unsigned i=0;i<n*2;i++)gate[i]=pattern==4?((i%5==0)?INFINITY:(i%5==1)?-INFINITY:(i%5==2)?1000:-1000):rnd();
 if(pattern==5)for(float &x:ct)x=std::copysign(std::numeric_limits<float>::denorm_min(),rnd());
 if(pattern==6)for(float &x:ct)x=std::copysign(0.0f,rnd());
 const auto maxima_in=maxima,sums_in=sums,ct_in=ct,gate_in=gate;const auto counts_in=counts;
 std::vector<float> old(n+16,123),fused(n+16,123),os(n/32+16,456),ns(n/32+16,456);
 std::vector<int8_t> oq(n+64,93),nq(n+64,93);const uint32_t *dp=dpos?&pos:nullptr;
 for(unsigned r=0;r<rows;r++)for(unsigned h=0;h<heads;h++){
  blockIdx={h,r,0};for(unsigned t=0;t<256;t++){threadIdx.x=t;
   qwen4exp_qsa_split_fold_kernel(maxima.data(),sums.data(),ct.data(),counts.data(),old.data()+8,
    rows,heads,256,dpos?999:pos,sparse,max_tiles,256,dp);
  }
  for(unsigned warp=0;warp<8;warp++){
   for(phase=0;phase<2;phase++)for(unsigned lane=0;lane<32;lane++){
    blockIdx={r*heads+h,0,0};threadIdx.x=warp*32+lane;
    qwen4exp_qsa_output_gate_doubled_quant_kernel(oq.data()+32,os.data()+8,gate.data(),old.data()+8,n);
   }
   for(phase=0;phase<2;phase++)for(unsigned lane=0;lane<32;lane++){
    blockIdx={h,r,0};threadIdx.x=warp*32+lane;
    qwen4exp_qsa_split_fold_quant_kernel(maxima.data(),sums.data(),ct.data(),counts.data(),fused.data()+8,
     rows,heads,256,dpos?999:pos,sparse,max_tiles,256,dp,nq.data()+32,ns.data()+8,gate.data());
   }
  }
 }
 assert(!memcmp(old.data(),fused.data(),old.size()*4));assert(oq==nq);assert(!memcmp(os.data(),ns.data(),os.size()*4));
 for(unsigned i=0;i<8;i++)assert(old[i]==123&&old[n+8+i]==123&&os[i]==456&&os[n/32+8+i]==456);
 for(unsigned i=0;i<32;i++)assert(oq[i]==93&&oq[n+32+i]==93);
 assert(maxima==maxima_in&&sums==sums_in&&ct==ct_in&&gate==gate_in&&counts==counts_in);
 values+=n;cases++;
}
int main(){
 for(unsigned rows:{1u,2u,3u})for(unsigned max_count:{3u,255u,256u,257u,1024u,2048u,4096u})
  for(bool sparse:{false,true})for(bool dpos:{false,true})for(unsigned pattern=0;pattern<7;pattern++)
   run(rows,max_count,sparse,dpos,pattern);
 printf("PASS %u extracted kernel cases, %llu values; bit-identical float/Q8/scales and guards; collective/reciprocal contracts simulated\n",cases,(unsigned long long)values);
}
'''
if a.rescale:
    code=code.replace('const auto maxima_in=maxima', """std::vector<float> scales(maxima.size()*256,123.25f);
 for(unsigned r=0;r<rows;r++)for(unsigned h=0;h<heads;h++){
  float prior=QWEN4EXP_QSA_MASKED_SCORE;
  for(unsigned t=0;t<max_tiles;t++){
   const unsigned at=(r*heads+h)*max_tiles+t;const float current=std::fmax(prior,maxima[at]);
   scales[at*256]=prior>QWEN4EXP_QSA_MASKED_LIMIT?std::exp(prior-current):0.0f;prior=current;
  }
 }
 const auto maxima_in=maxima""")
    code=code.replace('qwen4exp_qsa_split_fold_quant_kernel(maxima.data()', 'qwen4exp_qsa_split_fold_quant_kernel(scales.data()')
with tempfile.TemporaryDirectory(prefix='qsa-fold-oracle-') as d:
    src=Path(d)/'test.cpp';exe=src.with_suffix('');src.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=120)
