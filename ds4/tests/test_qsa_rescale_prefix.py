"""Run the actual changed probability prefix in arbitrary thread order.

Verify the prior prefix maximum, saved fold rescale, and every original score
load. The unchanged probability/reduction/contribution suffix is also checked.
"""
from pathlib import Path
import argparse,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--source');a=p.parse_args()
root=Path(__file__).resolve().parents[1];s=Path(a.source).read_text() if a.source else (root/'ds4_cuda_qwen4exp.cu').read_text()
def fn(name):
 at=s.index(name+'(');st=s.rindex('template <uint32_t GROUP',0,at);b=s.index('{',at);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[st:e]
old=fn('qwen4exp_qsa_split_probs_kernel');new=fn('qwen4exp_qsa_split_probs_rescale_kernel')
marker='    for (uint32_t h = 0; h < GROUP; h++) {\n        p[h] = (key >= 0) ? expf(p[h] - m[h]) : 0.0f;'
assert old[old.index(marker):]==new[new.index(marker):], 'probability/contribution suffix changed'
def prefix(text,name):
 text=text[:text.rindex('#pragma unroll',0,text.index(marker))]
 text=text.replace('__global__ ','').replace('__launch_bounds__(256, 1)','').replace('extern __shared__ __align__(16) float qwen4exp_attn_pr_shared[];','')
 text=text.replace(name+'(',name+'_prefix(').replace('const uint32_t *d_pos) {','const uint32_t *d_pos, float *saved_p, float *saved_m) {')
 return text+'''\n    for (uint32_t h=0;h<GROUP;h++) {
        saved_p[(row+h*max_tiles)*nth+tid]=p[h];
        saved_m[(row+h*max_tiles)*nth+tid]=m[h];
    }
}\n'''
code=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
using std::min;
struct Dim{unsigned x=0,y=0,z=0;};static Dim blockIdx,threadIdx,blockDim{256,1,1};
static float qwen4exp_attn_pr_shared[4*256*2+256];
constexpr float QWEN4EXP_QSA_MASKED_SCORE=-3e38f,QWEN4EXP_QSA_MASKED_LIMIT=-1e30f;
static int32_t qwen4exp_qsa_tile_key(const int32_t *sel,uint32_t token,uint32_t maxsel,uint32_t base,uint32_t tid,uint32_t n,uint32_t cap,uint32_t sparse){
 if(tid>=n)return -1;int32_t k=sparse?sel[token*maxsel+base+tid]:(int32_t)(base+tid);return k>=0&&(uint32_t)k<cap?k:-1;
}
'''+prefix(old,'qwen4exp_qsa_split_probs_kernel')+prefix(new,'qwen4exp_qsa_split_probs_rescale_kernel')+r'''
static unsigned cases;static uint64_t words;
template<unsigned GROUP,unsigned STEP>static void run(unsigned rows,unsigned cap,bool sparse,bool use_dpos,unsigned pattern){
 const unsigned heads=24,dim=256,nt=(cap+255)/256,total=rows*heads*nt*dim;
 std::mt19937 rng(1929+cap+rows+GROUP+pattern);std::vector<float> maxes(rows*heads*nt),scores(total),a(total,-731),b(total,-731),ma(total,-753),mb(total,-753);
 std::vector<int32_t> counts(rows),selected(rows*cap);unsigned pos=cap-rows;
 for(auto &v:maxes)v=pattern==1?QWEN4EXP_QSA_MASKED_SCORE:pattern==2?0.0f:((int)(rng()%2001)-1000)*0.125f;
 for(auto &v:scores)v=((int)(rng()%20001)-10000)*0.03125f;
 for(unsigned r=0;r<rows;r++){
  counts[r]=pattern==3?0:pattern==4?1:cap-r;
  for(unsigned i=0;i<cap;i++)selected[r*cap+i]=i%17==0?-1:(int32_t)(cap-i-1);
 }
 auto saved=scores;auto maxes_in=maxes;const uint32_t *dp=use_dpos?&pos:nullptr;
 unsigned order[256];std::iota(order,order+256,0);std::shuffle(order,order+256,rng);
 std::vector<unsigned> blocks(rows*(heads/GROUP)*nt);std::iota(blocks.begin(),blocks.end(),0);std::shuffle(blocks.begin(),blocks.end(),rng);
 for(unsigned index:blocks){
  const unsigned tile=index%nt,group=(index/nt)%(heads/GROUP),row=index/(nt*(heads/GROUP));blockIdx={group,tile,row};
  for(unsigned tid:order){threadIdx.x=tid;
   qwen4exp_qsa_split_probs_kernel_prefix<GROUP,STEP>(nullptr,selected.data(),counts.data(),saved.data(),maxes.data(),nullptr,nullptr,
     rows,heads,2,dim,use_dpos?999:pos,cap,cap,sparse,nt,dp,a.data(),ma.data());
   qwen4exp_qsa_split_probs_rescale_kernel_prefix<GROUP,STEP>(nullptr,selected.data(),counts.data(),scores.data(),maxes.data(),nullptr,nullptr,
     rows,heads,2,dim,use_dpos?999:pos,cap,cap,sparse,nt,dp,b.data(),mb.data());
  }
 }
 assert(!memcmp(a.data(),b.data(),total*4));assert(!memcmp(ma.data(),mb.data(),total*4));assert(maxes==maxes_in);
 for(unsigned r=0;r<rows;r++)for(unsigned h=0;h<heads;h++){
  float prior=QWEN4EXP_QSA_MASKED_SCORE;const unsigned count=sparse?(unsigned)counts[r]:pos+r+1;
  for(unsigned t=0;t<nt;t++){
   const unsigned at=(r*heads+h)*nt+t;const float current=std::fmax(prior,maxes[at]);
   const float expected=prior>QWEN4EXP_QSA_MASKED_LIMIT?std::exp(prior-current):0;
   for(unsigned d=0;d<256;d++){
    const float want=t*256<count&&d==0?expected:saved[at*256+d];assert(!memcmp(&want,&scores[at*256+d],4));
   }
   prior=current;
  }
 }
 words+=total;cases++;
}
int main(){
 for(unsigned r:{1u,2u,3u})for(unsigned c:{3u,255u,256u,257u,1024u,2048u,4096u})for(bool sp:{false,true})for(bool dp:{false,true})for(unsigned p=0;p<5;p++){
  run<2,8>(r,c,sp,dp,p);run<2,16>(r,c,sp,dp,p);run<4,16>(r,c,sp,dp,p);
 }
 printf("PASS %u actual producer-prefix cases, %llu score words, shuffled block/thread order, exact saved rescale and untouched probability inputs/suffix\n",cases,(unsigned long long)words);
}
'''
with tempfile.TemporaryDirectory(prefix='qsa-rescale-prefix-') as d:
 src=Path(d)/'prefix.cpp';exe=src.with_suffix('');src.write_text(code)
 subprocess.run(['c++','-O2','-std=c++17','-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(src),'-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True,timeout=120)
