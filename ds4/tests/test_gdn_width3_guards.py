"""Compile actual GDN fusion/staging predicates; not a GPU API execution test."""
from pathlib import Path
import subprocess,tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda.cu').read_text();g=(repo/'ds4/ds4_qwen4exp_graph.inc').read_text()
def condition(text,start):
 a=text.index(start)+4;p=a;depth=1
 while depth: depth+=(text[p]=='(')-(text[p]==')');p+=1
 return text[a:p-1]
api=condition(s,'if ((rows<=2u || (rows==3u &&')
graph=condition(g,'if ((n_tokens<=2u || (n_tokens==3u &&')
a=s.index('const int gdn_stage =');b=s.index(';',a);stage=s[a+len('const int gdn_stage ='):b]
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>
static bool dp;static int cuda_q8_use_dp4a(){return dp;}
int main(){unsigned cases=0;
 const char *envs[]={"DS4_QWEN4EXP_NO_WIDE_VERIFY","DS4_QWEN4EXP_WIDE_VERIFY_R2","DS4_QWEN4EXP_NO_ROW_TILE","DS4_QWEN4EXP_PAIR_LANES_R2","DS4_F32_NO_VECTOR_DECODE","DS4_QWEN4EXP_NO_GDN_PROJECTION_FUSION","DS4_QWEN4EXP_NO_GDN_PANEL"};
 for(unsigned bits=0;bits<128;bits++){
  for(unsigned j=0;j<7;j++)if(bits&(1u<<j))setenv(envs[j],"1",1);else unsetenv(envs[j]);
  for(unsigned rows:{1u,2u,3u,4u,8u})for(uint64_t in_dim:{2559ull,2560ull,2592ull})for(uint64_t qkv_dim:{512ull,513ull,516ull})for(uint64_t gate_dim:{512ull,515ull,520ull})for(unsigned off:{0u,1u,2u,4u})for(bool supported:{false,true}){
   dp=supported;unsigned n_tokens=rows;struct {const void*weights[4];const void*x;}a{{(void*)(uintptr_t)(4096+off),(void*)(uintptr_t)(8192+off),(void*)12288,(void*)16384},(void*)20480};
   size_t gdn_panel=4*80*34+16;
   bool actual=('''+api+r''');
   bool width=rows<=2||(rows==3&&!(bits&3));
   bool expected=width&&in_dim==2560&&qkv_dim>512&&gate_dim>512&&supported&&!(bits&60)&&!(off&1);
   assert(actual==expected);
   bool caller=('''+graph+r''');assert(caller==(width&&!(bits&32)));
   bool staged=('''+stage+r''');
   bool want_stage=(rows!=3||((qkv_dim%4==0)&&(gate_dim%4==0)))&&(off%4==0)&&!(bits&64);
   assert(staged==want_stage);cases++;
  }
 }
 printf("PASS actual GDN API/graph/staging predicates: %u cases; width3 diagnostics and partial tiles\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='gdn-r3-guards-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-O2','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
