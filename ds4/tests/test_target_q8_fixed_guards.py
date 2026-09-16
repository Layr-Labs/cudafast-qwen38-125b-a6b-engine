"""Exercise actual target dispatch predicates and uint32 padded-index bounds.
This does not execute CUDA dispatch; GPU kernel parity is a separate test.
"""
from pathlib import Path
import subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda.cu').read_text()
def condition(start):
 a=s.index(start)+len('if (');p=a;depth=1
 while depth:depth+=(s[p]=='(')-(s[p]==')');p+=1
 return s[a:p-1]
outer=condition('if (use_dp4a && (n_rows <= 2u || wide_verify3)')
inner=condition('if (target_head && in_dim == 2560u')
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
int main(){unsigned cases=0;
 for(unsigned bits=0;bits<64;bits++)for(uint32_t n_rows:{0u,1u,2u,3u,4u,8u})for(uint64_t in_dim:{320ull,2559ull,2560ull,2592ull})for(uint64_t out_dim:{512ull,513ull,248320ull,4294967295ull,4294967296ull}){
 bool target_head=bits&1,use_dp4a=bits&2,wide_verify3=n_rows==3||n_rows==4;
 const char*wptr=(const char*)(uintptr_t)((bits&4)?4097:4096);
 if(bits&8)setenv("DS4_QWEN4EXP_NO_ROW_TILE","1",1);else unsetenv("DS4_QWEN4EXP_NO_ROW_TILE");
 if(bits&16)setenv("DS4_QWEN4EXP_PAIR_LANES_R2","1",1);else unsetenv("DS4_QWEN4EXP_PAIR_LANES_R2");
 if(bits&32)setenv("DS4_Q8_NO_TARGET_FIXED","1",1);else unsetenv("DS4_Q8_NO_TARGET_FIXED");
 bool actual=('''+outer+') && ('+inner+r''');
 bool expected=(bits&3)==3 && !(bits&60) && (n_rows==1||n_rows==2) && in_dim==2560 && out_dim>=513 && out_dim<=UINT32_MAX;
 assert(actual==expected);cases++;
 }
 for(uint64_t dim:{1ull,3ull,4ull,513ull,248320ull,4294967292ull,4294967293ull,4294967294ull,4294967295ull}){
  uint64_t grid=(dim+3)/4;
  for(uint64_t block:{uint64_t(0),grid-1})for(uint32_t local=0;local<4;local++){
   uint64_t exact=block*4+local;uint32_t row=(uint32_t)block*4u+local;
   assert(exact<=UINT32_MAX && row==exact);
   assert((uint64_t)row*80u*34u==exact*2720u);
   for(uint64_t r=0;r<2;r++)assert(r*dim+row==r*dim+exact);
  }
 }
 printf("PASS actual target guards %u cases and UINT32_MAX padded-index bounds\n",cases);
}
'''
source=source.replace('#include <cassert>','#include <cassert>\n#include <initializer_list>')
with tempfile.TemporaryDirectory(prefix='target-q8-guards-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-O2','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
