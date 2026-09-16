#!/usr/bin/env python3
"""Actual CUDA API body with runtime/resolver/launch facades; no GPU execution."""
import re,subprocess,tempfile,os
from pathlib import Path
R=Path(__file__).resolve().parents[2];s=(R/'ds4/ds4_cuda.cu').read_text()
def block(m):
 a=s.index(m);p=s.index('{',a);i=p+1;n=1
 while n:n+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[a:i]
x=block('static bool qwen4exp_embed_disjoint(')+'\n'+block('extern "C" int ds4_gpu_qwen4exp_embed_short_tensor(')
x=re.sub(r'qwen4exp_embed_short_kernel<<<.*?>>>\(', 'qwen4exp_embed_short_kernel(',x,flags=re.S)
c=r'''
#include <cstdint>
#include <cassert>
#include <cstdio>
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;int device_id;};
static int g_n_gpus=1;static struct {int device_id;} g_gpu[1]={{0}};
static int current=0,resolved_tier=-1;
static int fail,launches,resolves,queries,seen0,seen1,seenrows;static uintptr_t weight=0x800000;
static int cudaGetDevice(int*p){queries++;*p=current;return fail==1?1:0;}
static int cudaGetLastError(){return fail==3?1:0;}
static int cuda_ok(int e,const char*){return e==0;}
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->device_id<0?0:t->device_id;}
static const char*cuda_resolve_weight_ptr(const void*,uint64_t,uint64_t,int tier,const char*){resolved_tier=tier;resolves++;return fail==2?nullptr:(const char*)weight;}
static void qwen4exp_embed_short_kernel(float*,float*,int32_t*,const unsigned char*,int32_t a,int32_t b,uint32_t n){launches++;seen0=a;seen1=b;seenrows=n;}
'''+x+r'''
int main(){unsigned tests=0;
 ds4_gpu_tensor H{(void*)0x100000,81920,0},R{(void*)0x200000,20480,0},T{(void*)0x300000,8,0};
 auto run=[&](ds4_gpu_tensor h,ds4_gpu_tensor r,ds4_gpu_tensor t,unsigned rows,int a,int b,int expected){launches=resolves=queries=0;int got=ds4_gpu_qwen4exp_embed_short_tensor(&h,&r,&t,(void*)0x400000,8*2720,0,8,8,rows,a,b);assert(got==expected);tests++;};
 for(unsigned rows=1;rows<=2;rows++)for(int a: {0,7}){fail=0;run(H,R,T,rows,a,7,1);assert(launches==1&&resolves==1&&queries==1&&seen0==a&&seen1==7&&seenrows==(int)rows);}
 current=3;g_gpu[0].device_id=3;run(H,R,T,2,0,7,1);assert(resolved_tier==0);current=0;run(H,R,T,2,0,7,0);assert(!launches);g_gpu[0].device_id=0;
 for(fail=1;fail<=3;fail++){run(H,R,T,2,0,7,0);assert(launches==(fail==3));assert(resolves==(fail!=1));}fail=0;
 run(H,R,T,2,0,-1,0);assert(!queries);run(H,R,T,2,0,8,0);assert(!queries);run(H,R,T,1,0,-1,1);
 for(unsigned which=0;which<3;which++)for(unsigned kind=0;kind<5;kind++){
  auto h=H,r=R,t=T;auto *v=which==0?&h:which==1?&r:&t;
  if(kind==0)v->ptr=nullptr;if(kind==1)v->bytes=0;if(kind==2)v->device_id=1;if(kind==3)v->ptr=(void*)((uintptr_t)v->ptr+2);if(kind==4)v->bytes=UINT64_MAX;
  run(h,r,t,2,0,7,0);assert(!launches);
 }
 for(unsigned which=0;which<3;which++){
  weight=(uintptr_t)(which==0?H.ptr:which==1?R.ptr:T.ptr)+4;run(H,R,T,2,0,7,0);assert(resolves==1&&!launches);
 }weight=0x800001;run(H,R,T,2,0,7,0);assert(!launches);weight=0x800000;
 assert(qwen4exp_embed_disjoint((void*)0x1000,16,(void*)0x1010,16));assert(!qwen4exp_embed_disjoint((void*)0x1000,17,(void*)0x1010,16));
 printf("PASS %u extracted API controls; query/resolver/launch errors, bounds/device/overflow/alias; no retry\n",tests);
}
'''
c=c.replace('#include <cstdio>','#include <cstdio>\n#include <initializer_list>')
with tempfile.TemporaryDirectory(prefix='short-embed-controls-') as d:
 p=Path(d)/'test.cc';p.write_text(c);exe=Path(d)/'test'
 subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17',str(p),'-o',str(exe)],check=True);subprocess.run([str(exe)],check=True)
