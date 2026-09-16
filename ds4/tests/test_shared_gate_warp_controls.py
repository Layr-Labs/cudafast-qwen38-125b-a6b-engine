#!/usr/bin/env python3
"""Actual final shared-gate dispatch slice; runtime/launches mocked."""
from pathlib import Path
import re,tempfile,subprocess,os
R=Path(__file__).resolve().parents[2];s=(R/'ds4/ds4_cuda_qwen4exp.cu').read_text();a=s.index('    if (specialize_shared && n_tokens >= 1u');b=s.index('    if (!pre_quantized)',a);x=s[a:b]
x=re.sub(r'<<<(.*?)>>>',r'',x,flags=re.S)
x=x.replace('qwen4exp_shared_gate_short_kernel(', 'short_launch(side,').replace('qwen4exp_shared_gate_kernel<DS4_QWEN4EXP_TY_f32>','old_launch<0>').replace('qwen4exp_shared_gate_kernel<-1>','old_launch<-1>')
x=re.sub(r'(old_launch<[^>]+>)\s*\(',r'\1(side,',x)
c=r'''
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#define DS4_QWEN4EXP_TY_f32 0
struct tensor{void*ptr;};struct slab{uint32_t type;};
static int calls,which,used_stream,error;
static void short_launch(int side,float*,const char*,const float*,uint32_t,uint32_t,uint32_t){calls++;which=1;used_stream=side;}
template<int T>static void old_launch(int side,float*,const char*,const float*,uint32_t,uint32_t,uint32_t){calls++;which=T==0?2:3;used_stream=side;}
static int cudaGetLastError(){return error;}
static int cuda_ok(int e,const char*){return !e;}
static int actual(bool specialize_shared,uint32_t n_tokens,uint32_t in_dim,slab*router_slab,int side){tensor z{nullptr};auto*gate_scale=&z;auto*x=&z;const char*router=nullptr;
'''+x+r'''
return 1;}
int main(){unsigned cases=0;for(unsigned rows=1;rows<=3;rows++)for(unsigned dim: {2560u,1280u})for(unsigned ty: {0u,8u})for(int specialized=0;specialized<2;specialized++)for(int disabled=0;disabled<2;disabled++)for(int side: {0,7})for(error=0;error<2;error++){
 if(disabled)setenv("DS4_QWEN4EXP_NO_SHARED_GATE_WARP_TAIL","1",1);else unsetenv("DS4_QWEN4EXP_NO_SHARED_GATE_WARP_TAIL");slab w{ty};calls=which=used_stream=0;
 int ok=actual(specialized,rows,dim,&w,side);int want=specialized&&ty==0?(rows<=2&&dim==2560&&!disabled?1:2):3;assert(ok==!error&&calls==1&&which==want&&used_stream==side);cases++;}
 printf("PASS %u actual dispatch-slice cases: shapes/type/generic/optout, supplied side/origin stream, fatal error without retry\n",cases);}
''';c=c.replace('#include <cstdio>','#include <cstdio>\n#include <initializer_list>')
with tempfile.TemporaryDirectory(prefix='shared-gate-controls-') as d:
 p=Path(d)/'test.cc';p.write_text(c);e=Path(d)/'test';subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17',str(p),'-o',str(e)],check=True);subprocess.run([str(e)],check=True)
