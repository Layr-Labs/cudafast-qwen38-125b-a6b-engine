"""Actual metadata host helper controls and optional CUDA wrapper/kernel test.
--gpu runs synthetic CUDA tests, not a full engine or model benchmark.
"""
from pathlib import Path
import os,sys,subprocess,tempfile
root=Path(__file__).resolve().parents[1]
def extract(s,name):
 a=s.index(name+'(');a=s.rfind('\n',0,a)+1;b=s.index('{',a);e=b+1;depth=1
 while depth:depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
graph=(root/'ds4_qwen4exp_graph.inc').read_text()
helper=extract(graph,'qwen4exp_publish_metadata')
head=r'''
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
typedef struct {int id;} ds4_gpu_tensor;
typedef struct {ds4_gpu_tensor*d_adopt,*d_pos;uint32_t adopt_state,adopt_device,pos;} ds4_qwen4exp_session;
static int calls,fail_publish,oldcalls;static unsigned mask;
static int ds4_gpu_qwen4exp_publish_metadata(ds4_gpu_tensor*p,uint32_t pv,ds4_gpu_tensor*a,uint32_t av,ds4_gpu_tensor*q,uint32_t qv){assert(!p&&pv==0);calls++;mask=(a?2:0)|(q?4:0);assert(av==2||av==0);assert(qv==19);return !fail_publish;}
static int ds4_gpu_qwen4exp_update_dpos(ds4_gpu_tensor*t,uint32_t v){(void)t;(void)v;oldcalls++;return fail_publish!=oldcalls;}
'''
host=head+helper+r'''
int main(){ds4_gpu_tensor f[2]={{0},{1}};
for(unsigned m=0;m<4;m++)for(int failure=0;failure<2;failure++){
 ds4_qwen4exp_session s={m&1?f:0,m&2?f+1:0,2,0,19};calls=0;oldcalls=0;fail_publish=failure;
 bool ok=qwen4exp_publish_metadata(&s);
#ifndef PORTABLE
 assert(calls==1&&mask==((m&1?2:0)|(m&2?4:0))&&ok==!failure);
 if(failure)assert(s.adopt_device==(m&1?UINT32_MAX:0));
#else
 assert(calls==0&&ok==(!failure||m==0));
#endif
 assert(s.adopt_state==2);
 fail_publish=0;oldcalls=0;assert(qwen4exp_publish_metadata(&s));assert(s.adopt_device==(m&1?2:0));
}
ds4_qwen4exp_session s={f,f+1,2,2,19};calls=0;fail_publish=0;oldcalls=0;
assert(qwen4exp_publish_metadata(&s)&&s.adopt_device==2);
#ifndef PORTABLE
assert(mask==4);
#endif
s.adopt_state=0;fail_publish=1;oldcalls=0;assert(!qwen4exp_publish_metadata(&s)&&s.adopt_state==0&&s.adopt_device==UINT32_MAX);
fail_publish=0;oldcalls=0;assert(qwen4exp_publish_metadata(&s)&&s.adopt_device==0);
#ifdef PORTABLE
s.adopt_state=2;s.adopt_device=0;oldcalls=0;fail_publish=2;assert(!qwen4exp_publish_metadata(&s)&&s.adopt_device==2&&oldcalls==2);
#endif
puts("PASS actual metadata-only session helper: 2-field masks, failure/retry, pending-zero reset, portable partial success");}
'''
with tempfile.TemporaryDirectory(prefix='qwen-metadata-') as tmp:
 p=Path(tmp);(p/'host.c').write_text(host)
 for flags in [[],['-D__APPLE__','-DPORTABLE']]:
  subprocess.run(['cc','-std=c11','-O2',*flags,str(p/'host.c'),'-o',str(p/'host')],check=True);subprocess.run([str(p/'host')],check=True)
 if '--gpu' not in sys.argv:sys.exit(0)
 src=(root/'ds4_cuda.cu').read_text()
 cuda=r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#define CK(x) assert((x)==cudaSuccess)
typedef struct{void*ptr;uint64_t bytes;int device_id;} ds4_gpu_tensor;
static int g_n_gpus=1;static struct{int device_id;}g_gpu[1];static cudaStream_t stream;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->device_id<0?0:t->device_id;}
static cudaStream_t cuda_decode_stream(){return stream;}
static int cuda_ok(cudaError_t e,const char*){return e==cudaSuccess;}
'''+extract(src,'qwen4exp_publish_metadata_kernel')+'\n'+extract(src,'ds4_gpu_qwen4exp_publish_metadata')+r'''
int main(){CK(cudaGetDevice(&g_gpu[0].device_id));CK(cudaStreamCreate(&stream));uint32_t*d;CK(cudaMalloc(&d,20));
ds4_gpu_tensor f[3]={{d+1,4,0},{d+2,4,0},{d+3,4,0}};unsigned checks=0;
for(unsigned mask=0;mask<8;mask++){
 uint32_t initial[5]={11,12,13,14,15},got[5];CK(cudaMemcpy(d,initial,20,cudaMemcpyHostToDevice));
 assert(ds4_gpu_qwen4exp_publish_metadata(mask&1?f:0,UINT32_MAX,mask&2?f+1:0,42,mask&4?f+2:0,77));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(got,d,20,cudaMemcpyDeviceToHost));
 assert(got[0]==11&&got[4]==15&&got[1]==(mask&1?UINT32_MAX:12)&&got[2]==(mask&2?42:13)&&got[3]==(mask&4?77:14));checks++;
 cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));assert(ds4_gpu_qwen4exp_publish_metadata(mask&1?f:0,0,mask&2?f+1:0,1,mask&4?f+2:0,2));CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
 for(unsigned r=0;r<20;r++){uint32_t change[5]={r+30,r+40,r+50,r+60,r+70};CK(cudaMemcpy(d,change,20,cudaMemcpyHostToDevice));CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(got,d,20,cudaMemcpyDeviceToHost));assert(got[0]==change[0]&&got[4]==change[4]&&got[1]==(mask&1?0:change[1])&&got[2]==(mask&2?1:change[2])&&got[3]==(mask&4?2:change[3]));checks++;}
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
}
uint32_t before[5],after[5];CK(cudaMemcpy(before,d,20,cudaMemcpyDeviceToHost));
for(int which=0;which<5;which++){ds4_gpu_tensor bad=f[0];if(which==0)bad.ptr=0;if(which==1)bad.bytes=3;if(which==2)bad.ptr=(char*)d+1;if(which==3)bad.device_id=1;if(which==4)bad.ptr=f[1].ptr;assert(!ds4_gpu_qwen4exp_publish_metadata(&bad,8,f+1,9,f+2,10));}
g_gpu[0].device_id++;assert(!ds4_gpu_qwen4exp_publish_metadata(f,1,0,0,0,0));assert(ds4_gpu_qwen4exp_publish_metadata(0,1,0,2,0,3));g_gpu[0].device_id--;CK(cudaDeviceSynchronize());CK(cudaMemcpy(after,d,20,cudaMemcpyDeviceToHost));for(int i=0;i<5;i++)assert(before[i]==after[i]);
CK(cudaFree(d));CK(cudaStreamDestroy(stream));printf("PASS actual metadata CUDA wrapper/kernel: %u eager/replay cases and malformed controls\n",checks);}
'''
 (p/'test.cu').write_text(cuda)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-std=c++17','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
