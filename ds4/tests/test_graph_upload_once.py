"""Actual decode-registry lifecycle with runtime mocks and optional real CUDA."""
from pathlib import Path
import os,subprocess,tempfile,sys,re
s=(Path(__file__).resolve().parents[1]/'ds4_cuda.cu').read_text()
def fn(name):
 m=re.search(r'^(?:static |extern "C" )[^;\n]*\b'+name+r'\s*\([^;]*?\)\s*\{',s,re.M);assert m,name
 a=m.start();b=m.end()-1;e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
a=s.index('struct ds4_decode_graph_key {');b=s.index('extern "C" int ds4_gpu_decode_graphs_supported',a);defs=s[a:b]
body='\n'.join(fn(n) for n in ['cuda_decode_graph_entry_kill','ds4_gpu_decode_graphs_invalidate','cuda_decode_graph_find','ds4_gpu_decode_graph_prefetch','ds4_gpu_decode_graph_begin','ds4_gpu_decode_graph_end'])
head=r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define CUDA_DECODE_GRAPH_LAYERS 64
#define CUDA_DECODE_GRAPH_ISLANDS 4
#define CUDA_DECODE_GRAPH_VARIANTS 4
static int upload_on=1,supported=1;
static int cuda_decode_graph_upload_on(){return upload_on;}
static int ds4_gpu_decode_graphs_supported(){return supported;}
static void*cuda_cublas_for_tier(int){return 0;}
'''
mocks=r'''
typedef void*cudaGraphExec_t;typedef void*cudaGraph_t;typedef void*cudaStream_t;typedef int cudaError_t;
static const int cudaSuccess=0,cudaStreamCaptureModeGlobal=0;
static int uploads,launches,destroys,fail_upload,fail_launch,fail_instantiate,fail_capture;static char order[256];static int used;
static void record(char c){order[used++]=c;order[used]=0;}
static int cudaGraphUpload(void*,void*){uploads++;record('U');if(fail_upload){fail_upload--;return 1;}return 0;}
static int cudaGraphLaunch(void*,void*){launches++;record('L');if(fail_launch){fail_launch=0;return 1;}return 0;}
static int cudaGraphExecDestroy(void*){destroys++;return 0;}
static int cudaGraphDestroy(void*){return 0;}
static int cudaGraphInstantiate(void**p,void*,void*,void*,unsigned){record('I');*p=(void*)1;return fail_instantiate;}
static int cudaStreamCreate(void**p){*p=(void*)2;return 0;}
static int cudaStreamBeginCapture(void*,int){return 0;}
static int cudaStreamEndCapture(void*,void**p){*p=(void*)3;return fail_capture;}
static int cublasSetStream(void*,void*){return 0;}
static int cudaGetLastError(){return 0;}
static const char*cudaGetErrorString(int){return "injected";}
static int cuda_ok(int e,const char*){return e==0;}
'''
host=head+mocks+defs+body+r'''
static void capture(ds4_decode_graph_key*k){assert(ds4_gpu_decode_graph_begin(k)==-1);assert(ds4_gpu_decode_graph_begin(k)==0);assert(ds4_gpu_decode_graph_end(k)==0);}
int main(){ds4_decode_graph_key k={0},other={0};other.variant=2;
assert(!ds4_gpu_decode_graph_prefetch(&k));capture(&k);assert(!strcmp(order,"ILU"));auto*e=cuda_decode_graph_find(&k);assert(e->upload_enqueued&&uploads==1);for(int i=0;i<10;i++)assert(ds4_gpu_decode_graph_prefetch(&k));assert(uploads==1);assert(ds4_gpu_decode_graph_begin(&k)==1);
capture(&other);assert(uploads==2);g_decode_graph_capturing=1;assert(!ds4_gpu_decode_graph_prefetch(&k));g_decode_graph_capturing=0;supported=0;assert(!ds4_gpu_decode_graph_prefetch(&k));supported=1;
ds4_gpu_decode_graphs_invalidate();assert(!e->upload_enqueued&&e->state==0);fail_upload=2;capture(&k);assert(!e->upload_enqueued);assert(!ds4_gpu_decode_graph_prefetch(&k)&&!e->upload_enqueued);assert(ds4_gpu_decode_graph_prefetch(&k)&&e->upload_enqueued);int u=uploads;assert(ds4_gpu_decode_graph_prefetch(&k)&&uploads==u);
fail_launch=1;assert(ds4_gpu_decode_graph_begin(&k)==-1&&e->state==3&&!e->exec&&!e->upload_enqueued);assert(!ds4_gpu_decode_graph_prefetch(&k));
ds4_gpu_decode_graphs_invalidate();upload_on=0;capture(&k);assert(!e->upload_enqueued&&uploads==u);assert(!ds4_gpu_decode_graph_prefetch(&k));upload_on=1;assert(ds4_gpu_decode_graph_prefetch(&k)&&uploads==u+1);
ds4_gpu_decode_graphs_invalidate();e->upload_enqueued=true;cuda_decode_graph_entry_kill(e);assert(!e->upload_enqueued);e->upload_enqueued=true;ds4_gpu_decode_graphs_invalidate();assert(!e->upload_enqueued);
assert(ds4_gpu_decode_graph_begin(&k)==-1);assert(ds4_gpu_decode_graph_begin(&k)==0);fail_launch=1;u=uploads;assert(ds4_gpu_decode_graph_end(&k)==-1&&uploads==u&&!e->exec&&!e->upload_enqueued);
ds4_gpu_decode_graphs_invalidate();capture(&k);assert(e->exec==(void*)1&&e->upload_enqueued&&uploads==u+1); // numeric handle reused, fresh upload
for(unsigned i=1;i<4;i++){other.variant=i+10;capture(&other);}other.variant=100;assert(ds4_gpu_decode_graph_begin(&other)==-1&&!ds4_gpu_decode_graph_prefetch(&other));
ds4_gpu_decode_graphs_invalidate();assert(ds4_gpu_decode_graph_begin(&k)==-1);assert(ds4_gpu_decode_graph_begin(&k)==0);fail_capture=1;u=uploads;assert(ds4_gpu_decode_graph_end(&k)==-1&&!e->exec&&!e->upload_enqueued&&uploads==u);fail_capture=0;
ds4_gpu_decode_graphs_invalidate();assert(ds4_gpu_decode_graph_begin(&k)==-1);assert(ds4_gpu_decode_graph_begin(&k)==0);fail_instantiate=1;assert(ds4_gpu_decode_graph_end(&k)==-1&&!e->exec&&!e->upload_enqueued&&uploads==u);fail_instantiate=0;
assert(!ds4_gpu_decode_graph_prefetch(NULL));other.il=CUDA_DECODE_GRAPH_LAYERS;assert(!ds4_gpu_decode_graph_prefetch(&other));
puts("PASS actual registry lifecycle: success/failure retries, first/replay launch, key variants, invalidate/kill/reused handle, disabled/capture guards");}
'''
gpuhead=r'''
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
static int uploads;
static cudaError_t tracked_upload(cudaGraphExec_t e,cudaStream_t s){uploads++;return cudaGraphUpload(e,s);}
#define cudaGraphUpload tracked_upload
static int cublasSetStream(void*,cudaStream_t){return 0;}
static int cuda_ok(cudaError_t e,const char*){return e==cudaSuccess;}
'''
gpu=head+gpuhead+defs+body+r'''
__global__ void compute(const int*x,int*y){if(!threadIdx.x){unsigned long long start=clock64();while(clock64()-start<1000){}*y=*x*3+1;}}
int main(){int*x,*y;CK(cudaMalloc(&x,4));CK(cudaMalloc(&y,4));ds4_decode_graph_key k={0};k.cur_hc=x;k.after_attn_hc=y;unsigned checked=0;
for(int repeat=0;repeat<2;repeat++){
 ds4_gpu_decode_graphs_invalidate();int start=uploads;
 for(int i=0;i<80;i++){
  if(i==40)ds4_gpu_decode_graphs_invalidate();int input=i+repeat*100,output=0;CK(cudaMemcpy(x,&input,4,cudaMemcpyHostToDevice));int state=ds4_gpu_decode_graph_begin(&k);
  if(state!=1){compute<<<1,32,0,state==0?g_decode_graph_stream:0>>>(x,y);CK(cudaGetLastError());if(state==0)assert(ds4_gpu_decode_graph_end(&k)==0);}
  // Repeat-upload oracle explicitly issues the old optional scheduling call.
  if(repeat&&state>=0){auto*entry=cuda_decode_graph_find(&k);CK(cudaGraphUpload(entry->exec,g_decode_graph_stream));}
  ds4_gpu_decode_graph_prefetch(&k);CK(cudaDeviceSynchronize());CK(cudaMemcpy(&output,y,4,cudaMemcpyDeviceToHost));assert(output==input*3+1);checked++;
 }
 assert(uploads-start==(repeat?80:2));
}
ds4_gpu_decode_graphs_invalidate();CK(cudaFree(x));CK(cudaFree(y));if(g_decode_graph_stream)CK(cudaStreamDestroy(g_decode_graph_stream));printf("PASS actual CUDA registry: %u changed-input captures/replays/reinstantiations, once versus repeated optional uploads\n",checked);}
'''
with tempfile.TemporaryDirectory(prefix='graph-upload-once-') as tmp:
 p=Path(tmp);(p/'host.cpp').write_text(host);subprocess.run(['c++','-O2','-std=c++17',str(p/'host.cpp'),'-o',str(p/'host')],check=True);subprocess.run([str(p/'host')],check=True)
 if '--gpu' in sys.argv:
  (p/'gpu.cu').write_text(gpu);subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-std=c++17','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'gpu.cu'),'-o',str(p/'gpu')],check=True);subprocess.run([str(p/'gpu')],check=True)
