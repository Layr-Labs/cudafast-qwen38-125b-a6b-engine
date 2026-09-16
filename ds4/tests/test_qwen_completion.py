"""Execute actual CUDA completion and Qwen routing helpers with fault mocks.
Both CUDA and portable routing branches compile. No CUDA timing/model claim.
"""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[1]
a=(root/'ds4_cuda.cu').read_text();b=(root/'ds4_qwen4exp_graph.inc').read_text()
def fn(s,name):
 p=s.index(name+'(');p=s.rfind('\n',0,p)+1;q=s.index('{',p);e=q+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[p:e]
helpers='\n'.join(fn(a,n)for n in ['ds4_gpu_end_commands','ds4_gpu_synchronize','ds4_gpu_complete_commands'])+'\n'+fn(b,'qwen4exp_complete_commands')
assert b.count('!qwen4exp_complete_commands()')==3
assert 'getenv' not in fn(a,'ds4_gpu_complete_commands')
source=r'''
#include <cassert>
#include <cstdio>
#include <cstring>
static int g_cuda_end_stream_sync;
static int globals,streams,fail_global,fail_stream;
static const char *error_label;
static int cudaDeviceSynchronize(){globals++;return globals==fail_global?1:0;}
static int cudaStreamSynchronize(int s){assert(s==0);streams++;return fail_stream;}
static int cuda_ok(int e,const char*label){if(e)error_label=label;return !e;}
'''+helpers+r'''
static void reset(int mode){g_cuda_end_stream_sync=mode;globals=streams=fail_global=fail_stream=0;error_label=nullptr;}
int main(){
 reset(0);assert(qwen4exp_complete_commands());
#ifdef DS4_ROCM_BUILD
 assert(globals==2&&streams==0);
#else
 assert(globals==1&&streams==0);
#endif
 reset(1);assert(qwen4exp_complete_commands()&&globals==1&&streams==1);
 reset(0);fail_global=1;assert(!qwen4exp_complete_commands()&&globals==1&&streams==0&&!strcmp(error_label,"end commands"));
 reset(1);fail_stream=1;assert(!qwen4exp_complete_commands()&&globals==0&&streams==1&&!strcmp(error_label,"end commands stream"));
 reset(1);fail_global=1;assert(!qwen4exp_complete_commands()&&globals==1&&streams==1&&!strcmp(error_label,"synchronize"));
 // Runtime failures, including capture-forbidden synchronization, propagate
 // from the original first completion API without retries or second calls.
 reset(0);fail_global=1;assert(!qwen4exp_complete_commands()&&globals==1);
#ifdef DS4_ROCM_BUILD
 puts("PASS portable Qwen commit/wait unchanged with first/second error propagation");
#else
 puts("PASS CUDA Qwen completion: one default global wait, diagnostic global guarantee, first/second failures, no retries");
#endif
}
'''
with tempfile.TemporaryDirectory(prefix='qwen-completion-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 for flags in [[],['-DDS4_ROCM_BUILD']]:
  subprocess.run(['c++','-O2','-std=c++17',*flags,str(p/'test.cpp'),'-o',str(p/'test')],check=True)
  subprocess.run([str(p/'test')],check=True)

# Optional real CUDA completion/capture tests, independent of model weights.
import sys,os
if '--gpu' in sys.argv:
 gpu_source=r'''
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
static int g_cuda_end_stream_sync;
static int cuda_ok(cudaError_t e,const char*){return e==cudaSuccess;}
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);return 2;}}while(0)
'''+helpers+r'''
__global__ void delayed_publish(unsigned*p){unsigned long long t=clock64();while(clock64()-t<100000000ull){}*p=0x51c0ffeeu;}
int main(){
 cudaStream_t independent,capture;cudaEvent_t done;unsigned*device;
 CK(cudaStreamCreateWithFlags(&independent,cudaStreamNonBlocking));CK(cudaStreamCreate(&capture));CK(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));CK(cudaMalloc(&device,4));
 for(int mode=0;mode<2;mode++){
  g_cuda_end_stream_sync=mode;CK(cudaMemset(device,0,4));
  delayed_publish<<<1,1,0,independent>>>(device);CK(cudaGetLastError());CK(cudaEventRecord(done,independent));
  if(!qwen4exp_complete_commands())return 3;
  CK(cudaEventQuery(done));unsigned value=0;CK(cudaMemcpy(&value,device,4,cudaMemcpyDeviceToHost));assert(value==0x51c0ffeeu);
  CK(cudaStreamBeginCapture(capture,cudaStreamCaptureModeGlobal));
  assert(!qwen4exp_complete_commands());
  cudaGraph_t graph=nullptr;cudaError_t e=cudaStreamEndCapture(capture,&graph);assert(e!=cudaSuccess&&graph==nullptr);(void)cudaGetLastError();
 }
 CK(cudaFree(device));CK(cudaEventDestroy(done));CK(cudaStreamDestroy(capture));CK(cudaStreamDestroy(independent));
 puts("PASS real CUDA completion: independent nonblocking-stream work complete in both modes; active capture still refused");
}
'''
 with tempfile.TemporaryDirectory(prefix='qwen-completion-gpu-') as t:
  p=Path(t);(p/'test.cu').write_text(gpu_source)
  subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
  subprocess.run([str(p/'test')],check=True)
