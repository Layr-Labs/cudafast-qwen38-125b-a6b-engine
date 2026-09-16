#!/usr/bin/env python3
"""Execute the real CUDA entry's control flow with launch stubs and optional
backend binding, including failure propagation. These are not GPU tests."""
from pathlib import Path
import re,subprocess,tempfile,os
root=Path(__file__).resolve().parents[1]
src=(root/'ds4_cuda.cu').read_text()
def definition(text,name):
    at=text.index(name+'(');a=text.rfind('\n',0,at)+1;b=text.index('\n}',at)+2;return text[a:b]+'\n'
code='struct indexer_top1_pair { float value; uint32_t index; };\n'
for n in ['indexer_top1_wide_shape','indexer_top1_ranges_overlap','ds4_gpu_indexer_top1_scratch_tensor']:code+=definition(src,n)
code=re.sub(r'indexer_top1_chunks_kernel<<<.*?>>>','indexer_top1_chunks_kernel',code,flags=re.S)
code=re.sub(r'indexer_top1_finish_kernel<<<.*?>>>','indexer_top1_finish_kernel',code,flags=re.S)
prefix=r'''
#include <cstdint>
#include <cstdlib>
#include <cstdio>
struct ds4_gpu_tensor {void *ptr; uint64_t bytes; int device_id;};
struct indexer_top1_pair;
static int original,chunks_called,finish_called,error_call,errors,old_result=1,g_cuda_no_top1;
static uint32_t observed_n,observed_chunks;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->device_id;}
static int cudaGetLastError(){return ++errors==error_call?1:0;}
static int cuda_ok(int e,const char*){return !e;}
static int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor*,const ds4_gpu_tensor*,uint32_t,uint32_t,uint32_t){original++;return old_result;}
static void indexer_top1_chunks_kernel(indexer_top1_pair*,const float*,uint32_t n,uint32_t c){chunks_called++;observed_n=n;observed_chunks=c;}
static void indexer_top1_finish_kernel(uint32_t*,const indexer_top1_pair*,uint32_t c){finish_called++;if(c!=observed_chunks)abort();}
static void need(bool b){if(!b){fprintf(stderr,"top1 control FAIL\n");exit(1);}}
'''
main=r'''
int main(){
 ds4_gpu_tensor out{(void*)0x100000,64,0},scores{(void*)0x200000,4500000,0},scratch{(void*)0x1000000,4096,0};
 unsigned checks=0;
 auto run=[&](uint32_t n,uint32_t rows,int expected,bool wide){original=chunks_called=finish_called=errors=0;
  int rc=ds4_gpu_indexer_top1_scratch_tensor(&out,&scores,&scratch,n,rows);
  if(rc!=expected)fprintf(stderr,"case %u n=%u rows=%u rc=%d expected=%d\n",checks,n,rows,rc,expected);need(rc==expected);if(expected){need(original==!wide&&chunks_called==wide&&finish_called==wide);}
  checks++;};
 run(248320,2,1,true);need(observed_n==248320&&observed_chunks==61);
 for(uint32_t n:{65535u,1048577u})run(n,1,1,false);
 run(248320,8,0,false);out.bytes=128;scores.bytes=9000000;run(248320,8,1,false);
 scratch.bytes=975;run(248320,2,1,false);scratch.bytes=976;run(248320,2,1,true);
 scratch.ptr=(void*)0x1000004;run(248320,2,1,false);scratch.ptr=(void*)0x1000000;
 scratch.ptr=scores.ptr;run(248320,2,1,false);scratch.ptr=out.ptr;run(248320,2,1,false);scratch.ptr=(void*)0x1000000;
 scratch.device_id=1;run(248320,2,1,false);scratch.device_id=0;
 scores.device_id=1;run(248320,2,1,false);scores.device_id=0;
 g_cuda_no_top1=1;run(248320,2,1,false);g_cuda_no_top1=0;
 setenv("DS4_CUDA_NO_TOP1_WIDE","1",1);run(248320,2,1,false);unsetenv("DS4_CUDA_NO_TOP1_WIDE");
 error_call=1;run(248320,2,0,true);need(chunks_called==1&&finish_called==0&&original==0);
 error_call=2;run(248320,2,0,true);need(chunks_called==1&&finish_called==1&&original==0);error_call=0;
 old_result=0;run(65535,2,0,false);need(original==1&&chunks_called==0);old_result=1;
 run(0,2,0,false);run(248320,0,0,false);
 printf("Top1 actual wrapper: %u capacity/alias/device/valve/failure controls PASS\n",checks);
}
'''
# initializer_list is used by the concise boundary loop.
prefix='#include <initializer_list>\n'+prefix
q=(root/'ds4_qwen4exp_graph.inc').read_text()
weak='''extern "C" int ds4_gpu_indexer_top1_scratch_tensor(ds4_gpu_tensor*,const ds4_gpu_tensor*,ds4_gpu_tensor*,uint32_t,uint32_t) __attribute__((weak));\n'''
helper=definition(q,'qwen4exp_compact_top1')
weaktest=r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
struct ds4_gpu_tensor{int id;};
static unsigned old_calls,new_calls;
static int answer;
static int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor*a,const ds4_gpu_tensor*b,uint32_t v,uint32_t r,uint32_t k){old_calls++;if(a->id!=1||b->id!=2||v!=248320||r!=2||k!=1)abort();return answer;}
'''+weak+helper+r'''
#if PROVIDE_BACKEND
extern "C" int ds4_gpu_indexer_top1_scratch_tensor(ds4_gpu_tensor*a,const ds4_gpu_tensor*b,ds4_gpu_tensor*c,uint32_t v,uint32_t r){new_calls++;if(a->id!=1||b->id!=2||c->id!=3||v!=248320||r!=2)abort();return answer;}
#endif
int main(){ds4_gpu_tensor a{1},b{2},c{3};for(int rc=0;rc<2;rc++){answer=rc;if(qwen4exp_compact_top1(&a,&b,&c,248320,2)!=rc)abort();}
if(old_calls!=2*(1-PROVIDE_BACKEND)||new_calls!=2*PROVIDE_BACKEND)abort();printf("Optional top1 backend=%d: forwarding/return PASS\n",PROVIDE_BACKEND);}
'''
with tempfile.TemporaryDirectory(prefix='top1-controls-') as d:
 d=Path(d)
 for name,contents,flags in [('api',prefix+code+main,[]),('absent',weaktest,['-DPROVIDE_BACKEND=0']),('present',weaktest,['-DPROVIDE_BACKEND=1'])]:
  p=d/(name+'.cpp');p.write_text(contents);exe=d/name
  subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',*flags,str(p),'-o',str(exe)],check=True)
  subprocess.run([str(exe)],check=True)
