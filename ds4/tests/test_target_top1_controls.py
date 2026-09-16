"""Execute extracted API validation/error routing with fake CUDA launch endpoints."""
from pathlib import Path
import re,subprocess,tempfile
repo=Path(__file__).resolve().parents[2];s=(repo/'ds4/ds4_cuda.cu').read_text()
def extract(name):
 a=s.index(name);a=s.rfind('\n',0,a)+1;b=s.index('{',a);d=1;e=b+1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
api=extract('extern "C" int ds4_gpu_target_top1_tensor(')
assert api.count('cuda_decode_stream()')==2
for stage,name in enumerate(['target_top1_partition_kernel','target_top1_finish_kernel']):
 api,n=re.subn(name+r'<<<.*?>>>\(.*?\);',f'mock_launch({stage});',api,flags=re.S);assert n==1
code=r'''
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <iostream>
struct ds4_gpu_tensor {void *ptr;uint64_t bytes;int device_id;};
struct target_top1_pair {float value;uint32_t id;};
struct {int device_id;}g_gpu[1]={{7}};
int g_n_gpus=1,g_cuda_no_top1=0,current=7,query_error=0,last_error=0,fail_stage=-1,old_calls=0,launches=0;
int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->device_id;}
int cudaGetDevice(int*p){*p=current;return query_error;}
int cudaGetLastError(){int e=last_error;last_error=0;return e;}
bool cuda_ok(int e,const char*){return e==0;}
int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor*,const ds4_gpu_tensor*,uint32_t,uint32_t,uint32_t){old_calls++;return 1;}
void mock_launch(int stage){launches++;last_error=stage==fail_stage?1:0;}
''' + extract('static bool target_top1_disjoint(')+'\n'+api+r'''
int main(){
 ds4_gpu_tensor score={(void*)0x100000,524288,0},out={(void*)0x300000,16,0},scratch={(void*)0x400000,512,0};
 auto run=[&](ds4_gpu_tensor*o,ds4_gpu_tensor*x,ds4_gpu_tensor*p,uint32_t n=65536,uint32_t rows=2){old_calls=launches=last_error=0;return ds4_gpu_target_top1_tensor(o,x,p,n,rows);};
 assert(run(&out,&score,&scratch)&&launches==2&&!old_calls);
 for(int stage:{0,1}){fail_stage=stage;assert(!run(&out,&score,&scratch)&&launches==stage+1&&!old_calls);}fail_stage=-1;
 assert(run(&out,&score,nullptr)&&old_calls==1&&!launches);
 assert(run(&out,&score,&scratch,65535)&&old_calls==1&&!launches);
 assert(run(&out,&score,&scratch,65536,3)&&old_calls==1&&!launches);
 g_cuda_no_top1=1;assert(run(&out,&score,&scratch)&&old_calls==1&&!launches);g_cuda_no_top1=0;
 setenv("DS4_NO_HIERARCHICAL_TARGET_TOP1","1",1);assert(run(&out,&score,&scratch)&&old_calls==1&&!launches);unsetenv("DS4_NO_HIERARCHICAL_TARGET_TOP1");
 for(int which:{0,1,2}){
  ds4_gpu_tensor bad=which==0?out:which==1?score:scratch;bad.ptr=(void*)((uintptr_t)bad.ptr+1);
  assert(!run(which==0?&bad:&out,which==1?&bad:&score,which==2?&bad:&scratch)&&!launches&&!old_calls);
 }
 for(int which:{0,1,2}){
  ds4_gpu_tensor bad=which==0?out:which==1?score:scratch;bad.bytes=1;
  assert(!run(which==0?&bad:&out,which==1?&bad:&score,which==2?&bad:&scratch)&&!launches&&!old_calls);
 }
 ds4_gpu_tensor alias=scratch;alias.ptr=score.ptr;assert(run(&out,&score,&alias)&&old_calls==1&&!launches);
 alias.ptr=out.ptr;assert(run(&out,&score,&alias)&&old_calls==1&&!launches);
 alias=out;alias.ptr=score.ptr;assert(run(&alias,&score,&scratch)&&old_calls==1&&!launches);
 alias=scratch;alias.ptr=(void*)(UINTPTR_MAX-127u);assert(run(&out,&score,&alias)&&old_calls==1&&!launches);
 for(int which:{0,1,2}){ds4_gpu_tensor bad=which==0?out:which==1?score:scratch;bad.device_id=1;assert(!run(which==0?&bad:&out,which==1?&bad:&score,which==2?&bad:&scratch)&&!launches&&!old_calls);}
 current=0;assert(!run(&out,&score,&scratch)&&!launches&&!old_calls);current=7;
 query_error=1;assert(!run(&out,&score,&scratch)&&!launches&&!old_calls);query_error=0;
 assert(!run(nullptr,&score,&scratch)&&!launches&&!old_calls);
 assert(!target_top1_disjoint((void*)16,16,(void*)(UINTPTR_MAX-15),32));
 assert(target_top1_disjoint((void*)16,16,(void*)32,16));
 std::cout<<"PASS extracted API success/fallback/alignment/bounds/alias/device/launch-error controls\n";
}
'''
with tempfile.TemporaryDirectory(prefix='target-top1-control-') as d:
 source=Path(d)/'test.cpp';source.write_text(code);binary=Path(d)/'test'
 subprocess.run(['c++','-O2','-std=c++17',str(source),'-o',str(binary)],check=True);subprocess.run([str(binary)],check=True)
