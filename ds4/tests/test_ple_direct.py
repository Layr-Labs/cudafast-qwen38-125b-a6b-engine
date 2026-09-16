"""Production IQ4 direct-gather controls and finite decoder CUDA parity.
The ATS-gated API may decline locally; direct device-buffer kernel tests do not
claim ATS validation. Optional file-backed test runs only when actual gates pass.
"""
from pathlib import Path
import os,subprocess,tempfile
root=Path(__file__).resolve().parents[1];src=(root/'ds4_cuda_qwen4exp.cu').read_text();hdr=(root/'ds4_gpu.h').read_text();cpu=(root/'ds4_qwen4exp_ple.c').read_text()
def fn(s,name):
 a=s.index(name+'(');a=s.rfind('\n',0,a)+1;b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
assert len({(r*37+b*17+j*13)&255 for r in range(32) for b in range(5) for j in range(16)}) == 256
a=hdr.index('typedef struct {\n    const uint8_t *rows[32];');b=hdr.index('} ds4_qwen4exp_ple_direct_rows;',a)+len('} ds4_qwen4exp_ple_direct_rows;');struct=hdr[a:b]
cap=fn(src,'qwen4exp_ple_direct_capability');prep=fn(src,'ds4_gpu_qwen4exp_ple_direct_prepare');gather=fn(src,'ds4_gpu_qwen4exp_ple_direct_gather');kernel=fn(src,'qwen4exp_ple_direct_kernel')
common=r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
typedef struct{void*ptr;uint64_t bytes;int device_id;}ds4_gpu_tensor;
'''+struct
host=common+r'''
enum{cudaSuccess,cudaDevAttrPageableMemoryAccess,cudaDevAttrPageableMemoryAccessUsesHostPageTables,cudaDevAttrConcurrentManagedAccess,cudaMemoryTypeDevice};
struct cudaPointerAttributes{int type,device;};static int device,attributes=7,queryfail,attrsbad,supported=1,queries;
static int cudaGetDevice(int*p){*p=device;return queryfail;}
static int cudaDeviceGetAttribute(int*p,int a,int){queries++;*p=(attributes>>(a-1))&1;return queryfail;}
static int cudaPointerGetAttributes(cudaPointerAttributes*p,const void*){p->type=cudaMemoryTypeDevice;p->device=device+attrsbad;return 0;}
static int ds4_gpu_decode_graphs_supported(){return supported;}
'''+cap+'\n'+prep+r'''
int main(){alignas(4)float out[5120];uint8_t table[32*90]={};uint64_t ids[32];for(int i=0;i<32;i++)ids[i]=i;ds4_gpu_tensor t={out,sizeof(out),0};ds4_qwen4exp_ple_direct_rows p={};
for(unsigned mask=0;mask<8;mask++){device=mask;attributes=mask;queries=0;int q=ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,32);assert(q==(mask==7));assert(p.row_count==(mask==7?32:0));int nq=queries;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,32)==q&&queries==nq);}
attributes=7;device=8;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,16)==1&&p.output==out);int saved=queries;
for(int exp=0;exp<2;exp++){table[1]=0x7c;table[0]=exp;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,16)==0&&p.row_count==0);}table[0]=table[1]=0;
ids[15]=32;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,16)==-1);ids[15]=15;
attrsbad=1;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,16)==-1);attrsbad=0;
supported=0;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,16)==0);supported=1;
queryfail=1;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&t,table,sizeof(table),32,90,ids,16)==-1);queryfail=0;
for(int bad=0;bad<5;bad++){ds4_gpu_tensor z=t;if(bad==0)z.ptr=0;if(bad==1)z.ptr=(char*)out+1;if(bad==2)z.bytes=1;if(bad==3)z.ptr=table;assert(ds4_gpu_qwen4exp_ple_direct_prepare(&p,&z,table,sizeof(table),32,bad==4?89:90,ids,16)==-1);}
puts("PASS actual preparation: all capability masks/cache, finite fallback, bounds, output alias/alignment/device and errors");}
'''
# Actual CPU codebook, half conversion and decoder serve as the parity oracle.
a=cpu.index('static const int8_t ple_kvalues_iq4nl[16]');b=cpu.index('void ds4_ple_dequant_iq4_nl(');dec=cpu[a:b]+fn(cpu,'ds4_ple_dequant_iq4_nl')
gpu=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define DS4_PLE_IQ4_NL_BLOCK_ELEMS 32
#define DS4_PLE_IQ4_NL_BLOCK_BYTES 18
'''+common+dec+r'''
static cudaStream_t stream;
static cudaStream_t ds4_cuda_qwen4exp_decode_stream(){return stream;}
static int ds4_gpu_decode_graphs_supported(){return 1;}
'''+cap+'\n'+prep+'\n'+kernel+'\n'+gather+r'''
int main(){cudaStream_t st;CK(cudaStreamCreate(&st));stream=st;uint8_t*dq;float*dy;CK(cudaMalloc(&dq,32*90));CK(cudaMalloc(&dy,(32*160+2)*4));
std::vector<uint8_t>q(32*90);std::vector<float>ref(32*160);std::vector<uint32_t>got(32*160+2);uint64_t ids[32];for(int i=0;i<32;i++)ids[i]=i;
ds4_gpu_tensor out={dy+1,32*160*4,0};ds4_qwen4exp_ple_direct_rows p={};CK(cudaGetDevice(&p.physical_device));p.output=out.ptr;p.row_count=32;for(int i=0;i<32;i++)p.rows[i]=dq+i*90;
unsigned halves=0,batches=0;
for(unsigned first=0;first<65536;first+=32){if((first&0x7c00)==0x7c00)continue;
 for(unsigned r=0;r<32;r++){unsigned h=first+r;for(int b=0;b<5;b++){uint8_t*z=q.data()+r*90+b*18;z[0]=h;z[1]=h>>8;for(int j=0;j<16;j++)z[2+j]=(j<<4)|j;}ds4_ple_dequant_iq4_nl(q.data()+r*90,5,ref.data()+r*160);halves++;}
 CK(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));CK(cudaMemset(dy,0xa5,(32*160+2)*4));assert(ds4_gpu_qwen4exp_ple_direct_gather(&out,&p));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(got.data(),dy,got.size()*4,cudaMemcpyDeviceToHost));assert(got.front()==0xa5a5a5a5&&got.back()==0xa5a5a5a5);assert(!memcmp(got.data()+1,ref.data(),ref.size()*4));batches++;
}

// Every packed byte occurs, with independent low/high nibbles and block rows.
for(unsigned r=0;r<32;r++)for(unsigned b=0;b<5;b++){uint8_t*z=q.data()+r*90+b*18;z[0]=0;z[1]=0x3c;for(unsigned j=0;j<16;j++)z[2+j]=(uint8_t)(r*37+b*17+j*13);}
CK(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));
// The gather remains eager on legacy stream0. A blocking-stream consumer
// graph reads fresh output after each changed selected-row launch.
float*dc;CK(cudaMalloc(&dc,32*160*4));cudaGraph_t graph;cudaGraphExec_t exec;
CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));CK(cudaMemcpyAsync(dc,dy+1,32*160*4,cudaMemcpyDeviceToDevice,st));CK(cudaStreamEndCapture(st,&graph));CK(cudaGraphInstantiate(&exec,graph,0));stream=0;
for(unsigned width:{16u,32u})for(unsigned pass=0;pass<20;pass++){
 p.row_count=width;for(unsigned r=0;r<width;r++){unsigned index=((pass&1)?r/2:r)+pass;index%=32;p.rows[r]=dq+index*90;ds4_ple_dequant_iq4_nl(q.data()+index*90,5,ref.data()+r*160);}assert(ds4_gpu_qwen4exp_ple_direct_gather(&out,&p));CK(cudaGraphLaunch(exec,st));CK(cudaStreamSynchronize(st));CK(cudaMemcpy(got.data(),dc,width*160*4,cudaMemcpyDeviceToHost));assert(!memcmp(got.data(),ref.data(),width*160*4));}
CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaFree(dc));stream=st;
ds4_gpu_tensor wrong=out;wrong.ptr=dy;assert(!ds4_gpu_qwen4exp_ple_direct_gather(&wrong,&p));
// Real ATS capability decision, never replaced by the synthetic device test.
int dev;CK(cudaGetDevice(&dev));int capable=qwen4exp_ple_direct_capability(dev);assert(capable>=0);ds4_qwen4exp_ple_direct_rows actual={};
FILE*f=tmpfile();assert(f);assert(fwrite(q.data(),1,q.size(),f)==q.size());fflush(f);void*m=mmap(0,q.size(),PROT_READ,MAP_PRIVATE,fileno(f),0);assert(m!=MAP_FAILED);
int result=ds4_gpu_qwen4exp_ple_direct_prepare(&actual,&out,(const uint8_t*)m,q.size(),32,90,ids,32);
if(capable){assert(result==1);assert(ds4_gpu_qwen4exp_ple_direct_gather(&out,&actual));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(got.data(),dy+1,32*160*4,cudaMemcpyDeviceToHost));for(unsigned r=0;r<32;r++)ds4_ple_dequant_iq4_nl(q.data()+r*90,5,ref.data()+r*160);assert(!memcmp(got.data(),ref.data(),ref.size()*4));puts("PASS actual ATS read-only file mapping");}else{assert(result==0&&actual.row_count==0);puts("PASS actual ATS capability refusal; file-backed execution SKIPPED");}
munmap(m,q.size());fclose(f);CK(cudaFree(dy));CK(cudaFree(dq));CK(cudaStreamDestroy(st));printf("PASS all %u finite halves x16 IQ4 values, %u batches and40 changed-row consumer graph launches\n",halves,batches);}
'''
with tempfile.TemporaryDirectory(prefix='ple-direct-') as tmp:
 p=Path(tmp);(p/'host.cpp').write_text(host);subprocess.run(['c++','-O2','-std=c++17',str(p/'host.cpp'),'-o',str(p/'host')],check=True);subprocess.run([str(p/'host')],check=True)
 (p/'gpu.cu').write_text(gpu);subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-std=c++17','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'gpu.cu'),'-o',str(p/'gpu')],check=True);subprocess.run([str(p/'gpu')],check=True)
