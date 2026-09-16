"""Run the actual screen dispatch body with CUDA launches stubbed on the host.

Checks readback count, decline-before-work guards and backend error propagation.
Does not execute the numerical kernels or CUB, or measure GPU performance.
"""
from pathlib import Path
import re, subprocess, tempfile
repo=Path(__file__).resolve().parents[2]
source=(repo/'ds4/ds4_cuda_mtp_native.cuh').read_text()
def body(needle):
    start=source.index('{',source.index(needle));end=start+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start+1:end-1]
screen=re.sub(r'<<<.*?>>>','',body('static int mtp_native_screen_impl('),flags=re.S)
screen=screen.replace('getenv(', 'probe_getenv(')
layout=source[source.index('struct mtp_native_layout {'):source.index('extern "C" int ds4_gpu_mtp_native_screen_init')]
code=r'''
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <vector>
constexpr uint32_t MTP_NATIVE_CAP=2048,MTP_NATIVE_DIM=2560,MTP_NATIVE_MAX_WIDTH=1u<<20;
constexpr uint32_t DS4_MTP_NATIVE_RETRY_ID=UINT32_MAX-1;
struct ds4_gpu_tensor{void *ptr;uint64_t bytes;int tier=0;};
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStreamCaptureStatus=int;
constexpr int cudaSuccess=0,cudaStreamCaptureStatusNone=0;
static unsigned launches,reads,sorts,force_invalid,fail_launch;
static bool key_valve,read_error;static int capture;
static const char *probe_getenv(const char *key){return key_valve&&!strcmp(key,"DS4_MTP_NO_FUSED_SCREEN_KEYS")?"1":nullptr;}
static bool cuda_q8_use_dp4a(){return true;}
static int ds4_tensor_device_idx(const ds4_gpu_tensor *p){return p->tier;}
static int cudaGetDevice(int *p){*p=0;return 0;}
static int cuda_decode_stream(){return 0;}
static int cudaStreamIsCapturing(int,cudaStreamCaptureStatus *p){*p=capture;return 0;}
static const char *cuda_resolve_weight_ptr(const void *map,uint64_t offset,uint64_t,int,const char*){return (const char*)((uintptr_t)map+offset);}
static bool cuda_ok(int e,const char*){return e==0;}
static int cudaMemsetAsync(void *p,int v,size_t n,int){memset(p,v,n);return 0;}
static int cudaGetLastError(){return fail_launch&&launches==fail_launch?1:0;}
template<typename... T>static void quantize_q8_0_f32_rows_warp_kernel(T...){launches++;}
template<bool Screen,bool EmitKeys=false>static void mtp_native_projection_kernel(
        float*,const unsigned char*,const int8_t*,const float*,uint32_t,
        const uint32_t*,uint32_t,uint32_t,uint32_t,uint64_t *keys=nullptr,uint32_t *invalid=nullptr){
    launches++;if(EmitKeys)*invalid=force_invalid;
}
static void mtp_native_keys(uint64_t*,uint32_t *invalid,const float*,uint32_t,uint32_t,uint32_t,uint32_t){launches++;*invalid=force_invalid;}
static void mtp_native_unpack_ids(uint32_t*,const uint64_t*){launches++;}
static int ds4_gpu_tensor_read(ds4_gpu_tensor *p,uint64_t off,void *out,uint64_t n){reads++;if(read_error)return 0;memcpy(out,(char*)p->ptr+off,n);return 1;}
namespace cub {struct DeviceRadixSort{
    template<typename... T>static int SortKeysDescending(T...){sorts++;return 0;}
    template<typename... T>static int SortKeys(T...){sorts++;return 0;}
};}
'''+layout
code+='static bool mtp_native_key_range_disjoint(const void*a,uint64_t an,const void*b,uint64_t bn){'+body('static bool mtp_native_key_range_disjoint(')+'}\n'
code+='static int screen(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,const void *map,uint64_t map_bytes,uint64_t offset,uint32_t in_dim,uint32_t vocab,uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x,bool defer_invalid){'+screen+'}\n'
code+=r'''
int main(){
    std::vector<char> smem(1000000),omem(8192),imem(8192),xmem(10240);
    ds4_gpu_tensor out{omem.data(),omem.size()},ids{imem.data(),imem.size()},scratch{smem.data(),smem.size()},x{xmem.data(),xmem.size()};
    const void *map=(void*)0x500000000ull;unsigned checked=0;
    auto call=[&](bool defer,uint32_t vocab=21000,uint64_t offset=0){launches=reads=sorts=0;return screen(&out,&ids,&scratch,map,100000000,offset,2560,vocab,20000,276,&x,defer);};
    for(bool defer:{false,true})for(bool bad:{false,true})for(bool valve:{false,true}){
        force_invalid=bad;key_valve=valve;int rc=call(defer);
        const bool early=bad&&!defer;
        assert(rc==(early?0:2048));assert(reads==(defer?0:1));
        assert(sorts==(early?0:2));assert(launches==(early?2:4)+(valve?1:0));checked++;
    }
    force_invalid=0;key_valve=false;
    read_error=true;assert(call(false)==-1&&reads==1);assert(call(true)==2048&&reads==0);read_error=false;checked+=2;
    for(unsigned fail=1;fail<=4;fail++){
        fail_launch=fail;assert(call(true)==-1&&reads==0);checked++;
    }fail_launch=0;
    capture=1;assert(call(true)==0&&launches==0&&reads==0);capture=0;checked++;
    g_n_gpus=2;assert(call(true)==0&&launches==0);g_n_gpus=1;checked++;
    ids.tier=1;assert(call(true)==0&&launches==0);ids.tier=0;checked++;
    assert(call(true,UINT32_MAX)==0&&launches==0);checked++;
    assert(call(true,21000,1)==0&&launches==0);checked++;
    uint64_t saved=scratch.bytes;scratch.bytes=4;assert(call(true)==-1&&launches==0);scratch.bytes=saved;checked++;
    for(ds4_gpu_tensor *p:{&out,&ids,&x}){
        void *old=p->ptr;p->ptr=smem.data()+16;assert(call(true)==0&&launches==0&&reads==0);p->ptr=old;checked++;
    }
    const void *oldmap=map;map=smem.data()+16;assert(call(true)==0&&launches==0);map=oldmap;checked++;
    // Adjacent private views remain eligible; wrapping intervals decline.
    void *old=out.ptr;out.ptr=(void*)((uintptr_t)scratch.ptr+scratch.bytes);assert(call(true)==2048&&reads==0);
    out.ptr=(void*)(UINTPTR_MAX-4095);assert(call(true)==0&&launches==0);out.ptr=old;checked+=2;
    std::cout<<"PASS "<<checked<<" actual screen control-flow cases under UBSan; CUDA launches/CUB are stubs\n";
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-deferred-controls-') as d:
    cpp=Path(d)/'controls.cpp';exe=cpp.with_suffix('');cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
