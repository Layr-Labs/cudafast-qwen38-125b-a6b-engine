"""UBSan host execution of extracted ordinary/deferred winner-map bodies.

No CUDA execution or timing. The independent oracle covers marker precedence,
NaN0, invalid packed/original IDs and the upper vocabulary/marker boundary.
"""
from pathlib import Path
import re, subprocess, tempfile
repo=Path(__file__).resolve().parents[2]
source=(repo/'ds4/ds4_cuda_mtp_native.cuh').read_text()
header=(repo/'ds4/ds4_mtp_native_contract.h').read_text()
def body(needle):
    start=source.index('{',source.index(needle));end=start+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start+1:end-1]
marker=re.search(r'^#define DS4_MTP_NATIVE_RETRY_ID .+$',header,re.M).group()
code=r'''
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <random>
static uint32_t raw_bits(float x){uint32_t u;memcpy(&u,&x,4);return u;}
static float value(uint32_t u){float x;memcpy(&x,&u,4);return x;}
#define __float_as_uint raw_bits
'''+marker+'\n'
code+='static void ordinary(uint32_t *winner,const float *logits,const uint32_t *ids,uint32_t count,uint32_t vocab){'+body('__global__ static void mtp_native_map(')+'}\n'
code+='static void deferred(uint32_t *winner,const float *logits,const uint32_t *ids,const uint32_t *invalid,uint32_t count,uint32_t vocab){'+body('__global__ static void mtp_native_map_deferred(')+'}\n'
code+=r'''
int main(){std::mt19937 rng(406379);uint64_t checked=0;
    const uint32_t special[]={0,0x80000000,0x3f800000,0x7f800000,0xff800000,
                              0x7fc00000,0xffc00001,0x7f800001,0xff800001};
    for(uint32_t vocab:{1u,8u,248320u,DS4_MTP_NATIVE_RETRY_ID}){
        for(unsigned i=0;i<100000;i++){
            const uint32_t raw=i<sizeof(special)/4?special[i]:rng();float logit=value(raw);
            uint32_t ids[4]={0u,(uint32_t)(rng()%vocab),vocab-1,UINT32_MAX};
            uint32_t packed=i%7==6?UINT32_MAX:i%7;
            uint32_t old=packed;ordinary(&old,&logit,ids,4,vocab);
            bool nan=(raw&0x7f800000u)==0x7f800000u && (raw&0x007fffffu)!=0;
            uint32_t p=nan?0:packed;
            uint32_t expected=p<4?ids[p]:UINT32_MAX;
            if(expected>=vocab)expected=UINT32_MAX;
            assert(old==expected);
            for(uint32_t invalid:{0u,1u,0x80000000u}){
                uint32_t got=packed;deferred(&got,&logit,ids,&invalid,4,vocab);
                assert(got==(invalid?DS4_MTP_NATIVE_RETRY_ID:expected));
                if(!invalid && got!=UINT32_MAX)assert(got<DS4_MTP_NATIVE_RETRY_ID);
                checked++;
            }
        }
    }
    std::cout<<"PASS "<<checked<<" actual winner-map body cases under UBSan; no GPU execution\n";
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-deferred-map-') as d:
    cpp=Path(d)/'oracle.cpp';exe=cpp.with_suffix('');cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
