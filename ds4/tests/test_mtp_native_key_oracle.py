# Host source-extracted key semantics and alias routing; no GPU execution.
from pathlib import Path
import re,subprocess,tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda_mtp_native.cuh').read_text();main=(repo/'ds4/ds4_cuda.cu').read_text()
def block(text,needle):
 start=text.index('{',text.index(needle));end=start+1;depth=1
 while depth:
  if text[end]=='{':depth+=1
  if text[end]=='}':depth-=1
  end+=1
 return text[start+1:end-1]
old=block(s,'__global__ static void mtp_native_keys(')
old=old[old.index('    if (i >= width)'):].replace('scores[i]','input').replace('keys[i]','result.key')
fused=block(s,'if (EmitKeys)')
fused=fused.replace('keys[row]','result.key')
for key in ['old','fused']:
 val=locals()[key].replace('value == 0.0f','probe_zero(value,ftz)').replace('!isfinite(value)','!probe_finite(value)')
 locals()[key]=val
range_body=block(s,'static bool mtp_native_key_range_disjoint(')
gate=s[s.index('    const bool fuse_keys ='):s.index('    if (fuse_keys)')].replace('getenv("DS4_MTP_NO_FUSED_SCREEN_KEYS")','(diagnostic ? "1" : nullptr)')
floatkey=block(main,'static uint32_t q8_top1_float_ordered_key(')
pack=block(main,'static uint64_t q8_top1_pack_key(')
code=r'''
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
struct Key {uint64_t key=0;uint32_t flag=0;};
static uint32_t bits(float f){uint32_t u;memcpy(&u,&f,4);return u;}
static float value(uint32_t u){float f;memcpy(&f,&u,4);return f;}
#define __float_as_uint bits
static bool probe_zero(float f,bool ftz){uint32_t a=bits(f)&0x7fffffffu;return a==0||(ftz&&a<0x00800000u);}
static bool probe_finite(float f){return (bits(f)&0x7fffffffu)<0x7f800000u;}
static void atomicOr(uint32_t*p,uint32_t v){*p|=v;}
'''
code+='static uint32_t q8_top1_float_ordered_key(float v){'+floatkey+'}\n'
code+='static uint64_t q8_top1_pack_key(float v,uint32_t idx){'+pack+'}\n'
code+='static Key original(uint32_t i,uint32_t width,uint32_t prefix,uint32_t tail,uint32_t vocab,float input,bool ftz){Key result;uint32_t*invalid=&result.flag;'+old.replace('return;','return result;')+'return result;}\n'
code+='static Key candidate(uint64_t row,uint32_t prefix,uint32_t tail,uint64_t n_vocab,float value,bool ftz){Key result;uint32_t*invalid=&result.flag;'+fused+'return result;}\n'
code+='static bool mtp_native_key_range_disjoint(const void*a,uint64_t an,const void*b,uint64_t bn){'+range_body+'}\n'
code+='struct Buffer {void*ptr;uint64_t bytes;};\nstatic bool allow(Buffer*scratch,Buffer*x,Buffer*out,Buffer*ids,void*w,uint32_t vocab,bool diagnostic){'+gate+'return fuse_keys;}\n'
code+=r'''
int main(){uint64_t checks=0;uint32_t rng=7;
uint32_t special[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc00001,0xffc00001};
for(uint32_t vocab:{21000u,248320u,0xffffffffu})for(uint32_t prefix:{16384u,20000u})for(uint32_t tail:{1u,276u}){
 if(prefix+tail>vocab)continue;uint32_t width=prefix+tail;
 for(unsigned mode=0;mode<2;mode++)for(unsigned j=0;j<100000;j++){
  rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
  uint32_t u=j<sizeof(special)/4?special[j]:rng;
  uint32_t row=j%5==0?0:j%5==1?prefix-1:j%5==2?prefix:j%5==3?width-1:rng%width;
  Key a=original(row,width,prefix,tail,vocab,value(u),mode),b=candidate(row,prefix,tail,vocab,value(u),mode);
  uint32_t id=row<prefix?row:(uint32_t)((uint64_t)vocab-tail+(row-prefix));
  uint32_t mag=u&0x7fffffffu;
  uint32_t canonical=mag==0||(mode&&mag<0x800000u)?0:u;
  uint32_t rank=canonical>>31?UINT32_MAX-canonical:canonical+0x80000000u;
  uint64_t expected=(!id||row>=prefix)?UINT64_MAX-id:((uint64_t)rank<<32)+(UINT32_MAX-id);
  assert(a.key==expected&&b.key==expected&&a.flag==(mag>=0x7f800000u)&&b.flag==a.flag);++checks;
 }
}
Buffer scratch{(void*)0x10000000,0x100000},x{(void*)0x20000000,4096},out{(void*)0x30000000,65536},ids{(void*)0x40000000,65536};void*w=(void*)0x50000000;
assert(allow(&scratch,&x,&out,&ids,w,21000,false));assert(!allow(&scratch,&x,&out,&ids,w,21000,true));
for(Buffer*t:{&x,&out,&ids}){void*save=t->ptr;t->ptr=(void*)0x10000004;assert(!allow(&scratch,&x,&out,&ids,w,21000,false));t->ptr=save;}
assert(!allow(&scratch,&x,&out,&ids,(void*)0x10000004,21000,false));
assert(mtp_native_key_range_disjoint((void*)16,16,(void*)32,16));
assert(!mtp_native_key_range_disjoint((void*)(UINTPTR_MAX-15),32,(void*)32,16));
assert(!mtp_native_key_range_disjoint((void*)16,16,(void*)(UINTPTR_MAX-15),32));
assert(!mtp_native_key_range_disjoint(nullptr,16,(void*)32,16));
std::cout<<"PASS "<<checks<<" extracted key/flag comparisons in IEEE and FTZ modes; disjoint/diagnostic/alias/overflow gates\n";
}
'''
temporary=tempfile.TemporaryDirectory(prefix='mtp-key-oracle-')
out=Path(temporary.name)/'oracle.cpp';out.write_text(code)
subprocess.run(['c++','-O2','-std=c++17',str(out),'-o',str(out.with_suffix(''))],check=True)
subprocess.run([str(out.with_suffix(''))],check=True)
