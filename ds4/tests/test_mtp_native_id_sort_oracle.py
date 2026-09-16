"""Compile the production ID decode/load/store and dispatch gates on the host.

CUB is replaced only at its documented collective boundary by a scalar radix
ordering oracle. This checks the surrounding indexing, bit bounds and alias
guard; it does NOT execute CUB or establish GPU performance. The CUDA fixture
test_mtp_native_screen.c compares the actual four key/ID-sort combinations.
"""
from pathlib import Path
import re
import subprocess
import tempfile

repo = Path(__file__).resolve().parents[2]
source = (repo / 'ds4/ds4_cuda_mtp_native.cuh').read_text()

def body(needle):
    start = source.index('{', source.index(needle))
    end, depth = start + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start + 1:end - 1]

kernel = body('__global__ static void mtp_native_unpack_sort_ids(')
assert 'cub::BlockRadixSort<uint32_t, 256, 8, cub::NullType, 6>' in kernel
split = 'Sort(storage).SortBlockedToStriped(items, 0, id_bits);'
assert kernel.count(split) == 1
load, save = kernel.split(split)
load = re.sub(r'    using Sort = .*?;\n', '', load)
load = re.sub(r'    __shared__ typename Sort::TempStorage storage;\n', '', load)
load = load.replace('    uint32_t items[8];\n', '')
range_body = body('static bool mtp_native_key_range_disjoint(')
gate = source[source.index('    const bool block_ids ='):source.index('    if (block_ids)')]
gate = gate.replace('getenv("DS4_MTP_NO_BLOCK_ID_SORT")', '(disabled ? "1" : nullptr)')
bits = source[source.index('    int id_bits = 1;'):source.index('    const bool block_ids =')]
old = body('__global__ static void mtp_native_unpack_ids(')

code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
static constexpr uint32_t MTP_NATIVE_CAP=2048;
struct Dim {uint32_t x;};
struct Buffer {void *ptr;};
'''
code += 'static void load_items(const uint64_t *keys,uint32_t *items,Dim threadIdx){' + load + '}\n'
code += 'static void save_items(uint32_t *ids,uint32_t *items,Dim threadIdx){' + save + '}\n'
code += 'static int bit_count(uint32_t vocab){' + bits + 'return id_bits;}\n'
code += 'static void old_unpack(uint32_t *ids,const uint64_t *keys,Dim blockIdx,Dim blockDim,Dim threadIdx){' + old + '}\n'
code += 'static bool mtp_native_key_range_disjoint(const void*a,uint64_t an,const void*b,uint64_t bn){' + range_body + '}\n'
code += 'static bool allow(Buffer *ids,const uint64_t *key_out,bool disabled){' + gate + 'return block_ids;}\n'
code += r'''
int main(){
    std::mt19937 rng(817031);
    uint64_t checked=0;
    for(uint32_t vocab:{2049u,4096u,4097u,65536u,65537u,248320u,1048576u,
                        0x80000000u,0x80000001u,0xffffffffu}){
        int n=bit_count(vocab), oracle_bits=1;
        for(uint64_t max_id=(uint64_t)vocab-1;max_id>>oracle_bits;oracle_bits++){}
        assert(n==oracle_bits && n>0 && n<=32);
        for(int mode=0;mode<6;mode++){
            std::array<uint64_t,2050> keys;
            std::array<uint32_t,2050> output,old_output;
            output.fill(0xa5a5a5a5u);old_output=output;
            keys.front()=keys.back()=0x9c59c59c59c59c59ull;
            std::vector<uint32_t> expected;
            for(unsigned j=0;j<2048;j++){
                uint32_t id=mode==0?j:mode==1?vocab-1-j:
                    mode==2?j%3:mode==3?(j&1?vocab-1:0):
                    mode==4?(uint32_t)((uint64_t)j*(vocab-1)/2047):rng()%vocab;
                expected.push_back(id);
                // Deliberately varied score bits; only the low ID word matters.
                keys[j+1]=((uint64_t)rng()<<32)|(0xffffffffu-id);
            }
            const auto input_copy=keys;
            std::sort(expected.begin(),expected.end());
            std::array<std::array<uint32_t,8>,256> lane{};
            std::vector<uint32_t> collective;
            for(unsigned t=0;t<256;t++){
                load_items(keys.data()+1,lane[t].data(),{t});
                collective.insert(collective.end(),lane[t].begin(),lane[t].end());
            }
            const uint64_t mask=(uint64_t{1}<<n)-1;
            std::stable_sort(collective.begin(),collective.end(),[mask](auto a,auto b){return (a&mask)<(b&mask);});
            // Documented CUB blocked-input -> striped-output contract.
            for(unsigned t=0;t<256;t++){
                for(unsigned i=0;i<8;i++)lane[t][i]=collective[i*256+t];
                save_items(output.data()+1,lane[t].data(),{t});
                for(unsigned b=0;b<8;b++)old_unpack(old_output.data()+1,keys.data()+1,{b},{256},{t});
            }
            std::sort(old_output.begin()+1,old_output.end()-1);
            assert(keys==input_copy && output.front()==0xa5a5a5a5u && output.back()==0xa5a5a5a5u);
            assert(output==old_output);
            assert(std::equal(expected.begin(),expected.end(),output.begin()+1));
            checked++;
        }
    }
    const uint64_t *keys=(const uint64_t*)0x100000;
    Buffer ids{(void*)0x200000};
    assert(allow(&ids,keys,false));assert(!allow(&ids,keys,true));
    ids.ptr=(void*)0x104000;assert(allow(&ids,keys,false));
    ids.ptr=(void*)0x0fe000;assert(allow(&ids,keys,false));
    for(uintptr_t overlap:{0xff000u,0x100000u,0x102000u,0x103ffcu}){
        ids.ptr=(void*)overlap;assert(!allow(&ids,keys,false));
    }
    ids.ptr=nullptr;assert(!allow(&ids,keys,false));
    ids.ptr=(void*)(UINTPTR_MAX-4095);assert(!allow(&ids,keys,false));
    ids.ptr=(void*)0x200000;assert(!allow(&ids,(const uint64_t*)(UINTPTR_MAX-8191),false));
    std::cout<<"PASS "<<checked<<" extracted load/unpack/striped-store cases and 11 dispatch controls under UBSan; CUB collective is a scalar contract stub, not GPU execution\n";
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-id-sort-oracle-') as directory:
    cpp = Path(directory) / 'oracle.cpp'
    exe = cpp.with_suffix('')
    cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
