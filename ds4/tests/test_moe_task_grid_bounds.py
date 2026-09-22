"""Exercise the production grid bound against the production task classifier."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

text = (Path(__file__).resolve().parents[1] / "ds4_cuda_qwen4exp.cu").read_text()
mark = "static uint32_t qwen4exp_moe_task_grid_limit("
start = text.index(mark)
end = text.index("\n}", start) + 2
helper = text[start:end]
task = text[text.index("__global__ static void qwen4exp_moe_pair_tasks_kernel("):]
classifier = re.search(r"const int32_t count = .*?;\n    const int32_t tiles = .*?;", task).group()
assert "(c0 > lo && c0 <= hi)" in classifier
assert "QW_MMA_BN == 32u" in text and "QW_GUH_BN == 64u" in text
assert 'getenv("DS4_QWEN4EXP_NO_MOE_TASK_GRID_BOUND") == NULL' in text
assert "TASKS ? light_task_grid : gu_rows" in text
assert "dim3(mid_dim / QW_GUH_BM, heavy_task_grid, 1)" in text

source = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
""" + helper + "\nstatic int tasks(int32_t c0,int32_t tile,int32_t lo,int32_t hi) {\n" + classifier + r"""
return tiles;
}
static uint64_t checks=0;
static void check(const std::vector<uint32_t>&c, uint32_t extra_pairs=0) {
    uint64_t pairs=extra_pairs,light=0,heavy=0;
    for(uint32_t x:c) {
        pairs+=x;
        light+=tasks(x,32,0,32);
        heavy+=tasks(x,64,32,0x7fffffff);
    }
    uint64_t capacity=pairs/32+std::min<uint64_t>(pairs,c.size());
    if(capacity>65535 || pairs>0x7fffffe0u) return; // original pair_tasks eligibility
    auto l=qwen4exp_moe_task_grid_limit(capacity,pairs,c.size(),false);
    auto h=qwen4exp_moe_task_grid_limit(capacity,pairs,c.size(),true);
    assert(light<=l && heavy<=h && l<=capacity && h<=capacity);
    // Legacy grid, used when heavy dispatch or the new diagnostic is disabled,
    // covers both classified lists as well as the unclassified 32-pair list.
    uint64_t full=0;for(auto x:c)full+=tasks(x,32,0,0x7fffffff);
    assert(full<=capacity);
    checks++;
}
static void partitions(std::vector<uint32_t>&c,unsigned pos,unsigned low,unsigned left) {
    if(pos==c.size()){check(c);return;}
    for(unsigned x=low;x<=left/(c.size()-pos);x++){
        c[pos]=x;partitions(c,pos+1,x,left-x);
    }
}
int main() {
    // Every nonnegative count partition with up to four experts, sum <= 128.
    for(unsigned e=1;e<=4;e++){std::vector<uint32_t>c(e);partitions(c,0,0,128);}
    // Per-expert inequality over every count that can reach the bounded grid.
    for(uint32_t c=33;c<=2097119;c++)
        assert(uint32_t(tasks(c,64,32,0x7fffffff))<=c/32);
    const uint32_t edge[]={0,1,31,32,33,63,64,65,95,96,97,127,128,129,255,256,257,10240};
    for(uint32_t a:edge)for(uint32_t b:edge)for(uint32_t c:edge)check({a,b,c});
    std::mt19937 rng(0x13370922);
    for(unsigned iter=0;iter<30000;iter++){
        unsigned e=1+rng()%512;
        // Include boundary of the original grid capacity check.
        unsigned total=(iter%3==0)?rng()%2097120:rng()%20000;
        std::vector<uint32_t>c(e);
        unsigned left=total;
        for(unsigned j=0;j<e;j++){
            unsigned take=(j+1==e)?left:rng()%(left+1);
            c[j]=take;left-=take;
        }
        std::shuffle(c.begin(),c.end(),rng);check(c,rng()%65);
    }
    for(unsigned e:{1u,32u,512u}){
        check(std::vector<uint32_t>(e,0));check(std::vector<uint32_t>(e,32));
        check(std::vector<uint32_t>(e,33));check(std::vector<uint32_t>(e,64));
        check(std::vector<uint32_t>(e,65));
    }
    assert(qwen4exp_moe_task_grid_limit(832,10240,512,false)==512);
    assert(qwen4exp_moe_task_grid_limit(832,10240,512,true)==320);
    // Helper does no add/multiply: no overflow even outside dispatched domain.
    assert(qwen4exp_moe_task_grid_limit(UINT32_MAX,UINT32_MAX,UINT32_MAX,true)==UINT32_MAX/32);
    printf("PASS task bounds: %llu partitions/random totals; 2,097,087 heavy counts; boundaries and inactive/legacy coverage\n",
           (unsigned long long)checks);
}
"""
with tempfile.TemporaryDirectory(prefix="gdn-moe-taskbounds-") as tmp:
    src = Path(tmp) / "test.cpp"
    exe = Path(tmp) / "test"
    src.write_text(source)
    subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17", str(src), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)

