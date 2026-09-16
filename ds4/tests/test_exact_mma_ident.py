"""Compile the composed hello-limits formatter with both padded probe reports."""
from pathlib import Path
import subprocess,tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda.cu').read_text()
start=s.index('{',s.index('extern "C" const char *ds4_gpu_hw_limits(void)'))
end=start+1;depth=1
while depth:
    depth+=(s[end]=='{')-(s[end]=='}');end+=1
body=s[start+1:end-1]
code=r'''
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using cudaDeviceAttr=int;
enum{cudaSuccess,cudaDevAttrMaxSharedMemoryPerBlockOptin,cudaDevAttrMultiProcessorCount,
cudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMinor,cudaDevAttrIntegrated,cudaDevAttrCooperativeLaunch};
static int mode,calls;
static int cudaGetDevice(int *p){*p=0;return mode==2?1:0;}
static int cudaGetLastError(){return 0;}
static int cudaDeviceGetAttribute(int *p,int,int){*p=mode==0?48:INT_MAX;return 0;}
static const char *hc_up_exact_tune(){
    static std::string report="hcUp["+std::string(186,'T')+"]";calls++;return report.c_str();
}
static const char *mtp_screen_mma_tune(){
    static std::string report="scrMma["+std::string(184,'S')+"]";calls++;return report.c_str();
}
static const char *ds4_gpu_qwen4exp_kernel_limits(){static std::string text(383,'K');return text.c_str();}
'''+ 'static const char *limits(){'+body+'}\n'+r'''
int main(int argc,char **argv){assert(argc==2);mode=argv[1][0]-'0';
    const std::string result=limits();assert(result==limits());
    if(mode==2){assert(result.empty()&&calls==0);}
    else{
        assert(calls==2&&result.size()<1024);
        auto first=result.find("hcUp[");assert(first!=std::string::npos);
        assert(result.find(']',first)!=std::string::npos);
        assert(result.find('K')>result.find(']',first));
        auto second=result.find("scrMma[");assert(second!=std::string::npos && second>first);
        assert(result.find(']',second)!=std::string::npos && result.find('K')>result.find(']',second));
        char ident[1280];std::string prefix(129,'I');
        int n=snprintf(ident,sizeof ident,"%s %s",prefix.c_str(),result.c_str());
        assert(n>0&&(size_t)n<sizeof ident);
        std::cout<<"mode="<<mode<<" limits="<<result.size()<<" ident="<<n<<" report-intact\n";
    }
}
'''
with tempfile.TemporaryDirectory(prefix='exact-mma-ident-') as d:
    cpp=Path(d)/'ident.cpp';exe=cpp.with_suffix('');cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    for mode in range(3):subprocess.run([str(exe),str(mode)],check=True)
print('PASS actual formatter: padded fields, intact tuning decision, resident budget, idempotence, unavailable device')
