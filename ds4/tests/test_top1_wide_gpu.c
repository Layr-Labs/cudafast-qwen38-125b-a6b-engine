/* Native same-binary and changed-input graph comparison. Compile/link off-box;
 * execute only on CUDA hardware. No weights or full model are required. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
enum { GUARD=64 };
static unsigned checks,replays;
static void need(int ok,const char *why){if(!ok){fprintf(stderr,"top1 wide: %s\n",why);exit(1);}}
static void trial(unsigned n,unsigned rows,unsigned scratch_mode) {
    size_t sb=(size_t)n*rows*4,ob=(size_t)rows*4;
    size_t workspace=(size_t)rows*((n+4095)/4096)*8;
    ds4_gpu_tensor *scores=ds4_gpu_tensor_alloc(sb+GUARD);
    ds4_gpu_tensor *out[2]={ds4_gpu_tensor_alloc(ob+GUARD),ds4_gpu_tensor_alloc(ob+GUARD)};
    ds4_gpu_tensor *parent=ds4_gpu_tensor_alloc(workspace+2*GUARD);
    ds4_gpu_tensor *scratch=ds4_gpu_tensor_view(parent,scratch_mode==2?4:0,
        scratch_mode==1?workspace-1:workspace+GUARD);
    need(scores&&out[0]&&out[1]&&parent&&scratch,"allocation");
    unsigned char *input=malloc(sb+GUARD),*input_back=malloc(sb+GUARD);
    unsigned char *poison=malloc(workspace+2*GUARD),*partial=malloc(workspace+2*GUARD);
    unsigned char *expected=malloc(ob+GUARD),*actual=malloc(ob+GUARD);
    need(input&&input_back&&poison&&partial&&expected&&actual,"host buffers");
    memset(poison,0x5a,workspace+2*GUARD);
    ds4_decode_graph_key keys[2];memset(keys,0,sizeof(keys));
    for(unsigned m=0;m<2;m++){keys[m].il=m;keys[m].cur_hc=scores;keys[m].after_attn_hc=out[m];}
    ds4_gpu_decode_graphs_invalidate();
#define RUN(M) need(ds4_gpu_indexer_top1_scratch_tensor(out[M],scores,scratch_mode==3?scores:scratch,n,rows),"reduction")
#define RESET(M) do { memset(actual,0xa5,ob+GUARD);need(ds4_gpu_tensor_write(out[M],0,actual,ob+GUARD),"output poison");need(ds4_gpu_tensor_write(parent,0,poison,workspace+2*GUARD),"scratch poison"); } while(0)
    for(unsigned step=0;step<7;step++) {
        memset(input,0xc7,sb+GUARD);float *x=(float*)input;
        for(unsigned t=0;t<rows;t++) {
            for(unsigned i=0;i<n;i++)x[(size_t)t*n+i]=step==2?NAN:step==3?-INFINITY:-1.f;
            if(step!=2&&step!=3){unsigned at=(n-1-t*19-step*31)%n;x[(size_t)t*n+at]=step==4?-0.f:INFINITY;if(step==5)x[(size_t)t*n+(at/2)]=INFINITY;}
        }
        need(ds4_gpu_tensor_write(scores,0,input,sb+GUARD),"scores upload");
        need(setenv("DS4_CUDA_NO_TOP1_WIDE","1",1)==0,"reference valve");RESET(0);RUN(0);
        need(ds4_gpu_tensor_read(out[0],0,expected,ob+GUARD),"reference output");
        for(unsigned m=0;m<2;m++) {
            need((m?unsetenv("DS4_CUDA_NO_TOP1_WIDE"):setenv("DS4_CUDA_NO_TOP1_WIDE","1",1))==0,"graph valve");RESET(m);
            int state=ds4_gpu_decode_graph_begin(&keys[m]);
            if(step>=2)need(state==1,"actual graph replay");
            if(state!=1){need(state==0||state==-1,"graph state");RUN(m);if(state==0)need(ds4_gpu_decode_graph_end(&keys[m])==0,"graph end");}else replays++;
            need(ds4_gpu_tensor_read(out[m],0,actual,ob+GUARD)&&!memcmp(actual,expected,ob+GUARD),"selected indices/guards");
            need(ds4_gpu_tensor_read(scores,0,input_back,sb+GUARD)&&!memcmp(input,input_back,sb+GUARD),"score mutation");
            need(ds4_gpu_tensor_read(parent,0,partial,workspace+2*GUARD),"scratch read");
            int active=m && !scratch_mode && n>=65536&&n<=1048576&&rows<=7;
            size_t first=active?workspace:0;
            need(!memcmp(partial+first,poison+first,workspace+2*GUARD-first),"scratch boundary/fallback");checks++;
        }
    }
#undef RUN
#undef RESET
    ds4_gpu_decode_graphs_invalidate();ds4_gpu_tensor_free(scratch);ds4_gpu_tensor_free(parent);
    ds4_gpu_tensor_free(out[0]);ds4_gpu_tensor_free(out[1]);ds4_gpu_tensor_free(scores);
    free(input);free(input_back);free(poison);free(partial);free(expected);free(actual);
}
int main(void) {
    need(setenv("DS4_CUDA_DECODE_GRAPHS","1",1)==0&&ds4_gpu_init(),"GPU init");
    unsigned widths[]={65535,65536,65537,248320,1048576,1048577};
    for(unsigned i=0;i<sizeof(widths)/sizeof(widths[0]);i++)trial(widths[i],2,0);
    trial(248320,1,0);trial(248320,7,0);trial(248320,8,0);
    for(unsigned mode=1;mode<4;mode++)trial(248320,2,mode);
    ds4_gpu_cleanup();printf("top1 wide native: %u buffer comparisons, %u actual graph replays PASS\n",checks,replays);return 0;
}
