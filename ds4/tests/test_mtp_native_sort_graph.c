/* Full native-sort CUDA contract; requires built libds4qwen and a supported
 * GPU, no GGUF. Compile mtp_native_weight_read.cu with nvcc and link its object
 * with this C test, libds4qwen, libcudart and libm. Pass 1 for the real vocabulary
 * dimensions; run separately with DS4_CUDA_DECODE_GRAPHS=0 for eager fallback.
 * This compares the same public API with all optimization diagnostics and
 * retains the inherited adversarial case documenting approximate screening. */
#include "ds4_gpu.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#define DIM 2560u
#define CAP 16384u
static unsigned test_prefix=20000u,test_vocab=21000u;
#define PREFIX test_prefix
#define TAIL 276u
#define VOCAB test_vocab
#define WIDTH (PREFIX+TAIL)
#define ROW (80u*34u)
static void need(int ok,const char *s) {if(!ok){fprintf(stderr,"native screen: %s\n",s);exit(1);}}
static uint32_t seed=1234567;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
/* Toggle only this optimization's diagnostics; every arm uses the same
 * shipped public API and allocation. Mode zero retains original CUDA kernels. */
static int native_mode(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,
 const void *map,uint64_t bytes,uint64_t off,uint32_t dim,uint32_t vocab,
 uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x,uint32_t mode){
 unsetenv("DS4_MTP_NO_NATIVE_SORT_OPT");unsetenv("DS4_MTP_NO_NATIVE_SORT_GRAPH");unsetenv("DS4_MTP_NO_NATIVE_COMPACT_KEYS");
 if(mode==0)setenv("DS4_MTP_NO_NATIVE_SORT_OPT","1",1);
 if(mode==1)setenv("DS4_MTP_NO_NATIVE_SORT_GRAPH","1",1);
 if(mode==2)setenv("DS4_MTP_NO_NATIVE_COMPACT_KEYS","1",1);
 int rc=ds4_gpu_mtp_native_screen(out,ids,scratch,map,bytes,off,dim,vocab,prefix,tail,x);
 unsetenv("DS4_MTP_NO_NATIVE_SORT_OPT");unsetenv("DS4_MTP_NO_NATIVE_SORT_GRAPH");unsetenv("DS4_MTP_NO_NATIVE_COMPACT_KEYS");
 return rc;
}
static int native_control(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,
 const void *map,uint64_t bytes,uint64_t off,uint32_t dim,uint32_t vocab,
 uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x){
 return native_mode(out,ids,scratch,map,bytes,off,dim,vocab,prefix,tail,x,0);
}

extern uint64_t ds4_cuda_mtp_native_sort_graph_replays(void),ds4_cuda_mtp_native_sort_graph_captures(void);
extern int mtp_native_read_weights(void*,const void*,unsigned long long);
enum { PARENT=16384*4+64 };
static ds4_gpu_tensor *ao[3],*ai[3],*as[3];
static unsigned char expected_out[PARENT],expected_ids[PARENT],actual[PARENT],poison[PARENT];
static uint64_t checks;
static int checked_screen(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,
    const void *w,uint64_t bytes,uint64_t off,uint32_t dim,uint32_t vocab,uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x){
    memset(poison,0x5a,sizeof poison);
    need(ds4_gpu_tensor_write(out,0,poison,PARENT)&&ds4_gpu_tensor_write(ids,0,poison,PARENT),"oracle poison");
    int rc=native_control(out,ids,scratch,w,bytes,off,dim,vocab,prefix,tail,x);
    need(ds4_gpu_tensor_read(out,0,expected_out,PARENT)&&ds4_gpu_tensor_read(ids,0,expected_ids,PARENT),"oracle parents");
    for(unsigned m=1;m<=3;m++){
        need(ds4_gpu_tensor_write(ao[m-1],0,poison,PARENT)&&ds4_gpu_tensor_write(ai[m-1],0,poison,PARENT),"candidate poison");
        int r=native_mode(ao[m-1],ai[m-1],as[m-1],w,bytes,off,dim,vocab,prefix,tail,x,m);
        need(r==rc,"candidate return value");
        need(ds4_gpu_tensor_read(ao[m-1],0,actual,PARENT)&&!memcmp(actual,expected_out,PARENT),"complete refined logits mismatch");
        need(ds4_gpu_tensor_read(ai[m-1],0,actual,PARENT)&&!memcmp(actual,expected_ids,PARENT),"complete original IDs mismatch");checks+=2;
    }
    return rc;
}
static void timing(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,
    const void *w,uint64_t bytes,uint64_t off,uint32_t dim,uint32_t vocab,uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x){
    cudaEvent_t begin,end;need(cudaEventCreate(&begin)==cudaSuccess&&cudaEventCreate(&end)==cudaSuccess,"events");
    ds4_gpu_tensor *flush=ds4_gpu_tensor_alloc(128u<<20);need(flush!=NULL,"flush");
    for(unsigned cold=0;cold<2;cold++){
        double totals[4]={0};
        for(unsigned rep=0;rep<96;rep++)for(unsigned pass=0;pass<4;pass++){
            unsigned m=(rep+pass)%4;
            if(cold)need(ds4_gpu_tensor_fill_f32(flush,0,(128u<<20)/4),"flush fill");
            need(cudaEventRecord(begin,0)==cudaSuccess,"begin event");
            int rc=m?native_mode(ao[m-1],ai[m-1],as[m-1],w,bytes,off,dim,vocab,prefix,tail,x,m):
                     native_control(out,ids,scratch,w,bytes,off,dim,vocab,prefix,tail,x);
            need(rc==16384,"timed screen return");
            need(cudaEventRecord(end,0)==cudaSuccess&&cudaEventSynchronize(end)==cudaSuccess,"end event");
            float dt;need(cudaEventElapsedTime(&dt,begin,end)==cudaSuccess,"elapsed");if(rep>=16)totals[m]+=dt;
        }
        printf("NATIVE_SORT_GRAPH_TIME vocab=%u width=%u offset=%llu cold=%u",vocab,prefix+tail,(unsigned long long)off,cold);
        for(unsigned m=0;m<4;m++)printf(" m%u_us=%.5f",m,totals[m]*1000/80);puts("");fflush(stdout);
    }
    need(cudaEventDestroy(begin)==cudaSuccess&&cudaEventDestroy(end)==cudaSuccess,"event destroy");ds4_gpu_tensor_free(flush);
}

static void run_case(int adversarial, uint32_t offset) {
    const uint64_t bytes=offset+(uint64_t)VOCAB*ROW;
    unsigned char *w=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    need(w!=MAP_FAILED,"host weights");
    for(uint32_t row=0;row<VOCAB;row++) for(uint32_t b=0;b<80;b++) {
        unsigned char *p=w+offset+(uint64_t)row*ROW+b*34;
        p[0]=0;p[1]=(adversarial==2 && row==0)?0x7e:0x3c; /* half NaN or 1 */
        for(unsigned j=2;j<34;j++) p[j]=adversarial?0:(unsigned char)((int)(rnd()%15)-7);
        /* Last prefix row has negative screen, positive complete dot. All
         * other rows tie at zero, so its omission is inevitable and explicit. */
        if(adversarial && row==PREFIX-1) memset(p+2,b<40?255:2,32);
    }
    need(ds4_gpu_init(),"GPU init");need(ds4_gpu_set_model_map(w,bytes),"register weights");
    uint64_t scratch_bytes=0;uint32_t cap=0;
    need(ds4_gpu_mtp_native_screen_init(WIDTH,&scratch_bytes,&cap)==1 && cap==CAP,"scratch query");
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(DIM*4),*out=ds4_gpu_tensor_alloc(PARENT),
        *ids=ds4_gpu_tensor_alloc(PARENT),*scratch=ds4_gpu_tensor_alloc(scratch_bytes),
        *full=ds4_gpu_tensor_alloc(PREFIX*4),*tail=ds4_gpu_tensor_alloc(TAIL*4),
        *winner=ds4_gpu_tensor_alloc(4);
    need(x&&out&&ids&&scratch&&full&&tail&&winner,"GPU allocations");
    for(unsigned m=0;m<3;m++){
        ao[m]=ds4_gpu_tensor_alloc(PARENT);ai[m]=ds4_gpu_tensor_alloc(PARENT);
        as[m]=ds4_gpu_tensor_alloc(scratch_bytes);need(ao[m]&&ai[m]&&as[m],"candidate allocations");
    }
    const uint64_t before=ds4_cuda_mtp_native_sort_graph_replays();
    float activation[DIM],selected[CAP],reference[PREFIX],tail_ref[TAIL];uint32_t found[CAP];
    for(unsigned replay=0;replay<(adversarial?1u:24u);replay++) {
        if(replay==12u)ds4_gpu_decode_graphs_invalidate();
        for(unsigned i=0;i<DIM;i++) activation[i]=adversarial?1.0f:(int)(rnd()%201)*0.01f-1.0f;
        need(ds4_gpu_tensor_write(x,0,activation,sizeof activation),"current activation");
        if(adversarial==2) {
            need(checked_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==0,"nonfinite score fallback");
            goto cleanup;
        }
        need(checked_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==CAP,"screen active");
        need(ds4_gpu_tensor_read(ids,0,found,sizeof found),"IDs read");
        need(ds4_gpu_tensor_read(out,0,selected,sizeof selected),"refine read");
        need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full,w,bytes,offset,DIM,PREFIX,x,1),"ordinary prefix");
        need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tail,w,bytes,offset+(uint64_t)(VOCAB-TAIL)*ROW,DIM,TAIL,x,1),"ordinary tail");
        need(ds4_gpu_tensor_read(full,0,reference,sizeof reference)&&ds4_gpu_tensor_read(tail,0,tail_ref,sizeof tail_ref),"oracle read");
        need(found[0]==0,"mandatory zero");
        for(unsigned i=0;i<CAP;i++) {
            need(i==0||found[i]>found[i-1],"sorted unique IDs");
            need(found[i]<PREFIX || (found[i]>=VOCAB-TAIL && found[i]<VOCAB),"static domain");
            float exact=found[i]<PREFIX?reference[found[i]]:tail_ref[found[i]-(VOCAB-TAIL)];
            need(!memcmp(&exact,&selected[i],4),"full refinement differs from ordinary row");
        }
        float unchanged_x[DIM];need(ds4_gpu_tensor_read(x,0,unchanged_x,sizeof unchanged_x)&&!memcmp(unchanged_x,activation,sizeof activation),"float activation changed");
        for(unsigned i=0;i<TAIL;i++) need(found[CAP-TAIL+i]==VOCAB-TAIL+i,"mandatory tail");
        if(adversarial) {
            need(reference[PREFIX-1]>0,"adversarial full winner");
            for(unsigned i=0;i<CAP;i++) need(found[i]!=PREFIX-1 && selected[i]==0,"screen is approximate");
            for(unsigned i=1;i<CAP-TAIL;i++) need(found[i]==i,"coarse tie lowest ID");
        }
    }
    if(!adversarial){
        if(!getenv("DS4_CUDA_DECODE_GRAPHS") || strcmp(getenv("DS4_CUDA_DECODE_GRAPHS"),"0"))
            need(ds4_cuda_mtp_native_sort_graph_replays()>=before+40,"changed activation graph replays");
        timing(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x);
    }
    /* Map original IDs, reject bad packed IDs, and preserve legacy NaN0. */
    uint32_t packed=CAP-1,mapped=0;
    need(ds4_gpu_tensor_write(winner,0,&packed,4)&&ds4_gpu_mtp_native_map(winner,out,ids,CAP,VOCAB)&&ds4_gpu_tensor_read(winner,0,&mapped,4),"winner map");
    need(mapped==VOCAB-1,"mapped tail ID");
    packed=CAP;
    need(ds4_gpu_tensor_write(winner,0,&packed,4)&&ds4_gpu_mtp_native_map(winner,out,ids,CAP,VOCAB)&&ds4_gpu_tensor_read(winner,0,&mapped,4)&&mapped==UINT32_MAX,"bad winner guard");
    float nan=NAN;
    need(ds4_gpu_tensor_write(out,0,&nan,4)&&ds4_gpu_mtp_native_map(winner,out,ids,CAP,VOCAB)&&ds4_gpu_tensor_read(winner,0,&mapped,4)&&mapped==0,"NaN0 pin");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,100,TAIL,x)==0,"small width fallback");
    setenv("DS4_QWEN4EXP_NO_ROW_TILE","1",1);
    need(checked_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==0,"diagnostic fallback");
    unsetenv("DS4_QWEN4EXP_NO_ROW_TILE");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes-1,offset,DIM,VOCAB,PREFIX,TAIL,x)==-1,"truncated map");
    ds4_gpu_tensor *tiny=ds4_gpu_tensor_alloc(4);
    need(ds4_gpu_mtp_native_screen(out,ids,tiny,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==-1,"scratch bound");
    ds4_gpu_tensor_free(tiny);
    if(ds4_gpu_decode_graphs_supported()){
    ds4_gpu_decode_graphs_invalidate();
    ds4_decode_graph_key key={.il=3,.island=0,.variant=1};
    need(ds4_gpu_decode_graph_begin(&key)==-1,"graph warmup");
    need(ds4_gpu_decode_graph_begin(&key)==0,"capture start");
    need(native_mode(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,3)==0,"capture declines screening");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full,w,bytes,offset,DIM,PREFIX,x,1),"captured static fallback");
    need(ds4_gpu_decode_graph_end(&key)==0,"capture end");
    ds4_gpu_decode_graphs_invalidate();
    }
cleanup:
    ds4_gpu_decode_graphs_invalidate();
    for(unsigned m=0;m<3;m++){ds4_gpu_tensor_free(ao[m]);ds4_gpu_tensor_free(ai[m]);ds4_gpu_tensor_free(as[m]);}
    unsigned char *verify=malloc(bytes);need(verify!=NULL,"weight readback host");
    need(mtp_native_read_weights(verify,w,bytes)&&!memcmp(verify,w,bytes),"resolved GPU weights changed");free(verify);
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(ids);ds4_gpu_tensor_free(scratch);
    ds4_gpu_tensor_free(full);ds4_gpu_tensor_free(tail);ds4_gpu_tensor_free(winner);
    ds4_gpu_cleanup();munmap(w,bytes);
}
int main(int argc,char **argv){
 if(argc>1 && atoi(argv[1])){test_prefix=98308;test_vocab=248320;}
 if(!getenv("DS4_CUDA_DECODE_GRAPHS"))setenv("DS4_CUDA_DECODE_GRAPHS","1",1);
 run_case(0,0);run_case(0,2);run_case(1,0);run_case(2,0);
 printf("NATIVE_SORT_GRAPH_PASS comparisons=%llu captures=%llu replays=%llu\n",(unsigned long long)checks,(unsigned long long)ds4_cuda_mtp_native_sort_graph_captures(),(unsigned long long)ds4_cuda_mtp_native_sort_graph_replays());return 0;
}
