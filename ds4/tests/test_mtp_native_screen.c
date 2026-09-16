/* Synthetic CUDA contract; needs built libds4qwen and a supported GPU, no GGUF.
 * cc -O2 -std=c11 -D_GNU_SOURCE -Ids4 ds4/tests/test_mtp_native_screen.c \
 *    -L.build/ds4 -lds4qwen -lm -o /tmp/test-mtp-native-screen
 * LD_LIBRARY_PATH=.build/ds4 /tmp/test-mtp-native-screen
 * Selection is APPROXIMATE: the adversarial case intentionally loses the full
 * native winner. Selected-row scores must nevertheless be bitwise exact. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#define DIM 2560u
#define CAP 2048u
#define PREFIX 20000u
#define TAIL 276u
#define VOCAB 21000u
#define WIDTH (PREFIX+TAIL)
#define ROW (80u*34u)
static void need(int ok,const char *s) {if(!ok){fprintf(stderr,"native screen: %s\n",s);exit(1);}}
static uint32_t seed=1234567;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
/* Compare actual old/fused implementations, including raw pre-sort keys.
 * Scratch scores are deliberately omitted by fusion and are not compared. */
static uint64_t aligned(uint64_t n){return (n+255u)&~255ull;}
static uint64_t test_flag_offset(void) {
    uint64_t at=aligned(DIM+80u*4u);
    at=aligned(at+(uint64_t)WIDTH*4u);
    at=aligned(at+(uint64_t)WIDTH*8u);
    at=aligned(at+(uint64_t)WIDTH*8u);
    return aligned(at+(uint64_t)CAP*4u);
}
static int compare_key_paths(ds4_gpu_tensor *winner,ds4_gpu_tensor *out,ds4_gpu_tensor *ids,
        ds4_gpu_tensor *scratch,const void *w,uint64_t bytes,uint64_t offset,
        ds4_gpu_tensor *x) {
    uint64_t scores_at=aligned(DIM+80u*4u);
    uint64_t ki=aligned(scores_at+(uint64_t)WIDTH*4u);
    uint64_t ko=aligned(ki+(uint64_t)WIDTH*8u);
    uint64_t it=aligned(ko+(uint64_t)WIDTH*8u);
    uint64_t flag_at=aligned(it+(uint64_t)CAP*4u);
    uint64_t *keys[2]={malloc(WIDTH*8u),malloc(WIDTH*8u)};
    uint32_t *selected_ids[2]={malloc(CAP*4u),malloc(CAP*4u)};
    float *values[2]={malloc(CAP*4u),malloc(CAP*4u)};
    unsigned char *score_canary=malloc(WIDTH*4u),*score_after=malloc(WIDTH*4u);
    need(score_canary&&score_after,"score witness allocation");
    memset(score_canary,0xa5,WIDTH*4u);
    float before[DIM],after[DIM];uint32_t flags[2],winners[2];int status[2];
    need(keys[0]&&keys[1]&&selected_ids[0]&&selected_ids[1]&&values[0]&&values[1],"AB host allocation");
    need(ds4_gpu_tensor_read(x,0,before,sizeof before),"AB input before");
    for(unsigned mode=0;mode<2;mode++) {
        if(mode==0)setenv("DS4_MTP_NO_FUSED_SCREEN_KEYS","1",1);
        else unsetenv("DS4_MTP_NO_FUSED_SCREEN_KEYS");
        memset(values[mode],0x5a,CAP*4u);memset(selected_ids[mode],0xa5,CAP*4u);
        need(ds4_gpu_tensor_write(out,0,values[mode],CAP*4u)&&ds4_gpu_tensor_write(ids,0,selected_ids[mode],CAP*4u),"AB canary init");
        need(ds4_gpu_tensor_write(scratch,scores_at,score_canary,WIDTH*4u),"score witness init");
        uint64_t pending_flag=UINT64_MAX;
        status[mode]=winner
            ? ds4_gpu_mtp_native_propose_async(winner,out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,&pending_flag)
            : ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x);
        if(winner)need(status[mode]==CAP&&pending_flag==flag_at&&
            ds4_gpu_tensor_read(winner,0,&winners[mode],4),"AB pending winner/flag location");
        need(status[mode]>=0,"AB backend success");
        need(ds4_gpu_tensor_read(scratch,scores_at,score_after,WIDTH*4u),"score witness read");
        need(mode ? !memcmp(score_canary,score_after,WIDTH*4u) : memcmp(score_canary,score_after,WIDTH*4u)!=0,"actual fused/old dispatch witness");
        need(ds4_gpu_tensor_read(scratch,ki,keys[mode],WIDTH*8u)&&ds4_gpu_tensor_read(scratch,flag_at,&flags[mode],4),"AB keys/flag");
        need(ds4_gpu_tensor_read(out,0,values[mode],CAP*4u)&&ds4_gpu_tensor_read(ids,0,selected_ids[mode],CAP*4u),"AB outputs");
        need(ds4_gpu_tensor_read(x,0,after,sizeof after)&&!memcmp(before,after,sizeof before),"AB input unchanged");
    }
    need(status[0]==status[1]&&flags[0]==flags[1],"AB status/flag parity");
    if(winner)need(winners[0]==winners[1],"AB original winner parity");
    need(!memcmp(keys[0],keys[1],WIDTH*8u),"AB raw key parity");
    need(!memcmp(values[0],values[1],CAP*4u)&&!memcmp(selected_ids[0],selected_ids[1],CAP*4u),"AB IDs/refinement or fallback canary parity");
    int result=status[1];for(unsigned i=0;i<2;i++){free(keys[i]);free(selected_ids[i]);free(values[i]);}
    free(score_canary);free(score_after);return result;
}
static void run_case(int adversarial, uint32_t offset) {
    const uint64_t bytes=offset+(uint64_t)VOCAB*ROW;
    unsigned char *w=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    need(w!=MAP_FAILED,"host weights");
    for(uint32_t row=0;row<VOCAB;row++) for(uint32_t b=0;b<80;b++) {
        unsigned char *p=w+offset+(uint64_t)row*ROW+b*34;
        p[0]=0;p[1]=(adversarial>=2 && adversarial<=4 && row==0)?(adversarial==2?0x7e:adversarial==3?0x7c:0xfc):0x3c; /* half NaN or 1 */
        for(unsigned j=2;j<34;j++) p[j]=adversarial?0:(unsigned char)((int)(rnd()%15)-7);
        /* Last prefix row has negative screen, positive complete dot. All
         * other rows tie at zero, so its omission is inevitable and explicit. */
        if((adversarial==5&&row==0&&b>=40)||(adversarial==6&&row==1&&b>=40))p[1]=0x7e;
        if(adversarial>=2 && adversarial<=4 && row==0) memset(p+2,1,32);
        if(adversarial && row==PREFIX-1) memset(p+2,b<40?255:2,32);
        /* Preserve promoted 24-versus-32-group policy witness. */
        if(adversarial==7 && row==PREFIX-1) memset(p+2,b<24?255:b<32?8:2,32);
    }
    need(ds4_gpu_init(),"GPU init");need(ds4_gpu_set_model_map(w,bytes),"register weights");
    uint64_t scratch_bytes=0;uint32_t cap=0;
    need(ds4_gpu_mtp_native_screen_init(WIDTH,&scratch_bytes,&cap)==1 && cap==CAP,"scratch query");
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(DIM*4),*out=ds4_gpu_tensor_alloc(WIDTH*4),
        *ids=ds4_gpu_tensor_alloc(CAP*4),*scratch=ds4_gpu_tensor_alloc(scratch_bytes),
        *full=ds4_gpu_tensor_alloc(PREFIX*4),*tail=ds4_gpu_tensor_alloc(TAIL*4),
        *winner=ds4_gpu_tensor_alloc(4);
    need(x&&out&&ids&&scratch&&full&&tail&&winner,"GPU allocations");
    float activation[DIM],selected[CAP],reference[PREFIX],tail_ref[TAIL];uint32_t found[CAP];
    for(unsigned replay=0;replay<(adversarial?1u:3u);replay++) {
        for(unsigned i=0;i<DIM;i++) activation[i]=adversarial?1.0f:(int)(rnd()%201)*0.01f-1.0f;
        need(ds4_gpu_tensor_write(x,0,activation,sizeof activation),"current activation");
        if(getenv("DS4_CUDA_NO_TOP1")) {
            uint64_t flag_off;
            need(ds4_gpu_mtp_native_propose_async(winner,out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,&flag_off)==0,"serial top1 diagnostic keeps old API");
            need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==CAP,"old screen remains available");
            goto cleanup;
        }
        if(adversarial>=2 && adversarial<=4) {
            need(compare_key_paths(NULL,out,ids,scratch,w,bytes,offset,x)==0,"nonfinite score fallback");
            uint64_t flag_off=UINT64_MAX;uint32_t id=0,flag=0;
            need(compare_key_paths(winner,out,ids,scratch,w,bytes,offset,x)==CAP,"async invalid queues");
            flag_off=test_flag_offset();
            need(ds4_gpu_tensor_read(winner,0,&id,4)&&id==UINT32_MAX,"async invalid sentinel");
            need(flag_off<=scratch_bytes-4&&ds4_gpu_tensor_read(scratch,flag_off,&flag,4)&&flag,"cold reason remains live");
            need(ds4_gpu_tensor_read(ids,0,found,sizeof found),"invalid IDs");
            unsigned char seen[VOCAB]={0};
            for(unsigned i=0;i<CAP;i++) {
                need(found[i]<VOCAB&&!seen[found[i]],"invalid-case IDs bounded and unique");seen[found[i]]=1;
            }
            float unchanged[DIM];need(ds4_gpu_tensor_read(x,0,unchanged,sizeof unchanged)&&!memcmp(activation,unchanged,sizeof unchanged),"fallback activation preserved");
            need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full,w,bytes,offset,DIM,PREFIX,x,1),"static oracle prefix");
            need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tail,w,bytes,offset+(uint64_t)(VOCAB-TAIL)*ROW,DIM,TAIL,x,1),"static oracle tail");
            need(ds4_gpu_tensor_read(full,0,reference,sizeof reference)&&ds4_gpu_tensor_read(tail,0,tail_ref,sizeof tail_ref),"static oracle read");
            need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out,w,bytes,offset,DIM,PREFIX,x,1)&&ds4_gpu_tensor_copy(out,PREFIX*4ull,tail,0,TAIL*4ull),"static overwrites discarded output");
            float packed[WIDTH];need(ds4_gpu_tensor_read(out,0,packed,sizeof packed)&&!memcmp(packed,reference,sizeof reference)&&!memcmp(packed+PREFIX,tail_ref,sizeof tail_ref),"complete static overwrite parity");
            goto cleanup;
        }
        need(compare_key_paths(NULL,out,ids,scratch,w,bytes,offset,x)==CAP,"screen active");
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
        for(unsigned i=0;i<TAIL;i++) need(found[CAP-TAIL+i]==VOCAB-TAIL+i,"mandatory tail");
        uint32_t old_id,new_id;uint64_t flag_off=UINT64_MAX;
        need(ds4_gpu_indexer_topk_tensor(winner,out,CAP,1,1)&&ds4_gpu_mtp_native_map(winner,out,ids,CAP,VOCAB)&&ds4_gpu_tensor_read(winner,0,&old_id,4),"old complete proposal");
        need(compare_key_paths(winner,out,ids,scratch,w,bytes,offset,x)==CAP,"async proposal active");
        need(ds4_gpu_tensor_read(winner,0,&new_id,4)&&new_id==old_id,"async original proposal parity");
        uint32_t async_ids[CAP];float async_logits[CAP];
        need(ds4_gpu_tensor_read(ids,0,async_ids,sizeof async_ids),"async IDs");
        need(ds4_gpu_tensor_read(out,0,async_logits,sizeof async_logits),"async full rows");
        unsigned char membership[VOCAB]={0},seen[VOCAB]={0};
        for(unsigned i=0;i<CAP;i++)membership[found[i]]=1;
        for(unsigned i=0;i<CAP;i++) {
            uint32_t id=async_ids[i];
            need(id<VOCAB&&membership[id]&&!seen[id],"same unique selected set by original ID");seen[id]=1;
            float exact=id<PREFIX?reference[id]:tail_ref[id-(VOCAB-TAIL)];
            need(!memcmp(&exact,&async_logits[i],4),"same refined score by original ID");
        }
        need(async_ids[0]==0&&async_ids[1]>=VOCAB-TAIL&&async_ids[TAIL+1]<PREFIX,"async score-order path active");
        if(adversarial==5)need(new_id==0,"NaN at full token-zero score pins zero");
        if(adversarial==1 || adversarial==7) {
            need(reference[PREFIX-1]>0,"adversarial full winner");
            for(unsigned i=0;i<CAP;i++) need(found[i]!=PREFIX-1 && selected[i]==0,"screen is approximate");
            for(unsigned i=1;i<CAP-TAIL;i++) need(found[i]==i,"coarse tie lowest ID");
        }
    }
    /* The generic API still supplies ascending IDs for its compact mapper. */
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==CAP,"restore old sorted API");
    /* Map original IDs, reject bad packed IDs, and preserve legacy NaN0. */
    float finite_zero=0;
    need(ds4_gpu_tensor_write(out,0,&finite_zero,4),"finite setup for independent mapper checks");
    uint32_t packed=CAP-1,mapped=0;
    need(ds4_gpu_tensor_write(winner,0,&packed,4)&&ds4_gpu_mtp_native_map(winner,out,ids,CAP,VOCAB)&&ds4_gpu_tensor_read(winner,0,&mapped,4),"winner map");
    need(mapped==VOCAB-1,"mapped tail ID");
    packed=CAP;
    need(ds4_gpu_tensor_write(winner,0,&packed,4)&&ds4_gpu_mtp_native_map(winner,out,ids,CAP,VOCAB)&&ds4_gpu_tensor_read(winner,0,&mapped,4)&&mapped==UINT32_MAX,"bad winner guard");
    float nan=NAN;
    need(ds4_gpu_tensor_write(out,0,&nan,4)&&ds4_gpu_mtp_native_map(winner,out,ids,CAP,VOCAB)&&ds4_gpu_tensor_read(winner,0,&mapped,4)&&mapped==0,"NaN0 pin");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,100,TAIL,x)==0,"small width fallback");
    setenv("DS4_QWEN4EXP_NO_ROW_TILE","1",1);
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==0,"diagnostic fallback");
    unsetenv("DS4_QWEN4EXP_NO_ROW_TILE");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes-1,offset,DIM,VOCAB,PREFIX,TAIL,x)==-1,"truncated map");
    ds4_gpu_tensor *tiny=ds4_gpu_tensor_alloc(4);
    need(ds4_gpu_mtp_native_screen(out,ids,tiny,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==-1,"scratch bound");
    ds4_gpu_tensor_free(tiny);
    uint64_t flag_off;
    ds4_gpu_tensor *alias=ds4_gpu_tensor_view(out,0,DIM*4ull);
    need(alias&&ds4_gpu_tensor_write(alias,0,activation,sizeof activation),"alias activation");
    need(ds4_gpu_mtp_native_propose_async(winner,out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,alias,&flag_off)==0,"async output/input alias declines");
    float unchanged[DIM];need(ds4_gpu_tensor_read(alias,0,unchanged,sizeof unchanged)&&!memcmp(unchanged,activation,sizeof activation),"alias decline before writes");ds4_gpu_tensor_free(alias);
    alias=ds4_gpu_tensor_view(ids,0,4);
    need(alias&&ds4_gpu_mtp_native_propose_async(alias,out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,&flag_off)==0,"winner/ID alias declines");ds4_gpu_tensor_free(alias);
    alias=ds4_gpu_tensor_view(scratch,0,CAP*4ull);
    need(alias&&ds4_gpu_mtp_native_propose_async(winner,alias,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,&flag_off)==0,"output/scratch alias declines");ds4_gpu_tensor_free(alias);
    tiny=ds4_gpu_tensor_alloc(1);
    need(ds4_gpu_mtp_native_propose_async(tiny,out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,&flag_off)==-1,"winner size guard");ds4_gpu_tensor_free(tiny);
    ds4_gpu_decode_graphs_invalidate();
    ds4_decode_graph_key key={.il=3,.island=0,.variant=1};
    need(ds4_gpu_decode_graph_begin(&key)==-1,"graph warmup");
    need(ds4_gpu_decode_graph_begin(&key)==0,"capture start");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==0,"capture declines screening");
    need(ds4_gpu_mtp_native_propose_async(winner,out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,&flag_off)==0,"capture declines async");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full,w,bytes,offset,DIM,PREFIX,x,1),"captured static fallback");
    need(ds4_gpu_decode_graph_end(&key)==0,"capture end");
    ds4_gpu_decode_graphs_invalidate();
cleanup:
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(ids);ds4_gpu_tensor_free(scratch);
    ds4_gpu_tensor_free(full);ds4_gpu_tensor_free(tail);ds4_gpu_tensor_free(winner);
    ds4_gpu_cleanup();munmap(w,bytes);
}
int main(int argc,char **argv){(void)argv;if(argc>1){setenv("DS4_CUDA_NO_TOP1","1",1);run_case(0,0);return 0;}setenv("DS4_CUDA_DECODE_GRAPHS","1",1);run_case(0,0);run_case(0,2);run_case(1,0);run_case(2,0);run_case(3,0);run_case(4,0);run_case(5,0);run_case(6,0);run_case(7,0);run_case(7,2);puts("native screen contracts pass (selection deliberately approximate)");return 0;}
