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
#include <float.h>
#include <sys/mman.h>
#define DIM 2560u
#define CAP 2048u
#define CAP2 16384u
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
static int compare_key_paths(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,
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
    float before[DIM],after[DIM];uint32_t flags[2];int status[2];
    need(keys[0]&&keys[1]&&selected_ids[0]&&selected_ids[1]&&values[0]&&values[1],"AB host allocation");
    need(ds4_gpu_tensor_read(x,0,before,sizeof before),"AB input before");
    for(unsigned mode=0;mode<2;mode++) {
        if(mode==0)setenv("DS4_MTP_NO_FUSED_SCREEN_KEYS","1",1);
        else unsetenv("DS4_MTP_NO_FUSED_SCREEN_KEYS");
        memset(values[mode],0x5a,CAP*4u);memset(selected_ids[mode],0xa5,CAP*4u);
        need(ds4_gpu_tensor_write(out,0,values[mode],CAP*4u)&&ds4_gpu_tensor_write(ids,0,selected_ids[mode],CAP*4u),"AB canary init");
        need(ds4_gpu_tensor_write(scratch,scores_at,score_canary,WIDTH*4u),"score witness init");
        status[mode]=ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,0);
        need(status[mode]>=0,"AB backend success");
        need(ds4_gpu_tensor_read(scratch,scores_at,score_after,WIDTH*4u),"score witness read");
        need(mode ? !memcmp(score_canary,score_after,WIDTH*4u) : memcmp(score_canary,score_after,WIDTH*4u)!=0,"actual fused/old dispatch witness");
        need(ds4_gpu_tensor_read(scratch,ki,keys[mode],WIDTH*8u)&&ds4_gpu_tensor_read(scratch,flag_at,&flags[mode],4),"AB keys/flag");
        need(ds4_gpu_tensor_read(out,0,values[mode],CAP*4u)&&ds4_gpu_tensor_read(ids,0,selected_ids[mode],CAP*4u),"AB outputs");
        need(ds4_gpu_tensor_read(x,0,after,sizeof after)&&!memcmp(before,after,sizeof before),"AB input unchanged");
    }
    need(status[0]==status[1]&&flags[0]==flags[1],"AB status/flag parity");
    need(!memcmp(keys[0],keys[1],WIDTH*8u),"AB raw key parity");
    need(!memcmp(values[0],values[1],CAP*4u)&&!memcmp(selected_ids[0],selected_ids[1],CAP*4u),"AB IDs/refinement or fallback canary parity");
    int result=status[1];for(unsigned i=0;i<2;i++){free(keys[i]);free(selected_ids[i]);free(values[i]);}
    free(score_canary);free(score_after);return result;
}
static void compare_r2_paths(const void *w,uint64_t bytes,uint64_t offset,
        const float *activation) {
    uint64_t scratch2_bytes=0;uint32_t cap2=0;
    need(ds4_gpu_mtp_native_screen2_init(WIDTH,&scratch2_bytes,&cap2)==1&&cap2==CAP2,
         "R2 scratch query");
    ds4_gpu_tensor *x2=ds4_gpu_tensor_alloc(2ull*DIM*4u),
        *out2=ds4_gpu_tensor_alloc(2ull*CAP2*4u),
        *ids2=ds4_gpu_tensor_alloc(2ull*CAP2*4u),
        *scratch2=ds4_gpu_tensor_alloc(scratch2_bytes),
        *scattered=ds4_gpu_tensor_alloc(2ull*VOCAB*4u),
        *full2=ds4_gpu_tensor_alloc(2ull*PREFIX*4u),
        *tail2=ds4_gpu_tensor_alloc(2ull*TAIL*4u),
        *winner2=ds4_gpu_tensor_alloc(3u*4u);
    need(x2&&out2&&ids2&&scratch2&&scattered&&full2&&tail2&&winner2,
         "R2 GPU allocations");
    float *a2=malloc(2ull*DIM*4u),*after=malloc(2ull*DIM*4u),
        *values[2]={malloc(2ull*CAP2*4u),malloc(2ull*CAP2*4u)},
        *dense=malloc(2ull*VOCAB*4u),*reference=malloc(2ull*PREFIX*4u),
        *tail_ref=malloc(2ull*TAIL*4u);
    uint32_t *selected_ids[2]={malloc(2ull*CAP2*4u),malloc(2ull*CAP2*4u)};
    need(a2&&after&&values[0]&&values[1]&&dense&&reference&&tail_ref&&
         selected_ids[0]&&selected_ids[1],"R2 host allocations");
    memcpy(a2,activation,DIM*4u);
    for(uint32_t i=0;i<DIM;i++) a2[DIM+i]=activation[DIM-1u-i]*0.75f+0.125f;
    need(ds4_gpu_tensor_write(x2,0,a2,2ull*DIM*4u),"R2 activations");
    unsetenv("DS4_QWEN4EXP_NO_TARGET_NATIVE_SCREEN_R2");
    for(uint32_t mode=0;mode<2;mode++) {
        if(mode==0)setenv("DS4_MTP_NO_FUSED_SCREEN_KEYS","1",1);
        else unsetenv("DS4_MTP_NO_FUSED_SCREEN_KEYS");
        need(ds4_gpu_mtp_native_screen2(out2,ids2,scratch2,w,bytes,offset,
             DIM,VOCAB,PREFIX,TAIL,x2,0)==CAP2,"R2 screen");
        need(ds4_gpu_tensor_read(out2,0,values[mode],2ull*CAP2*4u)&&
             ds4_gpu_tensor_read(ids2,0,selected_ids[mode],2ull*CAP2*4u),
             "R2 outputs");
        need(ds4_gpu_tensor_read(x2,0,after,2ull*DIM*4u)&&
             !memcmp(a2,after,2ull*DIM*4u),"R2 input unchanged");
    }
    need(!memcmp(selected_ids[0],selected_ids[1],2ull*CAP2*4u),
         "R2 fused-key shortlist differs");
    need(!memcmp(values[0],values[1],2ull*CAP2*4u),
         "R2 fused-key refinement differs");
    need(ds4_gpu_mtp_native_screen2(out2,ids2,scratch2,w,bytes,offset,
         DIM,VOCAB,PREFIX,TAIL,x2,1)==CAP2,"R2 deferred finite screen");
    need(ds4_gpu_tensor_read(out2,0,values[0],2ull*CAP2*4u)&&
         ds4_gpu_tensor_read(ids2,0,selected_ids[0],2ull*CAP2*4u),
         "R2 deferred finite outputs");
    need(!memcmp(selected_ids[0],selected_ids[1],2ull*CAP2*4u)&&
         !memcmp(values[0],values[1],2ull*CAP2*4u),
         "R2 deferred finite parity");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full2,w,bytes,offset,
         DIM,PREFIX,x2,2),"R2 ordinary prefix");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tail2,w,bytes,
         offset+(uint64_t)(VOCAB-TAIL)*ROW,DIM,TAIL,x2,2),
         "R2 ordinary tail");
    need(ds4_gpu_tensor_read(full2,0,reference,2ull*PREFIX*4u)&&
         ds4_gpu_tensor_read(tail2,0,tail_ref,2ull*TAIL*4u),
         "R2 oracle read");
    for(uint32_t r=0;r<2;r++) {
        const uint32_t *row_ids=selected_ids[1]+(uint64_t)r*CAP2;
        const float *row_values=values[1]+(uint64_t)r*CAP2;
        need(row_ids[0]==0,"R2 mandatory zero");
        for(uint32_t i=0;i<CAP2;i++) {
            const uint32_t id=row_ids[i];
            need(i==0||id>row_ids[i-1],"R2 sorted unique IDs");
            need(id<PREFIX||(id>=VOCAB-TAIL&&id<VOCAB),"R2 static domain");
            const float exact=id<PREFIX?
                reference[(uint64_t)r*PREFIX+id]:
                tail_ref[(uint64_t)r*TAIL+id-(VOCAB-TAIL)];
            need(!memcmp(&exact,&row_values[i],4),
                 "R2 refinement differs from ordinary row");
        }
        for(uint32_t i=0;i<TAIL;i++)
            need(row_ids[CAP2-TAIL+i]==VOCAB-TAIL+i,"R2 mandatory tail");
    }
    need(ds4_gpu_tensor_fill_f32(scattered,-FLT_MAX,2ull*VOCAB)&&
         ds4_gpu_mtp_native_scatter2(scattered,out2,ids2,scratch2,winner2,
                                     CAP2,VOCAB,WIDTH,1)&&
         ds4_gpu_indexer_topk_tensor(winner2,scattered,VOCAB,2,1)&&
         ds4_gpu_tensor_read(scattered,0,dense,2ull*VOCAB*4u),"R2 scatter");
    uint32_t winner_words[3];
    need(ds4_gpu_tensor_read(winner2,0,winner_words,sizeof winner_words)&&
         winner_words[2]==0u,"R2 deferred finite status");
    for(uint32_t r=0;r<2;r++) for(uint32_t i=0;i<CAP2;i++)
        need(!memcmp(&dense[(uint64_t)r*VOCAB+
                            selected_ids[1][(uint64_t)r*CAP2+i]],
                     &values[1][(uint64_t)r*CAP2+i],4),"R2 scattered value");
    need(ds4_gpu_mtp_native_top1_map2(winner2,out2,ids2,scratch2,CAP2,
         VOCAB,WIDTH,1)&&
         ds4_gpu_tensor_read(winner2,0,winner_words,sizeof winner_words),
         "R2 compact top1");
    for(uint32_t r=0;r<2;r++) {
        uint32_t best=0;
        for(uint32_t i=1;i<VOCAB;i++)
            if(dense[(uint64_t)r*VOCAB+i]>
               dense[(uint64_t)r*VOCAB+best])best=i;
        need(winner_words[r]==best,"R2 compact/dense top1 differs");
    }
    need(winner_words[2]==0u,"R2 compact deferred finite status");
    float *edge=malloc(2ull*CAP2*4u);
    need(edge!=NULL,"R2 edge allocation");
    for(uint32_t i=0;i<CAP2;i++) edge[i]=-INFINITY;
    edge[0]=NAN;
    for(uint32_t i=0;i<CAP2;i++) edge[CAP2+i]=-FLT_MAX;
    need(ds4_gpu_tensor_write(out2,0,edge,2ull*CAP2*4u)&&
         ds4_gpu_mtp_native_top1_map2(winner2,out2,ids2,scratch2,CAP2,
                                      VOCAB,WIDTH,1)&&
         ds4_gpu_tensor_read(winner2,0,winner_words,sizeof winner_words),
         "R2 compact top1 edge cases");
    uint32_t missing=0;
    while(missing<CAP2&&selected_ids[1][missing]==missing)missing++;
    need(winner_words[0]==missing,"R2 compact omitted -FLT_MAX sentinel");
    need(winner_words[1]==0,"R2 compact original-ID tie");
    need(winner_words[2]==0u,"R2 compact edge status");
    free(edge);
    setenv("DS4_QWEN4EXP_NO_TARGET_NATIVE_SCREEN_R2","1",1);
    need(ds4_gpu_mtp_native_screen2(out2,ids2,scratch2,w,bytes,offset,DIM,VOCAB,
         PREFIX,TAIL,x2,0)==0,"R2 valve fallback");
    unsetenv("DS4_QWEN4EXP_NO_TARGET_NATIVE_SCREEN_R2");
    ds4_gpu_decode_graphs_invalidate();
    ds4_decode_graph_key key={.il=4,.island=0,.variant=2};
    need(ds4_gpu_decode_graph_begin(&key)==-1,"R2 graph warmup");
    need(ds4_gpu_decode_graph_begin(&key)==0,"R2 capture start");
    need(ds4_gpu_mtp_native_screen2(out2,ids2,scratch2,w,bytes,offset,DIM,
         VOCAB,PREFIX,TAIL,x2,1)==0,"R2 capture declines screening");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full2,w,bytes,offset,
         DIM,PREFIX,x2,2),"R2 captured static fallback");
    need(ds4_gpu_decode_graph_end(&key)==0,"R2 capture end");
    ds4_gpu_decode_graphs_invalidate();
    free(a2);free(after);free(values[0]);free(values[1]);free(dense);
    free(reference);free(tail_ref);free(selected_ids[0]);free(selected_ids[1]);
    ds4_gpu_tensor_free(x2);ds4_gpu_tensor_free(out2);ds4_gpu_tensor_free(ids2);
    ds4_gpu_tensor_free(scratch2);ds4_gpu_tensor_free(scattered);
    ds4_gpu_tensor_free(full2);ds4_gpu_tensor_free(tail2);
    ds4_gpu_tensor_free(winner2);
}

static void check_r2_nonfinite_rows(const void *w,uint64_t bytes,uint64_t offset,
        const float *activation) {
    uint64_t scratch_bytes=0;uint32_t cap=0;
    need(ds4_gpu_mtp_native_screen2_init(WIDTH,&scratch_bytes,&cap)==1&&cap==CAP2,
         "R2 nonfinite scratch query");
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(2ull*DIM*4u),
        *out=ds4_gpu_tensor_alloc(2ull*CAP2*4u),
        *ids=ds4_gpu_tensor_alloc(2ull*CAP2*4u),
        *scratch=ds4_gpu_tensor_alloc(scratch_bytes),
        *dense=ds4_gpu_tensor_alloc(2ull*VOCAB*4u),
        *winner=ds4_gpu_tensor_alloc(3u*4u);
    float *rows=malloc(2ull*DIM*4u);
    need(x&&out&&ids&&scratch&&dense&&winner&&rows,
         "R2 nonfinite allocations");
    for(uint32_t bad=0;bad<2;bad++) {
        memcpy(rows,activation,DIM*4u);
        memcpy(rows+DIM,activation,DIM*4u);
        for(uint32_t i=0;i<32;i++) rows[(uint64_t)bad*DIM+i]=NAN;
        need(ds4_gpu_tensor_write(x,0,rows,2ull*DIM*4u),
             "R2 nonfinite activation upload");
        need(ds4_gpu_mtp_native_screen2(out,ids,scratch,w,bytes,offset,DIM,
             VOCAB,PREFIX,TAIL,x,0)==0,"R2 immediate nonfinite fallback");
        need(ds4_gpu_mtp_native_screen2(out,ids,scratch,w,bytes,offset,DIM,
             VOCAB,PREFIX,TAIL,x,1)==CAP2,"R2 deferred nonfinite screen");
        need(ds4_gpu_tensor_fill_f32(dense,-INFINITY,2ull*VOCAB)&&
             ds4_gpu_mtp_native_scatter2(dense,out,ids,scratch,winner,CAP2,
                                         VOCAB,WIDTH,1)&&
             ds4_gpu_indexer_topk_tensor(winner,dense,VOCAB,2,1),
             "R2 deferred nonfinite completion");
        uint32_t status[3];
        need(ds4_gpu_tensor_read(winner,0,status,sizeof status)&&status[2]==1u,
             bad?"R2 row-1 NaN status":"R2 row-0 NaN status");
        need(ds4_gpu_mtp_native_top1_map2(winner,out,ids,scratch,CAP2,VOCAB,
             WIDTH,1)&&ds4_gpu_tensor_read(winner,0,status,sizeof status)&&
             status[2]==1u,
             bad?"R2 compact row-1 NaN status":
                 "R2 compact row-0 NaN status");
    }
    setenv("DS4_MTP_NO_DEFER_INVALID_FLAG","1",1);
    need(ds4_gpu_mtp_native_screen2(out,ids,scratch,w,bytes,offset,DIM,VOCAB,
         PREFIX,TAIL,x,1)==0,"R2 deferred-invalid valve");
    unsetenv("DS4_MTP_NO_DEFER_INVALID_FLAG");
    free(rows);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(ids);ds4_gpu_tensor_free(scratch);
    ds4_gpu_tensor_free(dense);ds4_gpu_tensor_free(winner);
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
        /* 24 groups are negative; adding groups 24..31 makes the previous
         * 32-group screen positive. Both complete dots are positive, so
         * requiring exclusion witnesses the 24-group policy itself. */
        if(adversarial==3 && row==PREFIX-1) memset(p+2,b<24?255:b<32?8:2,32);
    }
    need(ds4_gpu_init(),"GPU init");need(ds4_gpu_set_model_map(w,bytes),"register weights");
    uint64_t scratch_bytes=0;uint32_t cap=0;
    need(ds4_gpu_mtp_native_screen_init(WIDTH,&scratch_bytes,&cap)==1 && cap==CAP,"scratch query");
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(DIM*4),*out=ds4_gpu_tensor_alloc(CAP*4),
        *ids=ds4_gpu_tensor_alloc(CAP*4),*scratch=ds4_gpu_tensor_alloc(scratch_bytes),
        *full=ds4_gpu_tensor_alloc(PREFIX*4),*tail=ds4_gpu_tensor_alloc(TAIL*4),
        *winner=ds4_gpu_tensor_alloc(8);
    need(x&&out&&ids&&scratch&&full&&tail&&winner,"GPU allocations");
    float activation[DIM],selected[CAP],reference[PREFIX],tail_ref[TAIL];uint32_t found[CAP];
    for(unsigned replay=0;replay<(adversarial?1u:3u);replay++) {
        for(unsigned i=0;i<DIM;i++) activation[i]=adversarial?1.0f:(int)(rnd()%201)*0.01f-1.0f;
        need(ds4_gpu_tensor_write(x,0,activation,sizeof activation),"current activation");
        if(adversarial==2) {
            need(compare_key_paths(out,ids,scratch,w,bytes,offset,x)==0,"nonfinite score fallback");
            need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,
                 VOCAB,PREFIX,TAIL,x,1)==CAP,"R1 deferred NaN screen");
            need(ds4_gpu_indexer_topk_tensor(winner,out,CAP,1,1)&&
                 ds4_gpu_mtp_native_map(winner,out,ids,scratch,CAP,VOCAB,
                                        WIDTH,1),
                 "R1 deferred NaN completion");
            uint32_t deferred[2];
            need(ds4_gpu_tensor_read(winner,0,deferred,sizeof deferred)&&
                 deferred[1]==1u,"R1 deferred NaN status");
            setenv("DS4_MTP_NO_DEFER_INVALID_FLAG","1",1);
            need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,
                 VOCAB,PREFIX,TAIL,x,1)==0,"R1 deferred-invalid valve");
            unsetenv("DS4_MTP_NO_DEFER_INVALID_FLAG");
            goto cleanup;
        }
        need(compare_key_paths(out,ids,scratch,w,bytes,offset,x)==CAP,"screen active");
        need(ds4_gpu_tensor_read(ids,0,found,sizeof found),"IDs read");
        need(ds4_gpu_tensor_read(out,0,selected,sizeof selected),"refine read");
        need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,
             VOCAB,PREFIX,TAIL,x,1)==CAP,"R1 deferred finite screen");
        uint32_t deferred_ids[CAP];float deferred_values[CAP];
        need(ds4_gpu_tensor_read(ids,0,deferred_ids,sizeof deferred_ids)&&
             ds4_gpu_tensor_read(out,0,deferred_values,sizeof deferred_values)&&
             !memcmp(found,deferred_ids,sizeof found)&&
             !memcmp(selected,deferred_values,sizeof selected),
             "R1 deferred finite parity");
        need(ds4_gpu_indexer_topk_tensor(winner,out,CAP,1,1)&&
             ds4_gpu_mtp_native_map(winner,out,ids,scratch,CAP,VOCAB,
                                    WIDTH,1),
             "R1 deferred finite completion");
        uint32_t deferred_status[2];
        need(ds4_gpu_tensor_read(winner,0,deferred_status,sizeof deferred_status)&&
             deferred_status[1]==0u,"R1 deferred finite status");
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
        if(adversarial) {
            need(reference[PREFIX-1]>0,"adversarial full winner");
            for(unsigned i=0;i<CAP;i++) need(found[i]!=PREFIX-1 && selected[i]==0,"screen is approximate");
            for(unsigned i=1;i<CAP-TAIL;i++) need(found[i]==i,"coarse tie lowest ID");
        }
        if(!adversarial && replay==0) {
            compare_r2_paths(w,bytes,offset,activation);
            check_r2_nonfinite_rows(w,bytes,offset,activation);
        }
    }
    /* Map original IDs, reject bad packed IDs, and preserve legacy NaN0. */
    uint32_t packed=CAP-1,mapped=0;
    need(ds4_gpu_tensor_write(winner,0,&packed,4)&&ds4_gpu_mtp_native_map(winner,out,ids,NULL,CAP,VOCAB,WIDTH,0)&&ds4_gpu_tensor_read(winner,0,&mapped,4),"winner map");
    need(mapped==VOCAB-1,"mapped tail ID");
    packed=CAP;
    need(ds4_gpu_tensor_write(winner,0,&packed,4)&&ds4_gpu_mtp_native_map(winner,out,ids,NULL,CAP,VOCAB,WIDTH,0)&&ds4_gpu_tensor_read(winner,0,&mapped,4)&&mapped==UINT32_MAX,"bad winner guard");
    float nan=NAN;
    need(ds4_gpu_tensor_write(out,0,&nan,4)&&ds4_gpu_mtp_native_map(winner,out,ids,NULL,CAP,VOCAB,WIDTH,0)&&ds4_gpu_tensor_read(winner,0,&mapped,4)&&mapped==0,"NaN0 pin");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,100,TAIL,x,0)==0,"small width fallback");
    setenv("DS4_QWEN4EXP_NO_ROW_TILE","1",1);
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,0)==0,"diagnostic fallback");
    unsetenv("DS4_QWEN4EXP_NO_ROW_TILE");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes-1,offset,DIM,VOCAB,PREFIX,TAIL,x,0)==-1,"truncated map");
    ds4_gpu_tensor *tiny=ds4_gpu_tensor_alloc(4);
    need(ds4_gpu_mtp_native_screen(out,ids,tiny,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,0)==-1,"scratch bound");
    ds4_gpu_tensor_free(tiny);
    ds4_gpu_decode_graphs_invalidate();
    ds4_decode_graph_key key={.il=3,.island=0,.variant=1};
    need(ds4_gpu_decode_graph_begin(&key)==-1,"graph warmup");
    need(ds4_gpu_decode_graph_begin(&key)==0,"capture start");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x,1)==0,"capture declines screening");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full,w,bytes,offset,DIM,PREFIX,x,1),"captured static fallback");
    need(ds4_gpu_decode_graph_end(&key)==0,"capture end");
    ds4_gpu_decode_graphs_invalidate();
cleanup:
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(ids);ds4_gpu_tensor_free(scratch);
    ds4_gpu_tensor_free(full);ds4_gpu_tensor_free(tail);ds4_gpu_tensor_free(winner);
    ds4_gpu_cleanup();munmap(w,bytes);
}
int main(void){setenv("DS4_CUDA_DECODE_GRAPHS","1",1);run_case(0,0);run_case(0,2);run_case(1,0);run_case(2,0);run_case(3,0);run_case(3,2);puts("native screen contracts pass (selection deliberately approximate)");return 0;}
