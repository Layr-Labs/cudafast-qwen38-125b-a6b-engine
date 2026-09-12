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
#define CAP 16384u
#define PREFIX 20000u
#define TAIL 276u
#define VOCAB 21000u
#define WIDTH (PREFIX+TAIL)
#define ROW (80u*34u)
static void need(int ok,const char *s) {if(!ok){fprintf(stderr,"native screen: %s\n",s);exit(1);}}
static uint32_t seed=1234567;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
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
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(DIM*4),*out=ds4_gpu_tensor_alloc(CAP*4),
        *ids=ds4_gpu_tensor_alloc(CAP*4),*scratch=ds4_gpu_tensor_alloc(scratch_bytes),
        *full=ds4_gpu_tensor_alloc(PREFIX*4),*tail=ds4_gpu_tensor_alloc(TAIL*4),
        *winner=ds4_gpu_tensor_alloc(4);
    need(x&&out&&ids&&scratch&&full&&tail&&winner,"GPU allocations");
    float activation[DIM],selected[CAP],reference[PREFIX],tail_ref[TAIL];uint32_t found[CAP];
    for(unsigned replay=0;replay<(adversarial?1u:3u);replay++) {
        for(unsigned i=0;i<DIM;i++) activation[i]=adversarial?1.0f:(int)(rnd()%201)*0.01f-1.0f;
        need(ds4_gpu_tensor_write(x,0,activation,sizeof activation),"current activation");
        if(adversarial==2) {
            need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==0,"nonfinite score fallback");
            goto cleanup;
        }
        need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==CAP,"screen active");
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
        if(adversarial) {
            need(reference[PREFIX-1]>0,"adversarial full winner");
            for(unsigned i=0;i<CAP;i++) need(found[i]!=PREFIX-1 && selected[i]==0,"screen is approximate");
            for(unsigned i=1;i<CAP-TAIL;i++) need(found[i]==i,"coarse tie lowest ID");
        }
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
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==0,"diagnostic fallback");
    unsetenv("DS4_QWEN4EXP_NO_ROW_TILE");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes-1,offset,DIM,VOCAB,PREFIX,TAIL,x)==-1,"truncated map");
    ds4_gpu_tensor *tiny=ds4_gpu_tensor_alloc(4);
    need(ds4_gpu_mtp_native_screen(out,ids,tiny,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==-1,"scratch bound");
    ds4_gpu_tensor_free(tiny);
    ds4_gpu_decode_graphs_invalidate();
    ds4_decode_graph_key key={.il=3,.island=0,.variant=1};
    need(ds4_gpu_decode_graph_begin(&key)==-1,"graph warmup");
    need(ds4_gpu_decode_graph_begin(&key)==0,"capture start");
    need(ds4_gpu_mtp_native_screen(out,ids,scratch,w,bytes,offset,DIM,VOCAB,PREFIX,TAIL,x)==0,"capture declines screening");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(full,w,bytes,offset,DIM,PREFIX,x,1),"captured static fallback");
    need(ds4_gpu_decode_graph_end(&key)==0,"capture end");
    ds4_gpu_decode_graphs_invalidate();
cleanup:
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(ids);ds4_gpu_tensor_free(scratch);
    ds4_gpu_tensor_free(full);ds4_gpu_tensor_free(tail);ds4_gpu_tensor_free(winner);
    ds4_gpu_cleanup();munmap(w,bytes);
}
int main(void){setenv("DS4_CUDA_DECODE_GRAPHS","1",1);run_case(0,0);run_case(0,2);run_case(1,0);run_case(2,0);puts("native screen contracts pass (selection deliberately approximate)");return 0;}
