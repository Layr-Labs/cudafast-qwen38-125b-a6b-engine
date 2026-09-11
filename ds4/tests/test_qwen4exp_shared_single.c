/* Full-buffer/guard and changed-input graph checks for the one-row Q8
 * shared-expert schedule. Uses synthetic weights with actual 0/2-byte
 * device alignment residues. Link against the normal built CUDA library. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { TYPE_F32=0, TYPE_Q8_0=8 };
#define ALIGN64(x) (((uint64_t)(x)+63u)&~63ull)
static uint32_t rng=0x193d8a7u;
static uint32_t rng_u32(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static float rng_unit(void){return ((int)(rng_u32()%2049u)-1024)*(1.0f/1024.0f);}
static void require_ok(int ok,const char *message){if(!ok){fprintf(stderr,"shared single: %s\n",message);exit(1);}}
static uint64_t type_row_bytes(unsigned type,unsigned n){(void)type;return (uint64_t)(n/32u)*34u;}
static void prod_seed_row_scales(unsigned char*row,unsigned type,unsigned n){
    (void)type;const uint16_t scale=0x1400;
    for(unsigned g=0;g<n/32u;g++)memcpy(row+g*34u,&scale,2);
}
/* The flag selects the original shared schedule as the direct oracle. */
static void shared_single_case(unsigned rows, unsigned alignment) {
    enum { K=2560, D=640, O=2560, CAP=2, GUARD=64, M=2 };
    const uint64_t gr=type_row_bytes(TYPE_Q8_0,K), dr=type_row_bytes(TYPE_Q8_0,D);
    const uint64_t go=ALIGN64(K*4u)+64+alignment, uo=go+D*gr+64, doff=uo+D*gr+64;
    const uint64_t bytes=doff+O*dr+64;
    unsigned char *image=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    require_ok(image!=MAP_FAILED,"shared tile mapping");
    for(uint64_t i=0;i<bytes;i++)image[i]=(unsigned char)rng_u32();
    for(unsigned i=0;i<K;i++){float v=rng_unit()*.02f;memcpy(image+i*4,&v,4);}
    for(unsigned i=0;i<D;i++){
        prod_seed_row_scales(image+go+i*gr,TYPE_Q8_0,K);
        prod_seed_row_scales(image+uo+i*gr,TYPE_Q8_0,K);
    }
    for(unsigned i=0;i<O;i++)prod_seed_row_scales(image+doff+i*dr,TYPE_Q8_0,D);
    const uint16_t specials[]={0,0x8000,1,0x8001,0xa400};
    for(unsigned i=0;i<5;i++){
        memcpy(image+go+i*gr,&specials[i],2);memcpy(image+uo+i*gr,&specials[i],2);
        memcpy(image+doff+i*dr,&specials[i],2);
    }
    ds4_gpu_qwen4exp_slab router={image,bytes,0,0,K*4,TYPE_F32};
    ds4_gpu_qwen4exp_slab gate={image,bytes,go,0,gr,TYPE_Q8_0},up=gate;
    up.offset=uo;ds4_gpu_qwen4exp_slab down={image,bytes,doff,0,dr,TYPE_Q8_0};
    require_ok(ds4_gpu_init()&&ds4_gpu_set_model_map(image,bytes),"shared tile model");
    const size_t xpbytes=CAP*K*4+2*GUARD,xoff=GUARD+(alignment?4:0);
    ds4_gpu_tensor*xp=ds4_gpu_tensor_alloc(xpbytes),*x=ds4_gpu_tensor_view(xp,xoff,CAP*K*4);
    unsigned char *xh=malloc(xpbytes),*xc=malloc(xpbytes);
    const size_t sizes[3]={CAP*O*4+GUARD,CAP*D*4+GUARD,CAP*4+GUARD};
    ds4_gpu_tensor*t[M][3];unsigned char*ref[3],*got[3],*poison[3];
    require_ok(xp&&x&&xh&&xc,"shared tile input allocation");
    for(unsigned j=0;j<3;j++){
        ref[j]=malloc(sizes[j]);got[j]=malloc(sizes[j]);poison[j]=malloc(sizes[j]);
        require_ok(ref[j]&&got[j]&&poison[j],"shared tile host output");
        memset(poison[j],0x3c,sizes[j]);
        for(unsigned m=0;m<M;m++){t[m][j]=ds4_gpu_tensor_alloc(sizes[j]);require_ok(t[m][j]!=NULL,"shared tile output");}
    }
#define RESET(m) do {for(unsigned j=0;j<3;j++)require_ok(ds4_gpu_tensor_write(t[m][j],0,poison[j],sizes[j]),"shared tile poison");}while(0)
#define PIN(m) require_ok(((m)?unsetenv("DS4_QWEN4EXP_NO_SHARED_R1"):setenv("DS4_QWEN4EXP_NO_SHARED_R1","1",1))==0,"shared single dispatch")
#define RUN(m) require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(t[m][0],t[m][1],t[m][2],&router,&gate,&up,&down,K,D,O,x,rows),"shared tile call")
#define READ_COMPARE(m, counter) do {for(unsigned j=0;j<3;j++){require_ok(ds4_gpu_tensor_read(t[m][j],0,got[j],sizes[j]),"shared tile output read");if(memcmp(ref[j],got[j],sizes[j])){fprintf(stderr,"SHARED_SINGLE mismatch rows=%u offset=%u mode=%u buffer=%u\n",rows,alignment,m,j);exit(1);}counter++;}}while(0)
    ds4_decode_graph_key keys[M];memset(keys,0,sizeof(keys));
    for(unsigned m=0;m<M;m++){keys[m].il=m+1;keys[m].cur_hc=x;keys[m].after_attn_hc=t[m][0];}
    ds4_gpu_decode_graphs_invalidate();unsigned eager=0,graphs=0,replays=0;
    const float mags[]={.2f,1e-30f,1e3f,0.0f};
    for(unsigned trial=0;trial<12;trial++){
        memset(xh,0xa6,xpbytes);float*v=(float*)(xh+xoff);
        for(unsigned i=0;i<CAP*K;i++)v[i]=rng_unit()*mags[trial%4];
        require_ok(ds4_gpu_tensor_write(xp,0,xh,xpbytes),"shared tile changed input");
        PIN(0);RESET(0);RUN(0);
        for(unsigned j=0;j<3;j++)require_ok(ds4_gpu_tensor_read(t[0][j],0,ref[j],sizes[j]),"shared tile reference");
        PIN(1);RESET(1);RUN(1);READ_COMPARE(1,eager);
        for(unsigned m=0;m<M;m++){
            PIN(m);RESET(m);int state=ds4_gpu_decode_graph_begin(&keys[m]);
            if(trial>=2)require_ok(state==1,"shared tile actual replay");
            if(state!=1){require_ok(state==-1||state==0,"shared tile graph state");RUN(m);if(state==0)require_ok(ds4_gpu_decode_graph_end(&keys[m])==0,"shared tile capture");}else replays++;
            READ_COMPARE(m,graphs);
        }
        require_ok(ds4_gpu_tensor_read(xp,0,xc,xpbytes)&&!memcmp(xh,xc,xpbytes),"shared tile input/guard mutation");
    }
    printf("SHARED_SINGLE rows=%u weight_offset=%llu eager_buffers=%u graph_buffers=%u replays=%u PASS\n",rows,(unsigned long long)go,eager,graphs,replays);fflush(stdout);
    ds4_gpu_decode_graphs_invalidate();
    for(unsigned j=0;j<3;j++){for(unsigned m=0;m<M;m++)ds4_gpu_tensor_free(t[m][j]);free(ref[j]);free(got[j]);free(poison[j]);}
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(xp);ds4_gpu_cleanup();munmap(image,bytes);free(xh);free(xc);
#undef READ_COMPARE
#undef RUN
#undef PIN
#undef RESET
}

int main(void){
    require_ok(unsetenv("DS4_QWEN4EXP_MOE_R")==0,"clear forced row tile");
    require_ok(setenv("DS4_CUDA_COPY_MODEL","1",1)==0&&setenv("DS4_CUDA_DECODE_GRAPHS","1",1)==0,"shared tile environment");
    for(unsigned a=0;a<2;a++)for(unsigned rows=1;rows<=2;rows++)shared_single_case(rows,a*2);
    unsetenv("DS4_QWEN4EXP_NO_SHARED_R1");return 0;
}
