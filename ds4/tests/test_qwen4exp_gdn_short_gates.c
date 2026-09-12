/* Compare short convolution gate publication with the original GDN API. Every
 * comparison covers nine complete buffers, unused rows, and 64-byte tails.
 * Capture/replay changes inputs, carried state, layout, and gate magnitudes. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define main retained_reference_main
#include "test_qwen4exp_gdn.c"
#undef main
#include "ds4_gpu_mgpu.h"

enum { OUT, HIST, STATE, QKV, ALPHA, BETA, OGATE, CSNAP, SSNAP, FIELDS };
typedef struct {
    ds4_gpu_tensor *v[FIELDS], *src[4], *short_gates;
    size_t bytes[FIELDS];
    unsigned cap, snapcap;
} gate_case;
static unsigned rnd_state=0xf0154a23u;
static float random_value(void) {
    rnd_state^=rnd_state<<13; rnd_state^=rnd_state>>17; rnd_state^=rnd_state<<5;
    return ((int)(rnd_state%20001)-10000)*.0001f;
}
static unsigned candidate_mode;
static void mode(unsigned candidate) {
    candidate_mode = candidate;
    if (candidate) unsetenv("DS4_QWEN4EXP_NO_GDN_SHORT_GATES");
    else require_ok(setenv("DS4_QWEN4EXP_NO_GDN_SHORT_GATES","1",1)==0,"reference mode");
}
static void write_values(ds4_gpu_tensor *v,size_t bytes,float scale,unsigned pattern) {
    float *h=require_alloc(bytes,"host values");
    for(size_t i=0;i<bytes/4;i++) h[i]=random_value()*scale;
    if(pattern) for(size_t i=0;i<bytes/4;i++) h[i]=(i&1)?90.0f:-90.0f;
    require_ok(ds4_gpu_tensor_write(v,0,h,bytes),"values write");free(h);
}
static void allocate_case(gate_case *a,unsigned cap,unsigned snapshots) {
    a->cap=cap;a->snapcap=snapshots;
    a->short_gates=ds4_gpu_tensor_alloc(2u*VALUE_HEADS*8u+64u);
    require_ok(a->short_gates!=NULL,"short gate scratch");
    const size_t counts[FIELDS]={cap*VALUE_DIM,HISTORY*CONV_DIM,STATE_ELEMENTS,
        cap*CONV_DIM,cap*VALUE_HEADS,cap*VALUE_HEADS,cap*VALUE_DIM,
        snapshots*HISTORY*CONV_DIM,snapshots*STATE_ELEMENTS};
    for(unsigned j=0;j<FIELDS;j++) {
        a->bytes[j]=counts[j]*4+64;
        a->v[j]=ds4_gpu_tensor_alloc(a->bytes[j]);
        require_ok(a->v[j]!=NULL,"case allocation");
    }
    const unsigned fields[]={QKV,HIST,STATE};
    for(unsigned j=0;j<3;j++) {
        a->src[j]=ds4_gpu_tensor_alloc(a->bytes[fields[j]]);
        require_ok(a->src[j]!=NULL,"source allocation");
    }
    size_t poison_bytes=0;
    for(unsigned j=0;j<FIELDS;j++)if(a->bytes[j]>poison_bytes)poison_bytes=a->bytes[j];
    a->src[3]=ds4_gpu_tensor_alloc(poison_bytes);
    require_ok(a->src[3]!=NULL,"poison source allocation");
    require_ok(ds4_gpu_tensor_fill_f32(a->src[3],.375f,poison_bytes/4),"initial poison fill");
}
static void refill(gate_case *a,float scale,unsigned pattern) {
    const unsigned fields[]={QKV,HIST,STATE};
    for(unsigned j=0;j<3;j++)write_values(a->src[j],a->bytes[fields[j]],scale,0);
    write_values(a->v[ALPHA],a->bytes[ALPHA],4.0f,pattern);
    write_values(a->v[BETA],a->bytes[BETA],4.0f,pattern);
    write_values(a->v[OGATE],a->bytes[OGATE],4.0f,0);
}
static void reset_case(gate_case *a,int full) {
    require_ok(ds4_gpu_tensor_fill_f32(a->short_gates,.375f,
        (2u*VALUE_HEADS*8u+64u)/4u),"scratch canary reset");
    require_ok(ds4_gpu_tensor_copy(a->v[QKV],0,a->src[0],0,a->bytes[QKV]),"qkv reset");
    if(!full)return;
    require_ok(ds4_gpu_tensor_copy(a->v[HIST],0,a->src[1],0,a->bytes[HIST]),"history reset");
    require_ok(ds4_gpu_tensor_copy(a->v[STATE],0,a->src[2],0,a->bytes[STATE]),"state reset");
    const unsigned fields[]={OUT,CSNAP,SSNAP};
    for(unsigned j=0;j<3;j++)require_ok(ds4_gpu_tensor_copy(a->v[fields[j]],0,a->src[3],0,a->bytes[fields[j]]),"canary reset");
}
static void operate(gate_case *a,const void *model,const weight_set *ws,
                    unsigned tokens,unsigned snapshots) {
    ds4_gpu_qwen4exp_slab c=gdn_slab(model,ws->conv_offset),l=gdn_slab(model,ws->a_log_offset);
    ds4_gpu_qwen4exp_slab b=gdn_slab(model,ws->dt_bias_offset),n=gdn_slab(model,ws->norm_offset);
    if(candidate_mode) {
        require_ok(ds4_gpu_qwen4exp_gdn_prefill_short_gates(a->short_gates,
            a->v[OUT],a->v[HIST],a->v[STATE],a->v[CSNAP],a->v[SSNAP],snapshots,
            a->v[QKV],a->v[ALPHA],a->v[BETA],a->v[OGATE],&c,&l,&b,&n,
            KEY_HEADS,VALUE_HEADS,tokens,ws->layout,QK_NORM_EPS,NORM_EPS),
            "short-gate GDN operator");
        return;
    }
    require_ok(ds4_gpu_qwen4exp_gdn_prefill(a->v[OUT],a->v[HIST],a->v[STATE],
        a->v[CSNAP],a->v[SSNAP],snapshots,a->v[QKV],a->v[ALPHA],a->v[BETA],
        a->v[OGATE],&c,&l,&b,&n,KEY_HEADS,VALUE_HEADS,tokens,ws->layout,
        QK_NORM_EPS,NORM_EPS),"full GDN operator");
}
static void check_scratch_contract(gate_case *a, const void *model) {
    const weight_set *ws=&g_tiled;
    ds4_gpu_qwen4exp_slab c=gdn_slab(model,ws->conv_offset),l=gdn_slab(model,ws->a_log_offset);
    ds4_gpu_qwen4exp_slab b=gdn_slab(model,ws->dt_bias_offset),n=gdn_slab(model,ws->norm_offset);
#define TRY(S) ds4_gpu_qwen4exp_gdn_prefill_short_gates(S,a->v[OUT],a->v[HIST],a->v[STATE], \
    a->v[CSNAP],a->v[SSNAP],1,a->v[QKV],a->v[ALPHA],a->v[BETA],a->v[OGATE], \
    &c,&l,&b,&n,KEY_HEADS,VALUE_HEADS,2,ws->layout,QK_NORM_EPS,NORM_EPS)
    unsigned char *before[FIELDS];
    for(unsigned j=0;j<FIELDS;j++) {
        before[j]=require_alloc(a->bytes[j],"contract before");
        require_ok(ds4_gpu_tensor_read(a->v[j],0,before[j],a->bytes[j]),"contract read");
    }
    ds4_gpu_tensor *small=ds4_gpu_tensor_view(a->short_gates,0,2u*VALUE_HEADS*8u-1u);
    ds4_gpu_tensor *unaligned=ds4_gpu_tensor_view(a->src[3],4,2u*VALUE_HEADS*8u);
    require_ok(small&&unaligned,"contract views");
    require_ok(!TRY(small),"short scratch rejected");
    require_ok(!TRY(unaligned),"unaligned scratch rejected");
    require_ok(!TRY(a->v[QKV]),"overlapping scratch rejected");
    ds4_gpu_tensor wrong=*a->short_gates; wrong.device_id=a->v[OUT]->device_id+1;
    require_ok(!TRY(&wrong),"wrong device scratch rejected");
    for(unsigned j=0;j<FIELDS;j++) {
        unsigned char *after=require_alloc(a->bytes[j],"contract after");
        require_ok(ds4_gpu_tensor_read(a->v[j],0,after,a->bytes[j]),"contract read");
        require_ok(!memcmp(before[j],after,a->bytes[j]),"rejection mutates no buffer");
        free(after);free(before[j]);
    }
    ds4_gpu_tensor_free(small);ds4_gpu_tensor_free(unaligned);
    require_ok(TRY(NULL),"NULL scratch legacy fallback");
#undef TRY
}
static void snapshot_or_compare(gate_case *a,unsigned char **ref,int compare) {
    for(unsigned j=0;j<FIELDS;j++) {
        unsigned char *got=require_alloc(a->bytes[j],"readback");
        require_ok(ds4_gpu_tensor_read(a->v[j],0,got,a->bytes[j]),"complete readback");
        if(compare) {
            if(memcmp(ref[j],got,a->bytes[j])) {
                for(size_t k=0;k<a->bytes[j];k++)if(ref[j][k]!=got[k]) {
                    fprintf(stderr,"GDN short-gate mismatch field=%u byte=%zu old=%02x new=%02x\n",j,k,ref[j][k],got[k]);break;
                }
                exit(1);
            }
            free(got);
        } else {free(ref[j]);ref[j]=got;}
    }
}
static void inspect_scratch(gate_case *a,unsigned tokens,unsigned candidate) {
    const unsigned bytes=2u*VALUE_HEADS*8u+64u;
    float *h=require_alloc(bytes,"scratch readback");
    require_ok(ds4_gpu_tensor_read(a->short_gates,0,h,bytes),"scratch readback");
    const unsigned written=candidate&&tokens<=2u?tokens*VALUE_HEADS*2u:0u;
    int changed=0;
    for(unsigned i=0;i<written;i++)if(h[i]!=.375f)changed=1;
    if(written)require_ok(changed,"actual short gate producer ran");
    for(unsigned i=written;i<bytes/4u;i++)require_ok(h[i]==.375f,"scratch tail preserved");
    free(h);
}
static void free_case(gate_case *a) {
    ds4_gpu_tensor_free(a->short_gates);
    for(unsigned j=0;j<FIELDS;j++)ds4_gpu_tensor_free(a->v[j]);
    for(unsigned j=0;j<4;j++)ds4_gpu_tensor_free(a->src[j]);
}
int main(void) {
    require_ok(setenv("DS4_CUDA_DECODE_GRAPHS","1",1)==0,"graphs enabled");
    uint8_t *model=mmap(NULL,MODEL_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    require_ok(model!=MAP_FAILED,"mapping");build_weights(model);
    require_ok(ds4_gpu_init(),"GPU init");require_ok(ds4_gpu_set_model_map(model,MODEL_BYTES),"model map");
    gate_case a={0};allocate_case(&a,1025,6);refill(&a,.2f,0);reset_case(&a,1);
    // Grow the convolution scratch before creating any captured operation.
    mode(0);operate(&a,model,&g_tiled,1025,6);
    mode(1);reset_case(&a,1);check_scratch_contract(&a,model);free_case(&a);
    unsigned char *ref[FIELDS]={0};unsigned eager=0,graphs=0,changed=0;
    const unsigned widths[]={1,2,3,63,64,65};
    const float scales[]={.2f,1e-20f,10.0f,0.0f};
    const weight_set *sets[]={&g_tiled,&g_grouped,&g_fast_decay};
    for(unsigned wi=0;wi<sizeof(widths)/sizeof(widths[0]);wi++) {
      memset(&a,0,sizeof(a));allocate_case(&a,widths[wi]+1u,6);
      for(unsigned si=0;si<4;si++) {
        unsigned tokens=widths[wi],snap=si==3?0:(tokens<8?tokens-1:6);
        ds4_gpu_decode_graphs_invalidate();
        ds4_decode_graph_key keys[2]={{.il=1},{.il=2}};
        for(unsigned trial=0;trial<6;trial++) {
            refill(&a,scales[trial%4],trial==2);
            if(trial>=4) {
                const uint32_t bits[]={0x7fc12345u,0x7f800000u,0xff800000u,0x80000000u};
                for(unsigned field=ALPHA;field<=BETA;field++) {
                    uint32_t *h=require_alloc(a.bytes[field],"nonfinite gates");
                    for(size_t j=0;j<a.bytes[field]/4;j++)h[j]=bits[(j+trial+field)%4];
                    require_ok(ds4_gpu_tensor_write(a.v[field],0,h,a.bytes[field]),"nonfinite upload");free(h);
                }
            }
            for(unsigned v=0;v<2;v++) {
                mode(v);reset_case(&a,1);operate(&a,model,sets[si%3],tokens,snap);
                snapshot_or_compare(&a,ref,v!=0);
                inspect_scratch(&a,tokens,v);
                if(v)eager++;
            }
            for(unsigned v=0;v<2;v++) {
                mode(v);
                if(trial==0) {
                    require_ok(ds4_gpu_decode_graph_begin(&keys[v])==-1,"graph warmup");
                    reset_case(&a,1);operate(&a,model,sets[si%3],tokens,snap);
                }
                int state=ds4_gpu_decode_graph_begin(&keys[v]);
                require_ok(state==0||state==1,"graph begin");
                if(trial>0){require_ok(state==1,"changed-input replay");changed++;}
                if(state==0){reset_case(&a,1);operate(&a,model,sets[si%3],tokens,snap);require_ok(ds4_gpu_decode_graph_end(&keys[v])==0,"graph capture");}
                snapshot_or_compare(&a,ref,v!=0);
                inspect_scratch(&a,tokens,v);
                if(v)graphs++;
            }
        }
        printf("GDN_SHORT_GATES_CHECK tokens=%u layout=%u eager=%u graph=%u changed=%u PASS\n",tokens,si,eager,graphs,changed);fflush(stdout);
      }
      ds4_gpu_decode_graphs_invalidate();free_case(&a);
    }
    printf("GDN_SHORT_GATES_COMPLETE eager=%u graph=%u changed=%u PASS\n",eager,graphs,changed);fflush(stdout);
    for(unsigned j=0;j<FIELDS;j++)free(ref[j]);
    unsetenv("DS4_QWEN4EXP_NO_GDN_SHORT_GATES");
    ds4_gpu_cleanup();munmap(model,MODEL_BYTES);return 0;
}
