/* Compare long-chunk value-row reuse with the retained recurrence.
 * --short instead isolates one/two-token value-row reuse; other widths exercise
 * its unchanged fallback. Every
 * comparison covers nine complete buffers, unused rows, and 64-byte tails.
 * Capture/replay changes inputs, carried state, layout, and gate magnitudes. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define main retained_reference_main
#include "test_qwen4exp_gdn.c"
#undef main

enum { OUT, HIST, STATE, QKV, ALPHA, BETA, OGATE, CSNAP, SSNAP, FIELDS };
typedef struct {
    ds4_gpu_tensor *v[FIELDS], *src[4];
    size_t bytes[FIELDS];
    unsigned cap, snapcap;
} gate_case;
static unsigned rnd_state=0xf0154a23u;
static float random_value(void) {
    rnd_state^=rnd_state<<13; rnd_state^=rnd_state>>17; rnd_state^=rnd_state<<5;
    return ((int)(rnd_state%20001)-10000)*.0001f;
}
static const char *mode_switch = "DS4_QWEN4EXP_NO_GDN_VALUE_REUSE";
static unsigned short_mode;
static void mode(unsigned candidate) {
    if (candidate) unsetenv(mode_switch);
    else require_ok(setenv(mode_switch,"1",1)==0,"reference mode");
}
static void write_values(ds4_gpu_tensor *v,size_t bytes,float scale,unsigned pattern) {
    float *h=require_alloc(bytes,"host values");
    for(size_t i=0;i<bytes/4;i++) h[i]=random_value()*scale;
    if(pattern) for(size_t i=0;i<bytes/4;i++) h[i]=(i&1)?90.0f:-90.0f;
    require_ok(ds4_gpu_tensor_write(v,0,h,bytes),"values write");free(h);
}
static void allocate_case(gate_case *a,unsigned cap,unsigned snapshots) {
    a->cap=cap;a->snapcap=snapshots;
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
    require_ok(ds4_gpu_qwen4exp_gdn_prefill(a->v[OUT],a->v[HIST],a->v[STATE],
        a->v[CSNAP],a->v[SSNAP],snapshots,a->v[QKV],a->v[ALPHA],a->v[BETA],
        a->v[OGATE],&c,&l,&b,&n,KEY_HEADS,VALUE_HEADS,tokens,ws->layout,
        QK_NORM_EPS,NORM_EPS),"full GDN operator");
}
static void snapshot_or_compare(gate_case *a,unsigned char **ref,int compare) {
    for(unsigned j=0;j<FIELDS;j++) {
        unsigned char *got=require_alloc(a->bytes[j],"readback");
        require_ok(ds4_gpu_tensor_read(a->v[j],0,got,a->bytes[j]),"complete readback");
        if(compare) {
            if(memcmp(ref[j],got,a->bytes[j])) {
                for(size_t k=0;k<a->bytes[j];k++)if(ref[j][k]!=got[k]) {
                    fprintf(stderr,"GDN value-row mismatch field=%u byte=%zu old=%02x new=%02x\n",j,k,ref[j][k],got[k]);break;
                }
                exit(1);
            }
            free(got);
        } else {free(ref[j]);ref[j]=got;}
    }
}
static void free_case(gate_case *a) {
    for(unsigned j=0;j<FIELDS;j++)ds4_gpu_tensor_free(a->v[j]);
    for(unsigned j=0;j<4;j++)ds4_gpu_tensor_free(a->src[j]);
}
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--short") == 0) {
        short_mode = 1;
        mode_switch = "DS4_QWEN4EXP_NO_GDN_SHORT_REUSE";
    } else require_ok(argc == 1, "usage: test-qwen4exp-gdn-value-reuse [--short]");
    require_ok(setenv("DS4_CUDA_DECODE_GRAPHS","1",1)==0,"graphs enabled");
    uint8_t *model=mmap(NULL,MODEL_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    require_ok(model!=MAP_FAILED,"mapping");build_weights(model);
    require_ok(ds4_gpu_init(),"GPU init");require_ok(ds4_gpu_set_model_map(model,MODEL_BYTES),"model map");
    gate_case a={0};allocate_case(&a,1025,6);refill(&a,.2f,0);reset_case(&a,1);
    // Grow the convolution scratch before creating any captured operation.
    mode(0);operate(&a,model,&g_tiled,1025,6);free_case(&a);
    unsigned char *ref[FIELDS]={0};unsigned eager=0,graphs=0,changed=0;
    const unsigned widths[]={1,2,3,7,8,63,64,65,256,1024,1025};
    const float scales[]={.2f,1e-20f,10.0f,0.0f};
    const weight_set *sets[]={&g_tiled,&g_grouped,&g_fast_decay};
    for(unsigned wi=0;wi<sizeof(widths)/sizeof(widths[0]);wi++) {
      memset(&a,0,sizeof(a));allocate_case(&a,widths[wi]+1u,6);
      for(unsigned si=0;si<3;si++) {
        unsigned tokens=widths[wi],snap=tokens<8?tokens-1:6;
        /* Short widths cover no snapshots, a prefix, and every live token.
         * Snapshot count is fixed per graph key, then data change on replay. */
        if (short_mode && tokens <= 2u) snap = si == 0u ? 0u : si == 1u ? 1u : tokens;
        ds4_gpu_decode_graphs_invalidate();
        ds4_decode_graph_key keys[2]={{.il=1},{.il=2}};
        for(unsigned trial=0;trial<4;trial++) {
            refill(&a,scales[trial],trial==2);
            for(unsigned v=0;v<2;v++) {
                mode(v);reset_case(&a,1);operate(&a,model,sets[si],tokens,snap);
                snapshot_or_compare(&a,ref,v!=0);
                if(v)eager++;
            }
            for(unsigned v=0;v<2;v++) {
                mode(v);
                if(trial==0) {
                    require_ok(ds4_gpu_decode_graph_begin(&keys[v])==-1,"graph warmup");
                    reset_case(&a,1);operate(&a,model,sets[si],tokens,snap);
                }
                int state=ds4_gpu_decode_graph_begin(&keys[v]);
                require_ok(state==0||state==1,"graph begin");
                if(trial>0){require_ok(state==1,"changed-input replay");changed++;}
                if(state==0){reset_case(&a,1);operate(&a,model,sets[si],tokens,snap);require_ok(ds4_gpu_decode_graph_end(&keys[v])==0,"graph capture");}
                snapshot_or_compare(&a,ref,v!=0);
                if(v)graphs++;
            }
        }
        printf("GDN_VALUE_REUSE_CHECK tokens=%u layout=%u eager=%u graph=%u changed=%u PASS\n",tokens,si,eager,graphs,changed);fflush(stdout);
      }
      ds4_gpu_decode_graphs_invalidate();free_case(&a);
    }
    printf("GDN_VALUE_REUSE_COMPLETE eager=%u graph=%u changed=%u PASS\n",eager,graphs,changed);fflush(stdout);
    for(unsigned j=0;j<FIELDS;j++)free(ref[j]);
    unsetenv(mode_switch);
    ds4_gpu_cleanup();munmap(model,MODEL_BYTES);return 0;
}
