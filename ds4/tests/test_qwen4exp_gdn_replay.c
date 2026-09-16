/* Production CUDA A/B contract, synthetic data only. No target checkpoint is
 * required. Run on a CUDA GPU; the host simulator is a separate check. */
#define main gdn_reference_suite_main
#include "test_qwen4exp_gdn.c"
#undef main

static void compare_tensor(const char *label, ds4_gpu_tensor *a,
                           ds4_gpu_tensor *b, uint64_t bytes) {
    void *av=require_alloc(bytes,label), *bv=require_alloc(bytes,label);
    require_ok(ds4_gpu_tensor_read(a,0,av,bytes),label);
    require_ok(ds4_gpu_tensor_read(b,0,bv,bytes),label);
    require_ok(memcmp(av,bv,bytes)==0,label);
    free(av);free(bv);
}

static void replay_case(void *model, const weight_set *ws, bool captured) {
    const uint64_t sb=(uint64_t)STATE_ELEMENTS*sizeof(float);
    const uint64_t cb=(uint64_t)HISTORY*CONV_DIM*sizeof(float);
    const uint64_t tb=DS4_QWEN4EXP_GDN_REPLAY_ROWS*
        (uint64_t)(KEY_DIM+VALUE_DIM+2*VALUE_HEADS)*sizeof(float);
    const uint64_t qbytes=2u*VALUE_DIM, soff=(qbytes+15u)&~15ull;
    const uint64_t qb=soff+2ull*(VALUE_DIM/32)*sizeof(float);
    gpu_buffers b[2];
    ds4_gpu_tensor *snap[2],*csnap[2],*quant[2];
    for(unsigned j=0;j<2;j++) {
        buffers_alloc(&b[j],1,2);buffers_clear_state(&b[j],1);
        snap[j]=ds4_gpu_tensor_alloc(6u*sb);csnap[j]=ds4_gpu_tensor_alloc(6u*cb);
        quant[j]=ds4_gpu_tensor_alloc(qb);
        require_ok(snap[j] && csnap[j] && quant[j],"snapshot allocation");
    }
    ds4_gpu_tensor *checkpoint=ds4_gpu_tensor_alloc(sb);
    ds4_gpu_tensor *tape=ds4_gpu_tensor_alloc(tb);
    ds4_gpu_tensor *control=ds4_gpu_tensor_alloc(4),*adopt=ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *materialized=ds4_gpu_tensor_alloc(sb);
    require_ok(checkpoint && tape && control && adopt && materialized,"replay allocation");
    ds4_gpu_tensor *gates=ds4_gpu_tensor_alloc(2u*VALUE_HEADS*2u*sizeof(float));
    require_ok(gates!=NULL,"gate scratch allocation");
    ds4_gpu_qwen4exp_gdn_replay replay={checkpoint,tape,control,gates};
    const ds4_gpu_qwen4exp_slab cw=gdn_slab(model,ws->conv_offset);
    const ds4_gpu_qwen4exp_slab aw=gdn_slab(model,ws->a_log_offset);
    const ds4_gpu_qwen4exp_slab dw=gdn_slab(model,ws->dt_bias_offset);
    const ds4_gpu_qwen4exp_slab nw=gdn_slab(model,ws->norm_offset);
    float *qkv=require_alloc(2u*CONV_DIM*4,"inputs");
    float *alpha=require_alloc(2u*VALUE_HEADS*4,"alpha");
    float *beta=require_alloc(2u*VALUE_HEADS*4,"beta");
    float *gate=require_alloc(2u*VALUE_DIM*4,"gate");
    uint32_t pending=0,prefix=0,phase=0;
    bool previous=false;
    const char *pattern="RRRAARRRRRARARAAA";
    unsigned replays=0;
    for(unsigned round=0;round<32;round++) {
        ds4_qwen4exp_gdn_replay_step p=ds4_qwen4exp_gdn_replay_plan(
            true,previous,prefix,DS4_QWEN4EXP_GDN_REPLAY_ROWS,2,1,7,pending,pending);
        require_ok(!p.settle && p.active,"replay policy");
        if(p.swap) {
            ds4_gpu_tensor *old=b[1].state;b[1].state=checkpoint;checkpoint=old;
            replay.checkpoint=checkpoint;phase^=1;
        }
        prefix=p.prefix;
        for(unsigned i=0;i<2u*CONV_DIM;i++) qkv[i]=(float)(.8*sample(0x1000000ull+round*CONV_DIM*2u+i));
        for(unsigned i=0;i<2u*VALUE_HEADS;i++) {
            alpha[i]=(float)(2*sample(0x4000000ull+round*VALUE_HEADS*2u+i));
            beta[i]=(float)(2*sample(0x5000000ull+round*VALUE_HEADS*2u+i));
        }
        for(unsigned i=0;i<2u*VALUE_DIM;i++) gate[i]=(float)(1.5*sample(0x6000000ull+round*VALUE_DIM*2u+i));
        require_ok(ds4_gpu_qwen4exp_update_dpos(adopt,pending),"adopt upload");
        require_ok(ds4_gpu_qwen4exp_update_dpos(control,prefix),"prefix upload");
        for(unsigned j=0;j<2;j++) {
            require_ok(ds4_gpu_tensor_write(b[j].qkv,0,qkv,2u*CONV_DIM*4),"qkv");
            require_ok(ds4_gpu_tensor_write(b[j].alpha,0,alpha,2u*VALUE_HEADS*4),"alpha");
            require_ok(ds4_gpu_tensor_write(b[j].beta,0,beta,2u*VALUE_HEADS*4),"beta");
            require_ok(ds4_gpu_tensor_write(b[j].output_gate,0,gate,2u*VALUE_DIM*4),"gate");
        }
        require_ok(ds4_gpu_qwen4exp_gdn_adopt_q8(
            b[0].out,b[0].conv_state,b[0].state,csnap[0],snap[0],1,adopt,
            b[0].qkv,b[0].alpha,b[0].beta,b[0].output_gate,&cw,&aw,&dw,&nw,
            KEY_HEADS,VALUE_HEADS,2,ws->layout,QK_NORM_EPS,NORM_EPS,quant[0],0,soff),"baseline GDN");
        require_ok(ds4_gpu_synchronize(),"baseline synchronization");
        require_ok(ds4_gpu_begin_commands(),"candidate commands");
        ds4_decode_graph_key key={0};
        key.variant=ds4_qwen4exp_gdn_graph_variant(2,1,phase,true);
        key._pad=0x4752504cu;key.cur_hc=b[1].out;
        int status=captured?ds4_gpu_decode_graph_begin(&key):-1;
        if(status==1) replays++;
        else {
            require_ok(ds4_gpu_qwen4exp_gdn_replay_q8(
                b[1].out,b[1].conv_state,b[1].state,csnap[1],snap[1],1,adopt,
                b[1].qkv,b[1].alpha,b[1].beta,b[1].output_gate,&cw,&aw,&dw,&nw,
                KEY_HEADS,VALUE_HEADS,2,ws->layout,QK_NORM_EPS,NORM_EPS,quant[1],0,soff,&replay),"replay GDN");
            if(status==0) require_ok(ds4_gpu_decode_graph_end(&key)==0,"graph capture");
        }
        require_ok(ds4_gpu_end_commands() && ds4_gpu_synchronize(),"candidate synchronization");
        compare_tensor("output bits",b[0].out,b[1].out,2u*VALUE_DIM*4);
        compare_tensor("quantized output bytes",quant[0],quant[1],qb);
        compare_tensor("final recurrent state bits",b[0].state,b[1].state,sb);
        compare_tensor("final convolution bits",b[0].conv_state,b[1].conv_state,cb);
        compare_tensor("convolution snapshot bits",csnap[0],csnap[1],cb);
        require_ok(ds4_gpu_qwen4exp_gdn_replay_materialize(materialized,checkpoint,tape,
            prefix==DS4_QWEN4EXP_GDN_REPLAY_ROWS?0:prefix+1,
            KEY_HEADS,VALUE_HEADS,ws->layout),"materialize row zero");
        compare_tensor("row-zero snapshot bits",materialized,snap[0],sb);
        pending=pattern[round%16]=='R'?1:0;previous=true;
    }
    require_ok(!captured || replays>0,"captured path actually replayed");
    require_ok(!ds4_gpu_qwen4exp_gdn_replay_materialize(checkpoint,checkpoint,tape,0,
        KEY_HEADS,VALUE_HEADS,ws->layout),"checkpoint alias refused");
    require_ok(!ds4_gpu_qwen4exp_gdn_replay_materialize(materialized,checkpoint,tape,3,
        KEY_HEADS,VALUE_HEADS,ws->layout),"oversized replay refused");
    printf("PASS CUDA replay layout=%u captured=%u rounds=32 graph_replays=%u\n",ws->layout,captured,replays);
    ds4_gpu_decode_graphs_invalidate();
    free(qkv);free(alpha);free(beta);free(gate);
    ds4_gpu_tensor_free(checkpoint);ds4_gpu_tensor_free(tape);ds4_gpu_tensor_free(gates);
    ds4_gpu_tensor_free(control);ds4_gpu_tensor_free(adopt);ds4_gpu_tensor_free(materialized);
    for(unsigned j=0;j<2;j++) {
        buffers_free(&b[j]);ds4_gpu_tensor_free(snap[j]);
        ds4_gpu_tensor_free(csnap[j]);ds4_gpu_tensor_free(quant[j]);
    }
}

int main(void) {
    uint8_t *model=mmap(NULL,MODEL_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require_ok(model!=MAP_FAILED,"model mmap");memset(model,0,MODEL_BYTES);build_weights(model);
    require_ok(ds4_gpu_init() && ds4_gpu_qwen4exp_gdn_replay_supported(),"single CUDA GPU");
    require_ok(ds4_gpu_set_model_map(model,MODEL_BYTES),"synthetic model map");
    for(unsigned captured=0;captured<2;captured++) {
        replay_case(model,&g_grouped,captured);
        replay_case(model,&g_tiled,captured);
        replay_case(model,&g_fast_decay,captured);
    }
    munmap(model,MODEL_BYTES);return 0;
}
