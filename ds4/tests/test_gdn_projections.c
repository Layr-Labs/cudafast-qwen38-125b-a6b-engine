/* GDN projection fusion: independent mappings, complete outputs and changed graphs. */
#define _GNU_SOURCE
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { SETS=4, CAP=8, MODES=2 };
static uint32_t rng=0x771ba358u;
static uint32_t word(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void must(int ok,const char *s){if(!ok){fprintf(stderr,"GDN four projection: %s\n",s);exit(1);}}
static void *alloc(size_t n){void *p=malloc(n);must(p!=NULL,"host allocation");return p;}
static uint64_t in_dim,od[4],groups,mb[4],offsets[4][SETS];
static size_t xb,qb,yb[4];
static unsigned char *maps[4],*xhost[SETS],*qref[SETS],*refs[SETS][4],*poison;
static ds4_gpu_tensor *xt[SETS],*qt[SETS],*yt[MODES][SETS][4];
static void inputs(unsigned rows,float scale) {
    unsigned char *qp=alloc(qb);memset(qp,0xa5,qb);
    for(unsigned i=0;i<SETS;i++) {
        float *x=(float *)xhost[i];
        for(size_t j=0;j<xb/4;j++)x[j]=((int)(word()%2001)-1000)*scale*.001f;
        must(ds4_gpu_tensor_write(xt[i],0,xhost[i],xb),"write input");
        must(ds4_gpu_tensor_write(qt[i],0,qp,qb),"quant poison");
        must(ds4_gpu_quantize_q8_0_decode_rows_exact_tensor(qt[i],0,(uint64_t)rows*groups*32,xt[i],in_dim,rows),"quantize shared input");
        must(ds4_gpu_tensor_read(qt[i],0,qref[i],qb),"quant reference");
    }
    free(qp);
}
static void reset(unsigned mode) {
    for(unsigned i=0;i<SETS;i++)for(unsigned b=0;b<4;b++)
        must(ds4_gpu_tensor_write(yt[mode][i][b],0,poison,yb[b]),"reset complete output");
}
static int four(ds4_gpu_tensor *const outs[4],unsigned i,unsigned rows) {
    const void *wm[4]={maps[0],maps[1],maps[2],maps[3]};
    uint64_t off[4];for(unsigned b=0;b<4;b++)off[b]=offsets[b][i];
    return ds4_gpu_qwen4exp_gdn_projections_exact_tensor(outs,wm,mb,off,
        in_dim,od[0],od[1],xt[i],qt[i],0,(uint64_t)rows*groups*32u,rows);
}
static void operate(unsigned mode,unsigned rows) {
    for(unsigned i=0;i<SETS;i++) {
        if(mode)must(four(yt[mode][i],i,rows),"four projections");
        else {
            for(unsigned b=0;b<2;b++)must(ds4_gpu_matmul_q8_0_preq_rows_exact_tensor(
                yt[mode][i][b],maps[b],mb[b],offsets[b][i],in_dim,od[b],
                qt[i],0,(uint64_t)rows*groups*32u,rows),"original Q8");
            for(unsigned b=2;b<4;b++)must(ds4_gpu_matmul_f32_decode_rows_exact_tensor(
                yt[mode][i][b],maps[b],mb[b],offsets[b][i],in_dim,od[b],xt[i],rows),"original F32");
        }
    }
}
static unsigned compare(unsigned mode) {
    for(unsigned i=0;i<SETS;i++)for(unsigned b=0;b<4;b++) {
        if(!mode)must(ds4_gpu_tensor_read(yt[mode][i][b],0,refs[i][b],yb[b]),"read reference");
        else {
            unsigned char *got=alloc(yb[b]);
            must(ds4_gpu_tensor_read(yt[mode][i][b],0,got,yb[b]),"read result");
            must(memcmp(got,refs[i][b],yb[b])==0,"complete output and tail equality");free(got);
        }
    }
    for(unsigned i=0;i<SETS;i++) {
        unsigned char *got=alloc(xb>qb?xb:qb);
        must(ds4_gpu_tensor_read(xt[i],0,got,xb),"input readback");
        must(memcmp(got,xhost[i],xb)==0,"input immutable");
        must(ds4_gpu_tensor_read(qt[i],0,got,qb),"quant readback");
        must(memcmp(got,qref[i],qb)==0,"quant immutable");free(got);
    }
    return mode?SETS*4:0;
}
static int graph(unsigned mode,unsigned rows,ds4_decode_graph_key *key) {
    int s=ds4_gpu_decode_graph_begin(key);must(s>=0,"graph begin");
    if(!s){operate(mode,rows);must(ds4_gpu_decode_graph_end(key)==0,"graph end");}
    return s;
}
static void check_shape(unsigned in,unsigned out0,unsigned out1,unsigned offset) {
    in_dim=in;od[0]=out0;od[1]=out1;od[2]=od[3]=48;groups=in/32;
    xb=CAP*in*4u+64;qb=CAP*groups*36u+64;
    for(unsigned b=0;b<2;b++) {
        const uint64_t qs=(od[b]*groups*34u+63u)&~63ull;
        const uint64_t fs=(48u*in*4u+63u)&~63ull;
        const uint64_t fo=(offset+SETS*qs+63u)&~63ull;
        mb[b]=mb[b+2]=fo+SETS*fs+64u;
        maps[b]=mmap(NULL,mb[b],PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        must(maps[b]!=MAP_FAILED,"weight map");maps[b+2]=maps[b];
        for(unsigned i=0;i<SETS;i++) {
            offsets[b][i]=offset+i*qs;offsets[b+2][i]=fo+i*fs;
            for(uint64_t g=0;g<od[b]*groups;g++) {
                unsigned char *p=maps[b]+offsets[b][i]+g*34u;
                p[0]=0;p[1]=(g&1)?0x98:0x18;
                for(unsigned j=2;j<34;j++)p[j]=(unsigned char)word();
            }
            float *fw=(float *)(maps[b+2]+offsets[b+2][i]);
            for(uint64_t j=0;j<48u*in;j++)fw[j]=((int)(word()%2001)-1000)*0.0002f;
        }
    }
    for(unsigned b=0;b<4;b++)yb[b]=CAP*od[b]*4u+64;
    must(ds4_gpu_init(),"GPU init");
    must(ds4_gpu_set_model_map(maps[0],mb[0]),"first mapping");
    must(ds4_gpu_set_aux_model_map_range(maps[1],mb[1],0,mb[1]),"second mapping");
    size_t maxy=yb[0]>yb[1]?yb[0]:yb[1];poison=alloc(maxy);memset(poison,0xa5,maxy);
    for(unsigned i=0;i<SETS;i++) {
        xt[i]=ds4_gpu_tensor_alloc(xb);qt[i]=ds4_gpu_tensor_alloc(qb);
        must(xt[i]&&qt[i],"input allocation");xhost[i]=alloc(xb);qref[i]=alloc(qb);
        for(unsigned b=0;b<4;b++) {
            refs[i][b]=alloc(yb[b]);
            for(unsigned m=0;m<MODES;m++){yt[m][i][b]=ds4_gpu_tensor_alloc(yb[b]);must(yt[m][i][b]!=NULL,"output allocation");}
        }
    }
    inputs(1,.2f);reset(0);operate(0,1);compare(0);
    for(unsigned a=0;a<4;a++)for(unsigned b=a+1;b<4;b++) {
        ds4_gpu_tensor *outs[4];for(unsigned c=0;c<4;c++)outs[c]=yt[1][0][c];
        outs[b]=outs[a];must(!four(outs,0,1),"same outputs rejected");
        ds4_gpu_tensor *over=ds4_gpu_tensor_view(outs[a],4,yb[a]-4);
        must(over!=NULL,"overlap view");outs[b]=over;
        must(!four(outs,0,1),"partial output overlap rejected");ds4_gpu_tensor_free(over);
    }
    ds4_gpu_tensor *outs[4];for(unsigned c=0;c<4;c++)outs[c]=yt[1][0][c];
    outs[2]=xt[0];must(!four(outs,0,1),"float input overlap rejected");
    outs[2]=qt[0];must(!four(outs,0,1),"quant input overlap rejected");
    size_t sum=0;for(unsigned b=0;b<4;b++)sum+=od[b]*4u;
    ds4_gpu_tensor *adj=ds4_gpu_tensor_alloc(sum+64);must(adj!=NULL,"adjacent allocation");
    must(ds4_gpu_tensor_write(adj,0,poison,sum+64),"adjacent poison");
    size_t pos=0;for(unsigned b=0;b<4;b++){
        outs[b]=ds4_gpu_tensor_view(adj,pos,od[b]*4u);must(outs[b]!=NULL,"adjacent view");pos+=od[b]*4u;
    }
    must(four(outs,0,1),"adjacent outputs accepted");
    unsigned char *got=alloc(sum+64);must(ds4_gpu_tensor_read(adj,0,got,sum+64),"adjacent readback");
    pos=0;for(unsigned b=0;b<4;b++){
        must(memcmp(got+pos,refs[0][b],od[b]*4u)==0,"adjacent exact");pos+=od[b]*4u;ds4_gpu_tensor_free(outs[b]);
    }
    must(memcmp(got+sum,poison,64)==0,"adjacent tail intact");free(got);ds4_gpu_tensor_free(adj);
    const unsigned widths[]={1,2,3,7};const float scales[]={.2f,1e-20f,8.0f,0.0f};
    unsigned eager=0,captured=0,replayed=0;
    for(unsigned wi=0;wi<4;wi++) {
        unsigned rows=widths[wi];ds4_gpu_decode_graphs_invalidate();
        ds4_decode_graph_key keys[MODES]={{.il=1},{.il=2}};
        for(unsigned trial=0;trial<4;trial++) {
            inputs(rows,scales[trial]);
            for(unsigned m=0;m<MODES;m++){reset(m);operate(m,rows);eager+=compare(m);}
            for(unsigned m=0;m<MODES;m++) {
                if(trial==0){must(ds4_gpu_decode_graph_begin(&keys[m])==-1,"warm capture");operate(m,rows);}
                reset(m);int state=graph(m,rows,&keys[m]);
                must(state==(trial?1:0),"actual capture/replay");if(trial)replayed++;
                captured+=compare(m);
            }
        }
    }
    printf("GDN_FOUR_CHECK in=%u out0=%u out1=%u offset=%u sets=4 eager_buffers=%u graph_buffers=%u changed_replays=%u PASS\n",in,out0,out1,offset,eager,captured,replayed);fflush(stdout);
    ds4_gpu_decode_graphs_invalidate();
    for(unsigned i=0;i<SETS;i++) {
        ds4_gpu_tensor_free(xt[i]);ds4_gpu_tensor_free(qt[i]);free(xhost[i]);free(qref[i]);
        for(unsigned b=0;b<4;b++) {
            free(refs[i][b]);for(unsigned m=0;m<MODES;m++)ds4_gpu_tensor_free(yt[m][i][b]);
        }
    }
    ds4_gpu_cleanup();for(unsigned b=0;b<2;b++)munmap(maps[b],mb[b]);free(poison);
}
int main(void) {
    setenv("DS4_CUDA_DECODE_GRAPHS","1",1);
    check_shape(2560,10240,6144,64);
    check_shape(2560,10240,6144,66);
    check_shape(2560,10243,6147,64);
    puts("GDN_FOUR_ALL PASS");return 0;
}
