/* Exact packed range projection versus the original two-call path.
 * Requires the CUDA library and generated Q8_0 rows, no model file. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static void need(int ok, const char *why) {
    if (!ok) { fprintf(stderr,"MTP static ranges: %s\n",why); exit(1); }
}
static void check(unsigned in, unsigned vocab, unsigned prefix, unsigned tail, unsigned offset) {
    const size_t groups=in/32, width=prefix+tail, bytes=offset+(size_t)vocab*groups*34;
    unsigned char *w=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    need(w!=MAP_FAILED,"weight allocation");
    for(size_t b=0;b<(size_t)vocab*groups;b++) {
        unsigned char *p=w+offset+b*34;
        p[0]=0;p[1]=(b&1)?0x98:0x18;
        for(unsigned j=2;j<34;j++)p[j]=(unsigned char)(b*37+j*19);
    }
    need(ds4_gpu_init(),"CUDA init");need(ds4_gpu_set_model_map(w,bytes),"weight registration");
    float *x=malloc(in*4),*ref=malloc((width+16)*4),*got=malloc((width+16)*4);
    ds4_gpu_tensor *xt=ds4_gpu_tensor_alloc(in*4),*yt=ds4_gpu_tensor_alloc((width+16)*4);
    ds4_gpu_tensor *pt=ds4_gpu_tensor_alloc(prefix*4),*tt=ds4_gpu_tensor_alloc(tail*4);
    need(x&&ref&&got&&xt&&yt&&pt&&tt,"buffers");
    for(unsigned pass=0;pass<4;pass++) {
        for(unsigned i=0;i<in;i++)x[i]=((int)((i+pass)%41)-20)*(pass==3?1e-30f:0.125f);
        for(unsigned i=0;i<width+16;i++)ref[i]=23.5f;
        need(ds4_gpu_tensor_write(xt,0,x,in*4),"activation write");
        need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(pt,w,bytes,offset,in,prefix,xt,1),"prefix oracle");
        need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tt,w,bytes,offset+(size_t)(vocab-tail)*groups*34,in,tail,xt,1),"tail oracle");
        need(ds4_gpu_tensor_read(pt,0,ref,prefix*4),"prefix read");
        need(ds4_gpu_tensor_read(tt,0,ref+prefix,tail*4),"tail read");
        for(unsigned i=0;i<width+16;i++)got[i]=23.5f;
        need(ds4_gpu_tensor_write(yt,0,got,(width+16)*4),"canaries");
        need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,in,vocab,prefix,tail,xt)==1,"range call");
        need(ds4_gpu_tensor_read(yt,0,got,(width+16)*4),"range read");
        need(!memcmp(ref,got,(width+16)*4),"complete packed rows/guards equality");
    }
    ds4_gpu_decode_graphs_invalidate();
    ds4_decode_graph_key key={.il=1,.island=0,.variant=76};
    need(ds4_gpu_decode_graph_begin(&key)==-1,"warmup");
    need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,in,vocab,prefix,tail,xt)==1,"warm range");
    need(ds4_gpu_decode_graph_begin(&key)==0,"capture");
    need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,in,vocab,prefix,tail,xt)==1,"capture range");
    need(ds4_gpu_decode_graph_end(&key)==0,"capture end");
    for(unsigned pass=0;pass<3;pass++) {
        for(unsigned i=0;i<in;i++)x[i]=((int)((i+pass)%17)-8)*0.25f;
        need(ds4_gpu_tensor_write(xt,0,x,in*4),"changed capture input");
        need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(pt,w,bytes,offset,in,prefix,xt,1),"replay prefix oracle");
        need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tt,w,bytes,offset+(size_t)(vocab-tail)*groups*34,in,tail,xt,1),"replay tail oracle");
        need(ds4_gpu_tensor_read(pt,0,ref,prefix*4),"replay prefix read");
        need(ds4_gpu_tensor_read(tt,0,ref+prefix,tail*4),"replay tail read");
        need(ds4_gpu_decode_graph_begin(&key)==1,"replay");
        need(ds4_gpu_tensor_read(yt,0,got,(width+16)*4),"replay read");
        need(!memcmp(ref,got,(width+16)*4),"changed-input replay equality");
    }
    need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,in,vocab,vocab,1,xt)==-1,"overlap refusal");
    need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,in,vocab,prefix,0,xt)==-1,"empty tail fallback");
    need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,in,vocab,0,tail,xt)==-1,"whole vocabulary fallback");
    need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,in+1,vocab,prefix,tail,xt)==-1,"partial group fallback");
    need(ds4_gpu_mtp_static_ranges(yt,w,bytes,offset,UINT64_MAX,vocab,prefix,tail,xt)==-1,"overflow guard");
    ds4_gpu_decode_graphs_invalidate();
    ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(yt);ds4_gpu_tensor_free(pt);ds4_gpu_tensor_free(tt);
    ds4_gpu_cleanup();free(x);free(ref);free(got);munmap(w,bytes);
}
int main(void) {
    need(setenv("DS4_CUDA_DECODE_GRAPHS","1",1)==0,"enable graphs");
    const unsigned shapes[][5]={{32,17,1,1,64},{96,37,3,7,66},{2560,1031,513,276,64},
        {2560,248320,98308,276,66}};
    /* PDL mode is cached once per process: run this executable separately
     * with the inherited kill switch set and unset. */
    for(unsigned j=0;j<sizeof(shapes)/sizeof(shapes[0]);j++)
        check(shapes[j][0],shapes[j][1],shapes[j][2],shapes[j][3],shapes[j][4]);
    puts("MTP static ranges: eager/captured original-row parity passed");return 0;
}
