/* Complete MoE comparison for the gate/up pair scheduler, including graph
 * replays, output guards, invalid routes and the shared-scratch growth seam.
 * Synthetic weights only. Build against the normal CUDA library:
 * cc -O2 -std=c11 -D_GNU_SOURCE -Ids4 ds4/tests/test_qwen4exp_moe_pair_tasks.c \
 *   -L.build/ds4 -lds4qwen -lm -o /tmp/test-moe-pair-tasks
 */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { K = 256, D = 256, O = 256, CAP = 1024, GUARD = 64 };
static const char *disable = "DS4_QWEN4EXP_NO_GU_PAIR_TASKS";
static uint32_t rng = 0xa231e599u;
static unsigned comparisons, graph_checks, actual_replays;
static uint32_t random_word(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "MoE pair tasks: %s\n", what); exit(1); }
}
static unsigned block_bytes(unsigned type) {
    return type == 7 ? 24 : type == 8 ? 34 : type == 12 ? 144 :
           type == 13 ? 176 : 210;
}
static unsigned row_bytes(unsigned type) {
    return block_bytes(type) * ((type == 7 || type == 8) ? K / 32 : K / 256);
}
static void fill_weights(unsigned char *p, size_t bytes, unsigned type) {
    const unsigned bs = block_bytes(type);
    for (size_t at = 0; at < bytes; at += bs) {
        for (unsigned j = 0; j < bs; j++) p[at + j] = (unsigned char)random_word();
        const uint16_t scale = 0x1400, minimum = 0x1000;
        memcpy(p + at + (type == 14 ? bs - 2 : 0), &scale, 2);
        if (type == 7 || type == 12 || type == 13) memcpy(p + at + 2, &minimum, 2);
    }
}

static void check_case(unsigned gt, unsigned dt, unsigned experts, unsigned alignment) {
    const unsigned used = experts == 1 ? 1 : experts == 17 ? 3 : 10;
    const unsigned stride = used * D + 16;
    const size_t gb = (size_t)experts * D * row_bytes(gt);
    const size_t db = (size_t)experts * O * row_bytes(dt);
    const size_t go = 64 + alignment, uo = go + gb + 64, d_o = uo + gb + 64;
    const size_t image_bytes = d_o + db;
    unsigned char *image = mmap(NULL, image_bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(image != MAP_FAILED, "model allocation");
    fill_weights(image + go, gb, gt); fill_weights(image + uo, gb, gt);
    fill_weights(image + d_o, db, dt);
    ds4_gpu_qwen4exp_slab gate = {image, image_bytes, go, D * row_bytes(gt), row_bytes(gt), gt};
    ds4_gpu_qwen4exp_slab up = gate; up.offset = uo;
    ds4_gpu_qwen4exp_slab down = {image, image_bytes, d_o, O * row_bytes(dt), row_bytes(dt), dt};
    require(ds4_gpu_init() && ds4_gpu_set_model_map(image, image_bytes), "model registration");

    const size_t xb = (size_t)CAP * K * 4, rb = (size_t)CAP * used * 4;
    const size_t xp_offset = GUARD + (alignment ? 4 : 0), xp_bytes = xb + 2 * GUARD;
    const size_t bytes[3] = {(size_t)CAP * O * 4 + GUARD,
                            (size_t)CAP * stride * 4 + GUARD,
                            (size_t)CAP * used * O * 4 + GUARD};
    ds4_gpu_tensor *xp = ds4_gpu_tensor_alloc(xp_bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_view(xp, xp_offset, xb);
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(rb + GUARD);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(rb + GUARD);
    unsigned char *xh = malloc(xp_bytes), *rh = malloc(rb + GUARD), *wh = malloc(rb + GUARD);
    unsigned char *input_check = malloc(xp_bytes > rb + GUARD ? xp_bytes : rb + GUARD);
    ds4_gpu_tensor *out[2][3]; unsigned char *expected[3], *got[3], *poison[3];
    require(xp && x && routes && weights && xh && rh && wh && input_check, "inputs");
    for (unsigned j = 0; j < 3; j++) {
        expected[j] = malloc(bytes[j]); got[j] = malloc(bytes[j]); poison[j] = malloc(bytes[j]);
        require(expected[j] && got[j] && poison[j], "host outputs");
        memset(poison[j], 0x5a, bytes[j]);
        for (unsigned m = 0; m < 2; m++) { out[m][j] = ds4_gpu_tensor_alloc(bytes[j]); require(out[m][j] != NULL, "device outputs"); }
    }
#define INVOKE(M,W) require(ds4_gpu_qwen4exp_routed_moe_tensor(out[M][0],out[M][1],out[M][2], \
    &gate,&up,&down,K,D,O,routes,weights,experts,used,x,W,stride),"routed MoE")
#define POISON(M) do { for(unsigned pj=0;pj<3;pj++) \
    require(ds4_gpu_tensor_write(out[M][pj],0,poison[pj],bytes[pj]),"output poison"); } while(0)
    const unsigned widths[] = {63, 64, 65, 257, CAP};
    const float scale[] = {0.2f, 1e-30f, 1e6f, 0.0f};
    for (unsigned wi = 0; wi < sizeof(widths)/sizeof(widths[0]); wi++) {
        const unsigned width = widths[wi];
        ds4_decode_graph_key keys[2]; memset(keys, 0, sizeof(keys));
        for (unsigned m = 0; m < 2; m++) { keys[m].il = m; keys[m].cur_hc = x; keys[m].after_attn_hc = out[m][0]; }
        ds4_gpu_decode_graphs_invalidate();
        for (unsigned trial = 0; trial < 10; trial++) {
            memset(xh, 0xa6, xp_bytes); memset(rh, 0xb7, rb + GUARD); memset(wh, 0xc8, rb + GUARD);
            float *values = (float *)(xh + xp_offset), *w = (float *)wh;
            int32_t *ids = (int32_t *)rh;
            for (unsigned t = 0; t < CAP; t++) {
                for (unsigned k = 0; k < K; k++) values[(size_t)t*K+k] =
                    ((int)(random_word()%2049)-1024) * (scale[trial%4]/1024.0f);
                for (unsigned slot = 0; slot < used; slot++) {
                    int e = (int)((t * 7 + slot) % experts);
                    if (trial % 5 == 1) e = (int)(experts - 1 - slot);
                    if (trial % 5 == 2 && (t + slot) % 3 == 0) e = -1;
                    if (trial % 5 == 3) e = -1;
                    ids[(size_t)t*used+slot] = e;
                    w[(size_t)t*used+slot] = 0.05f + (float)((t+slot)%13)*0.005f;
                }
            }
            require(ds4_gpu_tensor_write(xp,0,xh,xp_bytes) && ds4_gpu_tensor_write(routes,0,rh,rb+GUARD) &&
                    ds4_gpu_tensor_write(weights,0,wh,rb+GUARD), "input upload");
            if (!trial) {
                /* Grow the shared pool before either graph is captured. */
                require(unsetenv(disable) == 0, "enable warmup"); POISON(1); INVOKE(1,width);
            }
            require(setenv(disable,"1",1) == 0, "reference dispatch"); POISON(0); INVOKE(0,width);
            for (unsigned j = 0; j < 3; j++) require(ds4_gpu_tensor_read(out[0][j],0,expected[j],bytes[j]), "reference output");
            for (unsigned m = 0; m < 2; m++) {
                require((m ? unsetenv(disable) : setenv(disable,"1",1)) == 0, "graph dispatch");
                POISON(m);
                const int state = ds4_gpu_decode_graph_begin(&keys[m]);
                if (trial >= 2) require(state == 1, "actual graph replay");
                if (state != 1) {
                    require(state == -1 || state == 0, "graph state"); INVOKE(m,width);
                    if (state == 0) require(ds4_gpu_decode_graph_end(&keys[m]) == 0, "graph capture");
                } else actual_replays++;
                for (unsigned j = 0; j < 3; j++) {
                    require(ds4_gpu_tensor_read(out[m][j],0,got[j],bytes[j]), "candidate output");
                    if (memcmp(expected[j],got[j],bytes[j])) {
                        fprintf(stderr,"gt=%u dt=%u experts=%u align=%u width=%u trial=%u mode=%u buffer=%u\n",
                                gt,dt,experts,alignment,width,trial,m,j);
                        require(0,"complete output/guard mismatch");
                    }
                    comparisons++;
                }
                graph_checks++;
            }
            require(ds4_gpu_tensor_read(xp,0,input_check,xp_bytes) && !memcmp(xh,input_check,xp_bytes), "activation/guard mutation");
            require(ds4_gpu_tensor_read(routes,0,input_check,rb+GUARD) && !memcmp(rh,input_check,rb+GUARD), "route/guard mutation");
            require(ds4_gpu_tensor_read(weights,0,input_check,rb+GUARD) && !memcmp(wh,input_check,rb+GUARD), "weight/guard mutation");
        }
    }
#undef POISON
#undef INVOKE
    ds4_gpu_decode_graphs_invalidate();
    for (unsigned j = 0; j < 3; j++) { for (unsigned m = 0; m < 2; m++) ds4_gpu_tensor_free(out[m][j]); free(expected[j]); free(got[j]); free(poison[j]); }
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(routes); ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(xp);
    ds4_gpu_cleanup(); munmap(image,image_bytes); free(xh); free(rh); free(wh); free(input_check);
    printf("MoE pair tasks gt=%u dt=%u experts=%u offset=%u PASS\n",gt,dt,experts,64+alignment); fflush(stdout);
}

int main(void) {
    require(setenv("DS4_CUDA_COPY_MODEL","1",1) == 0 && setenv("DS4_CUDA_DECODE_GRAPHS","1",1) == 0, "test environment");
    const unsigned types[][2] = {{12,7},{13,8},{8,8},{12,14},{14,7}};
    for (unsigned t = 0; t < sizeof(types)/sizeof(types[0]); t++)
        for (unsigned a = 0; a < 2; a++) check_case(types[t][0],types[t][1],512,a*2);
    check_case(12,7,1,0); check_case(13,8,17,2);
    printf("MoE pair tasks: %u complete buffers, %u graph checks, %u actual replays PASS\n",
           comparisons,graph_checks,actual_replays);
    return 0;
}
