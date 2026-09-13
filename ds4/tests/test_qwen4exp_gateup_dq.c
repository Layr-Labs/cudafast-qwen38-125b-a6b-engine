/* A/B byte test for the q4_K slice-parity weight staging valve and the q5_1
 * word-direct down staging valve, with and without packed-word prefetch.
 *
 * DS4_QWEN4EXP_NO_GATEUP_DQ stands the q4_K-specialised
 * qwen4exp_moe_gateup_mma_kernel staging down and runs the per-group staging
 * the kernel has always had.  DS4_QWEN4EXP_NO_DOWN_DQ does the same for the
 * routed qwen4exp_moe_down_mma_kernel's and the shared down tile's word-
 * direct q5_1 staging.  Every arm combination must produce identical bytes
 * through the complete routed MoE (and, for the shared leg, the shared
 * expert): every output buffer including guards, with the inputs checked
 * for mutation besides.  There is no reference -- the comparison is arm
 * against arm.
 *
 * Coverage the arm difference needs: the specialised Q4_K tile on both
 * sides of the pair-task boundary (the 144-byte rows above 64 tokens, the
 * 132-byte rows below, whose group slots are not sixteen-byte aligned and
 * keep the word stores), widths with odd expert tails, K wide enough to
 * reach a second super-block so both halves of the packed scale split run,
 * a slab whose rows are NOT sixteen-byte aligned -- there the valve's fast
 * arm stands itself down and the kernel must fall back to its per-group
 * staging on its own -- and the formats the valves cannot touch (Q8_0 and
 * the generic Q5_K instantiation), which still have to agree with
 * themselves.  Synthetic weights only.  CUDA host; build and run with
 *   make test-qwen4exp-gateup-dq
 */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { D = 256, O = 256, CAP = 1024, GUARD = 64 };
static const char *const valve = "DS4_QWEN4EXP_NO_GATEUP_DQ";
static const char *const valve2 = "DS4_QWEN4EXP_NO_DOWN_DQ";
static const char *const prefetch_valve = "DS4_QWEN4EXP_NO_DOWN_MMA_PREFETCH";
static uint32_t rng = 0x6b13c7a5u;
static unsigned comparisons;
static uint32_t random_word(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "gate/up dq valve A/B: %s\n", what); exit(1); }
}
static unsigned block_bytes(unsigned type) {
    return type == 7 ? 24 : type == 8 ? 34 : type == 12 ? 144 :
           type == 13 ? 176 : 210;
}
static unsigned row_bytes(unsigned type, unsigned k) {
    return block_bytes(type) * ((type == 7 || type == 8) ? k / 32 : k / 256);
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

static void check_case(unsigned gt, unsigned dt, unsigned experts,
                       unsigned alignment, unsigned k) {
    const unsigned used = experts == 17 ? 3 : 10;
    const unsigned stride = used * D + 16;
    const size_t gb = (size_t)experts * D * row_bytes(gt, k);
    const size_t db = (size_t)experts * O * row_bytes(dt, k);
    const size_t go = 64 + alignment, uo = go + gb + 64, d_o = uo + gb + 64;
    const size_t image_bytes = d_o + db;
    unsigned char *image = mmap(NULL, image_bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(image != MAP_FAILED, "model allocation");
    fill_weights(image + go, gb, gt); fill_weights(image + uo, gb, gt);
    fill_weights(image + d_o, db, dt);
    ds4_gpu_qwen4exp_slab gate = {image, image_bytes, go, D * row_bytes(gt, k), row_bytes(gt, k), gt};
    ds4_gpu_qwen4exp_slab up = gate; up.offset = uo;
    ds4_gpu_qwen4exp_slab down = {image, image_bytes, d_o, O * row_bytes(dt, k), row_bytes(dt, k), dt};
    require(ds4_gpu_init() && ds4_gpu_set_model_map(image, image_bytes), "model registration");

    const size_t xb = (size_t)CAP * k * 4, rb = (size_t)CAP * used * 4;
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
    ds4_gpu_tensor *out[3]; unsigned char *expected[3], *got[3], *poison[3];
    require(xp && x && routes && weights && xh && rh && wh && input_check, "inputs");
    for (unsigned j = 0; j < 3; j++) {
        expected[j] = malloc(bytes[j]); got[j] = malloc(bytes[j]); poison[j] = malloc(bytes[j]);
        require(expected[j] && got[j] && poison[j], "host outputs");
        memset(poison[j], 0x5a, bytes[j]);
        out[j] = ds4_gpu_tensor_alloc(bytes[j]);
        require(out[j] != NULL, "device outputs");
    }
#define INVOKE(W) require(ds4_gpu_qwen4exp_routed_moe_tensor(out[0],out[1],out[2], \
    &gate,&up,&down,k,D,O,routes,weights,experts,used,x,W,stride),"routed MoE")
#define POISON() do { for(unsigned pj=0;pj<3;pj++) \
    require(ds4_gpu_tensor_write(out[pj],0,poison[pj],bytes[pj]),"output poison"); } while(0)
/* GU/DN/PF = 1 disables that optimization. Each leg explicitly selects
 * both decoders and the optional next-chunk packed-word register prefetch. */
#define LEG(GU, DN, PF) do { \
    if ((GU)) require(setenv(valve, "1", 1) == 0, "valve dispatch"); \
    else require(unsetenv(valve) == 0, "valve dispatch"); \
    if ((DN)) require(setenv(valve2, "1", 1) == 0, "valve dispatch"); \
    else require(unsetenv(valve2) == 0, "valve dispatch"); \
    if ((PF)) require(setenv(prefetch_valve, "1", 1) == 0, "prefetch dispatch"); \
    else require(unsetenv(prefetch_valve) == 0, "prefetch dispatch"); \
    POISON(); INVOKE(width); \
} while (0)
    const unsigned widths[] = {8, 63, 64, 65, 257, CAP};
    const float scale[] = {0.2f, 1e-30f, 1e6f, 0.0f};
    for (unsigned wi = 0; wi < sizeof(widths)/sizeof(widths[0]); wi++) {
        const unsigned width = widths[wi];
        for (unsigned trial = 0; trial < 4; trial++) {
            memset(xh, 0xa6, xp_bytes); memset(rh, 0xb7, rb + GUARD); memset(wh, 0xc8, rb + GUARD);
            float *values = (float *)(xh + xp_offset), *w = (float *)wh;
            int32_t *ids = (int32_t *)rh;
            for (unsigned t = 0; t < CAP; t++) {
                for (unsigned kk = 0; kk < k; kk++) values[(size_t)t*k+kk] =
                    ((int)(random_word()%2049)-1024) * (scale[trial%4]/1024.0f);
                for (unsigned slot = 0; slot < used; slot++) {
                    int e = (int)((t * 7 + slot) % experts);
                    if (trial % 3 == 1) e = (int)(experts - 1 - slot);
                    if (trial % 3 == 2 && (t + slot) % 3 == 0) e = -1;
                    ids[(size_t)t*used+slot] = e;
                    w[(size_t)t*used+slot] = 0.05f + (float)((t+slot)%13)*0.005f;
                }
            }
            require(ds4_gpu_tensor_write(xp,0,xh,xp_bytes) && ds4_gpu_tensor_write(routes,0,rh,rb+GUARD) &&
                    ds4_gpu_tensor_write(weights,0,wh,rb+GUARD), "input upload");
            /* Original staging is the oracle. Compare word-direct staging
             * alone and with prefetch, including its unaligned fallback. */
            LEG(1, 1, 1);
            for (unsigned j = 0; j < 3; j++)
                require(ds4_gpu_tensor_read(out[j],0,expected[j],bytes[j]), "baseline arm output");
            const unsigned legs[][3] = {
                {0,1,1}, {1,0,1}, {0,0,1}, {1,0,0}, {0,0,0}
            };
            for (unsigned li = 0; li < sizeof(legs)/sizeof(legs[0]); li++) {
                LEG(legs[li][0], legs[li][1], legs[li][2]);
                for (unsigned j = 0; j < 3; j++) {
                    require(ds4_gpu_tensor_read(out[j],0,got[j],bytes[j]), "arm output");
                    if (memcmp(expected[j],got[j],bytes[j])) {
                        fprintf(stderr,"gt=%u dt=%u experts=%u align=%u k=%u width=%u trial=%u leg=%u buffer=%u\n",
                                gt,dt,experts,alignment,k,width,trial,li,j);
                        require(0,"complete output/guard mismatch between the arms");
                    }
                    comparisons++;
                }
            }
            require(ds4_gpu_tensor_read(xp,0,input_check,xp_bytes) && !memcmp(xh,input_check,xp_bytes), "activation/guard mutation");
            require(ds4_gpu_tensor_read(routes,0,input_check,rb+GUARD) && !memcmp(rh,input_check,rb+GUARD), "route/guard mutation");
            require(ds4_gpu_tensor_read(weights,0,input_check,rb+GUARD) && !memcmp(wh,input_check,rb+GUARD), "weight/guard mutation");
        }
    }
#undef POISON
#undef INVOKE
#undef LEG
    for (unsigned j = 0; j < 3; j++) {
        ds4_gpu_tensor_free(out[j]); free(expected[j]); free(got[j]); free(poison[j]);
    }
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(routes); ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(xp);
    ds4_gpu_cleanup(); munmap(image,image_bytes); free(xh); free(rh); free(wh); free(input_check);
    printf("gate/up dq valve A/B gt=%u dt=%u experts=%u offset=%u k=%u PASS\n",
           gt,dt,experts,64+alignment,k); fflush(stdout);
}

/* The shared expert's down tile takes the same DS4_QWEN4EXP_NO_DOWN_DQ
 * valve over the same word-direct decode.  The shipped shared expert is
 * Q8_0, where neither arm differs, so this drives a Q5_1 shared expert
 * synthetically -- the one artifact shape whose shared down the new staging
 * can reach.  Widths of 64 tokens and up let the class tile take the call
 * unforced.  The gate/up valve is inert on this path (no routed kernels);
 * it is still left in a defined state per leg. */
static void shared_case(unsigned k) {
    const size_t gb = (size_t)D * row_bytes(7, k);
    const size_t db = (size_t)O * row_bytes(7, k);
    const size_t ro = 64, go = ro + (size_t)k * 4 + 64, uo = go + gb + 64,
                 d_o = uo + gb + 64;
    const size_t image_bytes = d_o + db;
    unsigned char *image = mmap(NULL, image_bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(image != MAP_FAILED, "shared model allocation");
    {
        float *router = (float *)(image + ro);
        for (unsigned i = 0; i < k; i++) router[i] = 0.01f * (float)(i % 37) - 0.18f;
    }
    fill_weights(image + go, gb, 7); fill_weights(image + uo, gb, 7);
    fill_weights(image + d_o, db, 7);
    ds4_gpu_qwen4exp_slab router = {image, image_bytes, ro, 0, k * 4u, 0};
    ds4_gpu_qwen4exp_slab gate = {image, image_bytes, go, 0, row_bytes(7, k), 7};
    ds4_gpu_qwen4exp_slab up = gate; up.offset = uo;
    ds4_gpu_qwen4exp_slab down = {image, image_bytes, d_o, 0, row_bytes(7, k), 7};
    require(ds4_gpu_init() && ds4_gpu_set_model_map(image, image_bytes),
            "shared model registration");

    const size_t xp_bytes = (size_t)CAP * k * 4 + 2 * GUARD;
    const size_t bytes[3] = {(size_t)CAP * O * 4 + GUARD,
                             (size_t)CAP * D * 4 + GUARD,
                             (size_t)CAP * 4 + GUARD};
    ds4_gpu_tensor *xp = ds4_gpu_tensor_alloc(xp_bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_view(xp, GUARD, (size_t)CAP * k * 4);
    ds4_gpu_tensor *t[3];
    unsigned char *expected[3], *got[3], *poison[3];
    require(xp && x, "shared inputs");
    for (unsigned j = 0; j < 3; j++) {
        expected[j] = malloc(bytes[j]); got[j] = malloc(bytes[j]);
        poison[j] = malloc(bytes[j]);
        require(expected[j] && got[j] && poison[j], "shared host outputs");
        memset(poison[j], 0x5a, bytes[j]);
        t[j] = ds4_gpu_tensor_alloc(bytes[j]);
        require(t[j] != NULL, "shared device outputs");
    }
#define SH_POISON() do { for(unsigned pj=0;pj<3;pj++) \
    require(ds4_gpu_tensor_write(t[pj],0,poison[pj],bytes[pj]),"output poison"); } while(0)
#define SH_INVOKE(W) require(ds4_gpu_qwen4exp_shared_expert_tensor( \
    t[0],t[1],t[2],&router,&gate,&up,&down,k,D,O,x,W),"shared expert")
#define SH_LEG(DN) do { \
    require(setenv(valve, "1", 1) == 0, "valve dispatch"); \
    if ((DN)) require(setenv(valve2, "1", 1) == 0, "valve dispatch"); \
    else require(unsetenv(valve2) == 0, "valve dispatch"); \
    SH_POISON(); SH_INVOKE(width); \
} while (0)
    const unsigned widths[] = {64, 65, 257};
    unsigned char *xh = malloc(xp_bytes), *input_check = malloc(xp_bytes);
    require(xh && input_check, "shared activation");
    for (unsigned wi = 0; wi < sizeof(widths)/sizeof(widths[0]); wi++) {
        const unsigned width = widths[wi];
        for (unsigned trial = 0; trial < 2; trial++) {
            memset(xh, 0xa6, xp_bytes);
            float *values = (float *)(xh + GUARD);
            for (size_t at = 0; at < (size_t)CAP * k; at++)
                values[at] = ((int)(random_word()%2049)-1024) *
                             ((trial ? 1e6f : 0.2f)/1024.0f);
            require(ds4_gpu_tensor_write(xp,0,xh,xp_bytes), "shared input upload");
            SH_LEG(1);
            for (unsigned j = 0; j < 3; j++)
                require(ds4_gpu_tensor_read(t[j],0,expected[j],bytes[j]), "shared baseline");
            SH_LEG(0);
            for (unsigned j = 0; j < 3; j++) {
                require(ds4_gpu_tensor_read(t[j],0,got[j],bytes[j]), "shared arm output");
                if (memcmp(expected[j],got[j],bytes[j])) {
                    fprintf(stderr,"shared q5_1 k=%u width=%u trial=%u buffer=%u\n",
                            k,width,trial,j);
                    require(0,"shared output/guard mismatch between the arms");
                }
                comparisons++;
            }
            require(ds4_gpu_tensor_read(xp,0,input_check,xp_bytes) &&
                    !memcmp(xh,input_check,xp_bytes), "shared activation mutation");
        }
    }
    free(xh); free(input_check);
#undef SH_POISON
#undef SH_INVOKE
#undef SH_LEG
    for (unsigned j = 0; j < 3; j++) {
        ds4_gpu_tensor_free(t[j]); free(expected[j]); free(got[j]); free(poison[j]);
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(xp);
    ds4_gpu_cleanup(); munmap(image,image_bytes);
    printf("shared down dq valve A/B k=%u PASS\n", k); fflush(stdout);
}

int main(void) {
    require(setenv("DS4_CUDA_COPY_MODEL","1",1) == 0, "test environment");
    /* Q4_K gate/up over a Q5_1 down is the ranked shape.  k 512 reaches a
     * second super-block, so both halves of the packed scale split and both
     * payload slices of a super-block run.  offset 66 leaves the q4_K rows
     * two mod sixteen: the fast arm must stand itself down there and the
     * kernel falls back to its per-group staging, in the same specialised
     * instantiation.  Q8_0 and Q5_K never take the new staging and only
     * check that the valve leaves them alone. */
    check_case(12, 7, 512, 0, 256);
    check_case(12, 7, 512, 0, 512);
    check_case(12, 7, 512, 2, 512);
    check_case(12, 7, 17, 0, 256);
    check_case(8, 8, 512, 0, 256);
    check_case(13, 8, 512, 0, 256);
    shared_case(256);
    shared_case(512);
    printf("gate/up + down dq/prefetch A/B: %u complete-buffer comparisons PASS\n",comparisons);
    return 0;
}
