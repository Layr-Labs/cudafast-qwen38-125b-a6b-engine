/* The actual graph prepare/select/copy/reset functions, compiled against an
 * independent symbolic state backend. Tests lifecycle paths without a model. */
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>
#include "ds4_qwen4exp_gdn_replay.h"

#define DS4_MAX_LAYER 3
#define DS4_QWEN4EXP_IMPLEMENTED_DEPTH 6u
#define DS4_QWEN4EXP_MTP_MAX_COMMIT 7u
#define DS4_N_GDN_KEY_HEAD 1u
#define DS4_N_GDN_VALUE_HEAD 1u
#define DS4_QWEN4EXP_GDN_HEADS_TILED 1u
struct ds4_gpu_tensor { std::vector<uint64_t> v; };
struct ds4_qwen4exp_config { unsigned n_layer=DS4_MAX_LAYER; };
static ds4_qwen4exp_config g_ds4_qwen4exp;
struct ds4_qwen4exp_session {
    ds4_gpu_tensor *gdn_state[3]{}, *gdn_conv[3]{}, *gdn_checkpoint[3]{};
    ds4_gpu_tensor *gdn_replay_tape[3]{}, *gdn_state_snapshot[3]{}, *gdn_conv_snapshot[3]{};
    ds4_gpu_tensor *qsa_k[3]{}, *qsa_v[3]{}, *idx_tape[3]{}, *idx_pool[3]{};
    ds4_gpu_tensor *d_adopt=nullptr,*d_gdn_replay=nullptr,*ple_conv_state=nullptr;
    bool gdn_replay_enabled=true,gdn_replay_active=false,gdn_replay_previous=false;
    bool state_dirty=false;
    uint32_t gdn_replay_prefix=0,gdn_replay_phase=0;
    uint32_t adopt_state=0,adopt_conv=0,adopt_device=0,spec_snapshot_rows=0;
    uint32_t head_cache_pos=0,pos=0;
    int ple_constants=0,ple_history=0;
};
static unsigned cases=0,forwards=0,materializations=0;
[[noreturn]] static void fail(const char *what) {
    std::fprintf(stderr,"FAIL controller: %s after %u cases, %u forwards\n",what,cases,forwards);
    std::exit(1);
}
static void need(bool ok,const char *what) { if(!ok) fail(what); }
static uint64_t transition(uint64_t state,uint64_t token) {
    return (state ^ (token+0x9e3779b97f4a7c15ull))*0xbf58476d1ce4e5b9ull;
}
static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t?t->v.size()*8:0; }
static bool batch=false, fail_update=false;
static int ds4_gpu_begin_commands() { need(!batch,"nested command batch");batch=true;return 1; }
static int ds4_gpu_end_commands() { need(batch,"missing command batch");batch=false;return 1; }
static int ds4_gpu_synchronize() { return 1; }
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *d,uint64_t off,const ds4_gpu_tensor *s,uint64_t from,uint64_t n) {
    need(d && s && off+n<=ds4_gpu_tensor_bytes(d) && from+n<=ds4_gpu_tensor_bytes(s),"copy bounds");
    std::memcpy((char*)d->v.data()+off,(const char*)s->v.data()+from,n);return 1;
}
static int ds4_gpu_qwen4exp_update_dpos(ds4_gpu_tensor *d,uint32_t v) {
    if(fail_update) {fail_update=false;return 0;}
    need(d!=nullptr,"missing device control"); d->v[0]=v;return 1;
}
static void qwen4exp_zero_tensor(ds4_gpu_tensor *d) {
    if(d) std::fill(d->v.begin(),d->v.end(),0);
}
static void ds4_ple_history_reset(int *,int *h) { *h=0; }
static int ds4_gpu_qwen4exp_gdn_replay_materialize(ds4_gpu_tensor *out,
        ds4_gpu_tensor *base,ds4_gpu_tensor *tape,unsigned rows,unsigned,unsigned,unsigned) {
    need(rows<=DS4_QWEN4EXP_GDN_REPLAY_ROWS,"materialization length");
    uint64_t h=base->v[0];
    for(unsigned i=0;i<rows;i++) h=transition(h,tape->v[i]);
    out->v[0]=h;materializations++;return 1;
}
#include "gdn_replay_controller.inc"

struct fixture {
    ds4_qwen4exp_session s;
    std::vector<std::unique_ptr<ds4_gpu_tensor>> allocations;
    std::array<uint64_t,3> state{},conv{};
    std::array<std::array<uint64_t,6>,3> snap{},conv_snap{};
    std::map<uint32_t,std::pair<ds4_gpu_tensor*,bool>> graphs;
    uint64_t token=1;
    explicit fixture(bool enabled=true,bool lazy=true) {
        s.gdn_replay_enabled=enabled&&lazy;
        if(lazy) s.d_adopt=alloc(1);
        if(enabled&&lazy) s.d_gdn_replay=alloc(1);
        for(unsigned il:{0u,2u}) {
            s.gdn_state[il]=alloc(1);s.gdn_checkpoint[il]=alloc(1);
            s.gdn_conv[il]=alloc(1);s.gdn_replay_tape[il]=alloc(2);
            s.gdn_state_snapshot[il]=alloc(6);s.gdn_conv_snapshot[il]=alloc(6);
        }
    }
    ds4_gpu_tensor *alloc(size_t n) {
        allocations.emplace_back(new ds4_gpu_tensor{std::vector<uint64_t>(n)});
        return allocations.back().get();
    }
    void reset() {
        unsigned phase=s.gdn_replay_phase;
        ds4_qwen4exp_session_reset(&s);
        need(phase==s.gdn_replay_phase,"reset changed pointer parity");
        state.fill(0);conv.fill(0);
        for(unsigned il:{0u,2u}) need(s.gdn_state[il]->v[0]==0,"reset canonical state");
        need(!s.gdn_replay_previous && !s.gdn_replay_active && !s.adopt_state && !s.adopt_conv,"reset controller");
    }
    void select(unsigned row,bool recurrent=true,bool convolution=true) {
        need(qwen4exp_session_select_layers(&s,row,recurrent,convolution),"selection refused");
        for(unsigned il:{0u,2u}) {
            if(recurrent) state[il]=snap[il][row];
            if(convolution) conv[il]=conv_snap[il][row];
        }
    }
    void inspect() {
        for(unsigned il:{0u,2u}) {
            uint64_t live=s.gdn_state[il]->v[0];
            unsigned prefix=s.gdn_replay_prefix;
            if(s.gdn_replay_previous) {
                need(qwen4exp_gdn_replay_snapshot(&s,il,0),"virtual snapshot refused");
                need(s.gdn_state_snapshot[il]->v[0]==snap[il][0],"virtual snapshot mismatch");
                need(s.gdn_state[il]->v[0]==live && s.gdn_replay_prefix==prefix,"inspection mutated state");
            }
        }
    }
    void forward(unsigned width,unsigned snapshots,bool inspection=false) {
        s.spec_snapshot_rows=snapshots;s.state_dirty=true;
        need(prepare(&s,width),"prepare refused");
        unsigned key=ds4_qwen4exp_gdn_graph_variant(width,snapshots,s.gdn_replay_phase,s.gdn_replay_active);
        const auto identity=std::make_pair(s.gdn_state[0],s.gdn_replay_active);
        if(graphs.count(key)) need(graphs[key]==identity,"graph reused a different buffer or kernel");
        graphs[key]=identity;
        for(unsigned il:{0u,2u}) {
            uint64_t h=s.gdn_state[il]->v[0];
            uint64_t c=s.gdn_conv[il]->v[0];
            unsigned adopt=s.d_adopt?(unsigned)s.d_adopt->v[0]:0;
            unsigned prefix=s.d_gdn_replay?(unsigned)s.d_gdn_replay->v[0]:0;
            if(adopt) c=s.gdn_conv_snapshot[il]->v.at(adopt-1);
            if(s.gdn_replay_active) {
                h=s.gdn_checkpoint[il]->v[0];
                for(unsigned i=0;i<prefix;i++) h=transition(h,s.gdn_replay_tape[il]->v.at(i));
            } else if(adopt) h=s.gdn_state_snapshot[il]->v.at(adopt-1);
            need(h==state[il] && c==conv[il],"forward did not adopt selected state");
            for(unsigned t=0;t<width;t++) {
                uint64_t input=token+t+1000*il;
                h=transition(h,input);c=transition(c,input+123);
                state[il]=transition(state[il],input);
                conv[il]=transition(conv[il],input+123);
                need(h==state[il] && c==conv[il],"row output mismatch");
                if(t<snapshots) {snap[il][t]=state[il];conv_snap[il][t]=conv[il];}
                if(t<snapshots) s.gdn_conv_snapshot[il]->v[t]=c;
                if(s.gdn_replay_active && t==0) {
                    if(prefix==2) s.gdn_checkpoint[il]->v[0]=h;
                    else s.gdn_replay_tape[il]->v.at(prefix)=input;
                } else if(!s.gdn_replay_active && t<snapshots) s.gdn_state_snapshot[il]->v[t]=h;
            }
            s.gdn_state[il]->v[0]=h;s.gdn_conv[il]->v[0]=c;
        }
        s.gdn_replay_previous=s.gdn_replay_active;s.adopt_state=0;s.adopt_conv=0;
        token+=width;s.pos+=width;forwards++;
        if(inspection) inspect();
    }
};
static uint32_t rng=1;
static uint32_t random_word() {rng=rng*1664525u+1013904223u;return rng;}
int main() {
    for(unsigned mask=0;mask<4096;mask++) for(unsigned style=0;style<3;style++) {
        fixture f;
        f.forward(1,0);
        for(unsigned r=0;r<12;r++) {
            f.forward(2,1,style!=0);
            if(!(mask&(1u<<r))) {
                if(style==2) { f.select(0,true,false); f.select(0,false,true); f.select(0); }
                else f.select(0);
            }
        }
        f.forward(1,0,true); // settle a pending virtual state on the tail
        f.reset();f.reset();f.forward(2,1,true);cases++;
    }
    // Changes of width, partial selections and no-lazy/no-replay valves.
    for(unsigned mode=0;mode<3;mode++) for(unsigned run=0;run<128;run++) {
        fixture f(mode!=1,mode!=2);
        for(unsigned r=0;r<80;r++) {
            unsigned width=1+random_word()%7;
            unsigned snapshots=width-1;
            f.forward(width,snapshots,true);
            if(snapshots && (random_word()%3)) {
                unsigned rs=random_word()%snapshots,rc=random_word()%snapshots;
                f.select(rs,true,false);f.select(rc,false,true);
                if(random_word()%2) f.select(random_word()%snapshots,true,false);
            }
            if(random_word()%11==0) f.reset();
        }
        f.forward(8,0);f.reset();f.forward(2,1);cases++;
    }
    // An update can fail after a pointer swap. Reset must clear that canonical
    // buffer and retain its graph identity, then a new verify must work.
    {
        fixture f;f.forward(2,1);f.select(0);
        f.s.spec_snapshot_rows=1;f.s.state_dirty=true;fail_update=true;
        need(!prepare(&f.s,2),"injected prefix failure ignored");
        f.reset();f.forward(2,1,true);
        f.forward(2,1); // full acceptance forces swap
        fail_update=true;
        need(!prepare(&f.s,2),"post-swap update failure ignored");
        f.reset();f.forward(2,1,true);cases++;
    }
    need(!batch,"command batch leaked");
    std::printf("PASS actual graph lifecycle: %u cases, %u forwards, %u materializations\n",cases,forwards,materializations);
}
