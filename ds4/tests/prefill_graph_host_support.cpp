/* Consumed by test_prefill_graph_host.py; CUDA APIs are deterministic mocks. */
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#define DS4_TEST_HOOKS
constexpr uint32_t DS4_QWEN4EXP_MTP_MAX_COMMIT=7, DS4_N_INDEXER_TOP_K=2048;
enum { QW_SLICE_PLE, QW_SLICE_ATTN_MIX, QW_SLICE_QSA, QW_SLICE_GDN,
       QW_SLICE_ATTN_INJECT, QW_SLICE_FFN_MIX, QW_SLICE_MOE, QW_SLICE_FFN_INJECT };
struct Device { uint64_t bank[2]; uint32_t input; };
struct ds4_qwen4exp_session {
    void *hyper; uint32_t *d_pos; uint32_t pos, spec_snapshot_rows, gdn_replay_phase;
    bool gdn_replay_active, hc_pending;
};
struct ds4_qwen4exp_layer_weights { bool has_ple, is_full_attention; };
struct ds4_qwen4exp_weights { ds4_qwen4exp_layer_weights layer[64]; };
struct ds4_model { int marker; };
struct ds4_decode_graph_key {
    uint32_t il,island,variant,_pad;
    void *cur_hc,*after_attn_hc,*after_ffn_hc,*attn_norm;
};
struct ds4_qwen4exp_config { uint32_t n_layer; };
static ds4_qwen4exp_config g_ds4_qwen4exp;
static uint32_t chunks=4, g_qwen4exp_trace_len;
static bool supported=true,hb=false,dump=false,defer_off=false,g_qwen4exp_trace_overflow;
static unsigned cases, encodes, replays, captures, aborts, begins, prefetched, eager_islands;
static int fail_end=-1, fail_begin=-1, fail_encode=-1, fail_replay=-1;
static std::vector<int> trace;
#define QW_TRACE(x) do { trace.push_back(x); g_qwen4exp_trace_len=(uint32_t)trace.size(); } while(0)
static void qw_decode_graph_trace_range(const ds4_qwen4exp_weights*,uint32_t,uint32_t);
static bool qw_hb_time_on(){return hb;}
static bool qwen4exp_layer_dump_on(){return dump;}
static bool ds4_gpu_decode_graphs_supported(){return supported;}
static bool qwen4exp_hc_defer_ok(uint32_t n){return n>=48&&!defer_off;}
static uint32_t qw_decode_graph_chunks(){return chunks;}
static uint32_t ds4_qwen4exp_gdn_graph_variant(uint32_t n,uint32_t snap,uint32_t phase,bool active){
 return n|(snap<<8)|(phase<<16)|(uint32_t(active)<<17);
}
struct Command { Device *device; uint32_t *pos; uint32_t il,width,phase; bool pending; };
struct Entry { ds4_decode_graph_key key{}; int state=0; std::vector<Command> commands; };
static std::array<std::vector<Entry>,64> cache;
static Entry *recording;
static void execute(const Command& c){
 c.device->bank[c.phase]=c.device->bank[c.phase]*3u+7u*c.il+c.width+
  *c.pos+c.device->input+(c.pending?11u:0u);
}
static void reset_cache(){for(auto& row:cache)row.clear();recording=nullptr;}
static int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *k){
 assert(!recording&&k->il<64&&k->island==3);begins++;
 if(int(k->il)==fail_begin)return -1;
 auto& row=cache[k->il];Entry *e=nullptr;
 for(auto& x:row)if(!memcmp(&x.key,k,sizeof *k)){e=&x;break;}
 if(!e){if(row.size()==8)return -1;row.push_back({*k,0,{}});e=&row.back();}
 if(e->state==3)return -1;
 if(e->state==0){e->state=1;return -1;}
 if(e->state==2){
  if(int(k->il)==fail_replay){e->state=3;return -1;}
  for(auto& c:e->commands)execute(c);replays++;return 1;
 }
 e->commands.clear();recording=e;return 0;
}
static int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *k){
 assert(recording&&!memcmp(&recording->key,k,sizeof *k));
 Entry *e=recording;recording=nullptr;
 if(int(k->il)==fail_end){e->state=3;return -1;}
 for(auto& c:e->commands)execute(c);e->state=2;captures++;return 0;
}
static void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key*){
 assert(recording);recording->state=3;recording=nullptr;aborts++;
}
static int ds4_gpu_decode_graph_prefetch(const ds4_decode_graph_key *k){
 assert(k->il<64);prefetched++;return 0;
}
static bool qwen4exp_graph_layers_encode_range(ds4_qwen4exp_session *s,
 const ds4_qwen4exp_weights *w,const ds4_model*,uint32_t n,bool islands,uint32_t first,uint32_t last){
 encodes++;eager_islands+=islands;assert(first<last&&last<=g_ds4_qwen4exp.n_layer);
 if(recording&&int(recording->key.il)==fail_encode){s->hc_pending=!s->hc_pending;return false;}
 // Trim the mock trace after a failed captured encode, as the real fixed array does.
 trace.resize(g_qwen4exp_trace_len);
 for(uint32_t il=first;il<last;il++){
  Command c{static_cast<Device*>(s->hyper),s->d_pos,il,n,s->gdn_replay_phase,s->hc_pending};
  if(recording)recording->commands.push_back(c);else execute(c);
  s->hc_pending=qwen4exp_hc_defer_ok(n);
 }
 qw_decode_graph_trace_range(w,first,last);return true;
}
static bool qwen4exp_graph_layers_encode(ds4_qwen4exp_session *s,
 const ds4_qwen4exp_weights *w,const ds4_model *m,uint32_t n,bool islands){
 if(!g_ds4_qwen4exp.n_layer)return true;
 return qwen4exp_graph_layers_encode_range(s,w,m,n,islands,0,g_ds4_qwen4exp.n_layer);
}
/* ACTUAL_CONTROLLER */
static ds4_qwen4exp_weights weights;
static ds4_model model;
static uint64_t expected(uint64_t value,uint32_t input,uint32_t pos,uint32_t width,bool pending){
 for(uint32_t il=0;il<g_ds4_qwen4exp.n_layer;il++){
  value=value*3u+7u*il+width+pos+input+(pending?11u:0u);
  pending=qwen4exp_hc_defer_ok(width);
 }return value;
}
static void forward(ds4_qwen4exp_session& s,uint32_t width,bool pending,uint32_t input){
 auto& d=*static_cast<Device*>(s.hyper);d.bank[0]=d.bank[1]=5;d.input=input;
 uint32_t p=s.pos;uint32_t *saved=s.d_pos;s.d_pos=&p;
 // A real d_pos allocation is stable; caller supplies it for captured tests.
 if(saved){s.d_pos=saved;*saved=p;}
 s.hc_pending=pending;trace.clear();g_qwen4exp_trace_len=0;
 assert(qwen4exp_graph_layers(&s,&weights,&model,width));
 assert(d.bank[s.gdn_replay_phase]==expected(5,input,p,width,pending));
 assert(d.bank[1u-s.gdn_replay_phase]==5);
 assert(s.hc_pending==(g_ds4_qwen4exp.n_layer?qwen4exp_hc_defer_ok(width):pending));
 unsigned want=0;for(uint32_t i=0;i<g_ds4_qwen4exp.n_layer;i++)want+=6+weights.layer[i].has_ple;
 assert(g_qwen4exp_trace_len==want);s.d_pos=saved;cases++;
}
int main(){
 for(unsigned i=0;i<64;i++)weights.layer[i]={i%5==0,i%4==3};
 Device a{},b{};uint32_t pos=0,pos2=0;
 ds4_qwen4exp_session s{&a,&pos,0,0,0,false,false};
 for(unsigned layers:{1u,4u,48u,55u})for(unsigned nc:{1u,2u,4u,8u})
 for(unsigned width:{8u,47u,48u,64u,128u,1024u,2048u})
 for(unsigned phase:{0u,1u})for(bool pending:{false,true})for(bool off:{false,true}){
  reset_cache();g_ds4_qwen4exp.n_layer=layers;chunks=nc;s.gdn_replay_phase=phase;defer_off=off;
  unsigned before=replays;
  for(unsigned pass=0;pass<4;pass++){s.pos=(2048-width)*pass/3;forward(s,width,pending,31+pass);}
  assert(replays>before);
 }
 defer_off=false;g_ds4_qwen4exp.n_layer=48;chunks=4;
 // One cache across changing widths/parities/entry flags and distinct storage.
 reset_cache();
 for(unsigned pass=0;pass<4;pass++)for(unsigned width:{47u,48u})for(unsigned phase:{0u,1u})
 for(bool pending:{false,true}){
  s.gdn_replay_phase=phase;s.pos=pass*7;forward(s,width,pending,pass+51);
 }
 auto other=s;other.hyper=&b;other.d_pos=&pos2;
 for(unsigned pass=0;pass<4;pass++)forward(other,48,false,100+pass);
 // Capacity exhaustion must stay eager, without replacing a captured pointer.
 reset_cache();for(unsigned pass=0;pass<4;pass++)for(unsigned w=48;w<59;w++)forward(s,w,false,w+pass);
 for(int kind=0;kind<4;kind++)for(unsigned chunk=0;chunk<4;chunk++){
  reset_cache();s.gdn_replay_phase=0;s.pos=0;forward(s,64,false,1);
  fail_end=kind==0?56+int(chunk):-1;fail_begin=kind==1?56+int(chunk):-1;
  if(kind==2){forward(s,64,false,2);fail_replay=56+int(chunk);}
  if(kind==3){
   fail_encode=56+int(chunk);s.hc_pending=false;
   assert(!qwen4exp_graph_layers(&s,&weights,&model,64));assert(!recording);
   assert(s.hc_pending==(chunk!=0));cases++;fail_encode=-1;
  }else{forward(s,64,false,3);forward(s,64,false,4);}
  fail_end=fail_begin=fail_replay=-1;
 }
 // Eligibility failures: these invoke the actual dispatcher and eager path.
 for(unsigned mode=0;mode<10;mode++){
  reset_cache();s={&a,&pos,0,0,0,false,false};g_ds4_qwen4exp.n_layer=48;
  if(mode==0)setenv("DS4_QWEN4EXP_NO_PREFILL_GRAPHS","1",1);
  if(mode==1)setenv("DS4_QWEN4EXP_TIME_SLICES","1",1);
  if(mode==2)hb=true;if(mode==3)dump=true;if(mode==4)supported=false;
  if(mode==5)s.spec_snapshot_rows=1;if(mode==6)s.gdn_replay_active=true;
  if(mode==7)s.pos=2040;if(mode==8)g_ds4_qwen4exp.n_layer=56;
  if(mode==9){s.d_pos=nullptr;assert(!qw_prefill_graph_ok(&s,64));cases++;s.d_pos=&pos;continue;}
  unsigned before=begins;forward(s,64,false,7);assert(begins==before);
  unsetenv("DS4_QWEN4EXP_NO_PREFILL_GRAPHS");unsetenv("DS4_QWEN4EXP_TIME_SLICES");
  hb=dump=false;supported=true;
 }
 // Existing decode widths retain their original rows and dispatch.
 s={&a,&pos,0,0,0,false,false};g_ds4_qwen4exp.n_layer=48;
 for(unsigned nc:{1u,2u,4u,8u})for(unsigned width:{1u,2u,7u}){
  reset_cache();chunks=nc;for(unsigned pass=0;pass<4;pass++)forward(s,width,false,pass+1);
  for(unsigned row=56;row<64;row++)assert(cache[row].empty());
 }
 printf("Actual prefill/decode graph controller: %u forwards/guards PASS; captures=%u replays=%u aborts=%u\n",cases,captures,replays,aborts);
}
