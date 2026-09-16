#!/usr/bin/env python3
"""Synthetic, file-free PLE snapshot-arm regression.

Extracts actual old/current snapshot gather arms and the real row-address helper;
links the actual production hash/decoder. Only fprintf and prefetch are spies.
Does not execute full gather validation, uploads, non-snapshot workers or a model.
No network access. Requires the promoted base commit in the local Git object DB.
"""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = '34e4049d4032ecbc7af7cef67d2de28b95ab1181'

def block(text, marker):
    start = text.index(marker)
    op = text.index('{', start)
    depth = 1
    at = op + 1
    while depth:
        depth += (text[at] == '{') - (text[at] == '}')
        at += 1
    return text[start:at]

def snapshot(text):
    # First occurrence is the actual gather arm, not any downstream reader.
    gather = text[text.index('static bool qwen4exp_graph_ple_gather('):]
    return block(gather, 'if (s->spec_snapshot_rows > 0)')

old = subprocess.check_output(['git', 'show', f'{BASE}:ds4/ds4_qwen4exp_graph.inc'], cwd=ROOT, text=True)
new = (ROOT/'ds4/ds4_qwen4exp_graph.inc').read_text()
row = block((ROOT/'ds4/ds4_qwen4exp.h').read_text(), 'static inline const uint8_t *ds4_qwen4exp_ple_row(')
source = r'''
#include "ds4_qwen4exp_ple.h"
#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DS4_N_PLE_HEAD 16u
#define DS4_N_PLE_ROW_DIM 160u
#define DS4_N_PLE_EMBD 2560u
#define MAXROWS 4u
#define TABLE_ROWS 128u
#define TABLE_BYTES (TABLE_ROWS*90u)
typedef struct { const uint8_t *base; uint64_t rows,row_bytes,bytes,row_dim; uint32_t type; } ds4_qwen4exp_ple_table;
typedef struct {ds4_qwen4exp_ple_table ple;} ds4_qwen4exp_weights;
typedef struct {
 ds4_ple_constants ple_constants; ds4_ple_history ple_history,ple_history_rows[4];
 uint64_t *ple_ids; float *ple_rows_host; uint32_t spec_snapshot_rows;
} ds4_qwen4exp_session;
static char errors[512];
static uintptr_t hints[256]; static unsigned nhints;
static uintptr_t table_low,table_high,ids_low,ids_high;
static int log_error(FILE *f,const char *fmt,...) { (void)f; va_list ap;va_start(ap,fmt);int n=vsnprintf(errors,sizeof(errors),fmt,ap);va_end(ap);return n; }
static void hint(const void *p,int rw,int locality) {
 (void)rw;(void)locality;uintptr_t u=(uintptr_t)p;
 if(u>=ids_low && u<ids_high) {assert((u-ids_low)%sizeof(uint64_t)==0);return;}
 assert(u>=table_low && u<table_high);assert(nhints<256);hints[nhints++]=u;
}
'''+row+r'''
#define fprintf log_error
#define __builtin_prefetch hint
'''
for name, arm in [('old_arm', snapshot(old)), ('new_arm', snapshot(new))]:
    source += f'''static bool {name}(ds4_qwen4exp_session *s,const ds4_qwen4exp_weights *w,const int32_t *tokens,uint32_t n_tokens,uint32_t il) {{
 const uint32_t blocks_per_row=5u;
 {arm}
 return true;
}}
'''
source += r'''
#undef fprintf
#undef __builtin_prefetch
static void constants(ds4_ple_constants *c,int mode,int badhead) {
 memset(c,0,sizeof(*c)); c->ngram_size=3;c->heads_per_ngram=8;c->head_count=16;
 c->row_dim=160;c->eos_token_id=0;c->multipliers[0]=1;c->multipliers[1]=3;c->multipliers[2]=5;
 for(unsigned h=0;h<16;h++) {
  c->head_vocab_sizes[h]=(mode==1?64:1);
  c->head_offsets[h]=(mode==1?0:(mode==2?127:h*2));
 }
 if(badhead>=0)c->head_offsets[badhead]=TABLE_ROWS;
}
static unsigned cases,three_line,lastrow,secondinvalid;
static uint32_t float_bits(const float *p) {uint32_t u;memcpy(&u,p,sizeof(u));return u;}
static void set_float_bits(float *p,uint32_t u) {memcpy(p,&u,sizeof(u));}
static void run(unsigned rows,unsigned snaps,unsigned alignment,unsigned pattern,int disabled,int mode,int badhead,int metadata) {
 uint8_t *allocation=NULL;assert(posix_memalign((void**)&allocation,64,TABLE_BYTES+128)==0);
 uint8_t *table=allocation+alignment;
 static const uint16_t scales[]={0x3c00,0x0001,0x8000,0x7c00,0xfc00,0x7e13,0x7d05,0x7bff};
 for(unsigned r=0;r<TABLE_ROWS;r++)for(unsigned b=0;b<5;b++) {
  uint8_t *q=table+r*90+b*18;uint16_t d=scales[(pattern+b)%8];memcpy(q,&d,2);
  for(unsigned j=0;j<16;j++)q[2+j]=(uint8_t)(r*29+b*43+j*17+pattern);
 }
 uint8_t saved[TABLE_BYTES];memcpy(saved,table,TABLE_BYTES);
 table_low=(uintptr_t)table;table_high=table_low+TABLE_BYTES;
 uint64_t ids_a[MAXROWS*16+2],ids_b[MAXROWS*16+2];
 float out_a[MAXROWS*2560+2],out_b[MAXROWS*2560+2];
 memset(ids_a,0xa5,sizeof(ids_a));memcpy(ids_b,ids_a,sizeof(ids_a));
 for(unsigned i=0;i<MAXROWS*2560+2;i++)set_float_bits(&out_a[i],0x5a5a1234);memcpy(out_b,out_a,sizeof(out_a));
 ds4_qwen4exp_session a={0},b={0}; constants(&a.ple_constants,mode,badhead);
 ds4_ple_history_reset(&a.ple_constants,&a.ple_history);
 memset(a.ple_history_rows,0x5a,sizeof(a.ple_history_rows));
 a.spec_snapshot_rows=snaps;b=a;a.ple_ids=ids_a+1;b.ple_ids=ids_b+1;
 a.ple_rows_host=out_a+1;b.ple_rows_host=out_b+1;
 ds4_qwen4exp_weights w={.ple={table,TABLE_ROWS,90,TABLE_BYTES,160,0}};
 int32_t tokens[4]={0,1,0,-1};
 if(mode==1)w.ple.rows=1; // Real hash makes token0 valid, token1 invalid.
 if(metadata==1)w.ple.bytes=1; // Only hint eligibility declines; real backing remains valid.
 if(metadata==2)w.ple.row_bytes=91; // Allocate enough and keep referenced rows bounded.
 if(metadata==3)w.ple.base=NULL;
 if(metadata==4)w.ple.rows=UINT64_MAX; // Overflow guard declines hints; referenced IDs remain small.
 if(metadata==5)w.ple.bytes=UINT64_MAX; // Integer pointer-end guard declines hints.
 if(disabled)setenv("DS4_QWEN4EXP_NO_PLE_ROW_PREFETCH","1",1);else unsetenv("DS4_QWEN4EXP_NO_PLE_ROW_PREFETCH");
 ids_low=(uintptr_t)a.ple_ids;ids_high=ids_low+rows*16*sizeof(uint64_t);
 errors[0]=0;nhints=0;bool oa=old_arm(&a,&w,tokens,rows,7);char old_error[512];strcpy(old_error,errors);
 ids_low=(uintptr_t)b.ple_ids;ids_high=ids_low+rows*16*sizeof(uint64_t);
 errors[0]=0;nhints=0;bool ob=new_arm(&b,&w,tokens,rows,7);
 assert(oa==ob);assert(!strcmp(old_error,errors));assert(!memcmp(out_a,out_b,sizeof(out_a)));
 assert(!memcmp(ids_a,ids_b,sizeof(ids_a)));assert(!memcmp(&a.ple_history,&b.ple_history,sizeof(a.ple_history)));
 assert(!memcmp(a.ple_history_rows,b.ple_history_rows,sizeof(a.ple_history_rows)));
 assert(!memcmp(saved,table,TABLE_BYTES));
 // Independent ordered-valid-prefix oracle: no table hints after the first bad row.
 unsigned valid=0;
 if(snaps)for(;valid<rows*16;valid++)if(!w.ple.base || ids_b[1+valid]>=w.ple.rows)break;
 bool eligible=snaps && rows<=2 && !disabled && metadata==0;
 assert(nhints==(eligible?3*valid:0));
 for(unsigned i=0;i<nhints;i++) {
  unsigned pair=i/3,off=(unsigned[]){0,64,89}[i%3];
  uintptr_t expected=table_low+ids_b[1+pair]*90+off;
  assert(hints[i]==expected);assert(hints[i]>=table_low && hints[i]<table_high);
 }
 if(eligible && valid && mode==2) {
  uintptr_t p=table_low+127*90;
  if(p/64 != (p+89)/64 && (p+89)/64-p/64==2)three_line++;
 }
 if(eligible && mode==2 && valid)lastrow++;
 if(mode==1 && rows==2 && !oa && strstr(errors,"token 1 head 0"))secondinvalid++;
 // Untouched output beyond the actual decode prefix, including leading/trailing canaries.
 unsigned written=valid*160;
 assert(float_bits(&out_b[0])==0x5a5a1234);
 for(unsigned i=written+1;i<MAXROWS*2560+2;i++)assert(float_bits(&out_b[i])==0x5a5a1234);
 free(allocation);cases++;
}
int main(void) {
 for(unsigned r=1;r<=3;r++)for(unsigned snaps=1;snaps<=2;snaps++)
 for(unsigned align=0;align<64;align+=8)for(unsigned pat=0;pat<8;pat++)for(int off=0;off<2;off++)
  run(r,snaps,align,pat,off,0,-1,0);
 for(unsigned r=1;r<=2;r++)for(unsigned snaps=1;snaps<=2;snaps++)for(int h=0;h<16;h++)for(int off=0;off<2;off++)
  run(r,snaps,40,h%8,off,0,h,0);
 for(unsigned r=1;r<=2;r++)for(int off=0;off<2;off++)for(int mode=1;mode<=2;mode++)
  run(r,1,40,3,off,mode,-1,0);
 for(int metadata=1;metadata<=5;metadata++)for(unsigned r=1;r<=2;r++)run(r,1,0,0,0,0,-1,metadata);
 run(2,0,0,0,0,0,-1,0);
 assert(lastrow && secondinvalid); // Distinct last-row and token1 failure witnesses.
 // Verify a last-row placement whose90bytes spans three64B lines.
 for(unsigned align=0;align<64;align++)run(1,1,align,0,0,2,-1,0);
 assert(three_line);
 printf("PASS %u synthetic snapshot cases; last-row=%u three-line=%u second-token-invalid=%u\n",cases,lastrow,three_line,secondinvalid);
 return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='ple-prefetch-test-') as tmp:
    c = Path(tmp)/'test.c'; exe=Path(tmp)/'test'
    c.write_text(source)
    command=[os.environ.get('CC','cc'),'-O2','-ffast-math','-fno-finite-math-only','-std=c11','-D_GNU_SOURCE','-I'+str(ROOT/'ds4'),str(c),str(ROOT/'ds4/ds4_qwen4exp_ple.c'),'-lm','-o',str(exe)]
    subprocess.run(command,check=True)
    subprocess.run([str(exe)],check=True)
