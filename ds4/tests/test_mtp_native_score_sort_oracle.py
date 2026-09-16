# Host ordering oracle for the native MTP screen score sort; no GPU, no CUDA,
# no compiler. ds4/tests/test_mtp_native_key_oracle.py pins HOW keys are
# produced (standalone and fused producers, IEEE and FTZ). This one pins WHAT
# the production radix bit range must order: a stable descending sort on the
# packed high score word alone (CUB bits [32,64)) reproduces the full 64-bit
# descending key order, because keys are written in row order and original IDs
# strictly increase over the whole input, so the packed low words (~ID) already
# strictly descend and CUB radix sort is stable. Compares the COMPLETE sorted
# key arrays and the selected IDs (unpack + ascending ID sort, the next two
# production steps) for both orderings, over ties (including canonical signed
# zero), finite specials and the mandatory zero/tail rows. Refined outputs are
# an exact per-row function of the selected IDs (test_mtp_native_screen.c pins
# them row-by-row against ordinary rows), so selected-ID parity is
# refined-output parity. Non-finite scores never reach the sort: they raise
# the invalid flag and the screen falls back first.
from pathlib import Path
import random, re, struct

repo = Path(__file__).resolve().parents[2]
cuh = (repo/'ds4/ds4_cuda_mtp_native.cuh').read_text()
cu = (repo/'ds4/ds4_cuda.cu').read_text()

def radix_calls(text):
    calls = []
    for m in re.finditer(r'DeviceRadixSort::(SortKeys(?:Descending)?)\(', text):
        depth, i = 1, m.end()
        while depth:
            depth += (text[i] == '(') - (text[i] == ')')
            i += 1
        calls.append((m.group(1), text[m.end():i-1], re.sub(r'\s+', '', text[m.end():i-1])))
    return calls

def trailing(args):
    m = re.search(r',(\w+),(\d+),(\d+),cuda_decode_stream\(\)$', args)
    assert m, 'radix call tail shape changed: ' + args
    return m.groups()

calls = radix_calls(cuh)
def one(pattern, what):
    hits = [c for c in calls if re.search(pattern, c[1])]
    assert len(hits) == 1, what + ': expected exactly one radix call'
    return hits[0]

# Wiring: the score sort and its scratch-size query must both sort the high
# word only. The fused selected-ID sort is exercised on a GPU by
# test_mtp_native_unpack_sort.py.
score_name, _, score_args = one(r'key_in\s*,\s*key_out', 'score sort')
init_name, _, init_args = one(r'\(\s*const\s+uint64_t\s*\*\s*\)\s*nullptr', 'score sort scratch query')
score_n, score_lo, score_hi = trailing(score_args)
init_n, init_lo, init_hi = trailing(init_args)
assert score_name == 'SortKeysDescending' and score_n == 'width' \
    and (score_lo, score_hi) == ('32', '64'), 'production score sort bits drifted'
assert init_name == 'SortKeysDescending' and init_n == 'width' \
    and (init_lo, init_hi) == ('32', '64'), 'scratch query bits must match the sort'

# Key layout, extracted from the producers rather than restated.
fk = re.search(r'q8_top1_float_ordered_key\(float v\)\s*\{\s*const uint32_t u = '
    r'__float_as_uint\(v\);\s*return \(u & (0x[0-9a-f]+)u\) \? ~u : \(u \^ (0x[0-9a-f]+)u\);', cu)
pk = re.search(r'q8_top1_pack_key\(float v, uint32_t idx\)\s*\{\s*'
    r'return \(\(uint64_t\)q8_top1_float_ordered_key\(v\) << (\d+)u\) \|\s*'
    r'\(uint64_t\)\((0x[0-9a-f]+)u - idx\);', cu)
assert fk and pk, 'key packing helpers changed; revisit this oracle'
sign, flip, shift, invert = (int(fk.group(1), 16), int(fk.group(2), 16),
                             int(pk.group(1)), int(pk.group(2), 16))
assert sign == flip == 0x80000000 and shift == 32 and invert == 0xffffffff
assert len(re.findall(r'if \(!id \|\| \w+ >= prefix\) keys\[\w+\] = UINT64_MAX - id;'
    r'\s*else keys\[\w+\] = q8_top1_pack_key\(value == 0\.0f \? 0\.0f : value, id\);', cuh)) == 2, \
    'key producers changed (standalone mtp_native_keys and fused EmitKeys must agree)'
cap = int(re.search(r'MTP_NATIVE_CAP = (\d+)u', cuh).group(1))
m32 = 0xffffffff

def ordered(bits):
    return (~bits & m32) if bits & sign else (bits ^ flip)

def build(prefix, tail, vocab, score_bits):
    """Keys exactly as either producer writes them, in row order."""
    width = prefix + tail
    # Mirrors the screen's width guards; only this domain is ever sorted.
    assert cap < width <= 1 << 20 and 0 < tail < cap \
        and prefix <= vocab and tail <= vocab - prefix
    keys, ids = [], []
    for row in range(width):
        idx = row if row < prefix else vocab - tail + row - prefix
        if not idx or row >= prefix:
            keys.append(0xffffffffffffffff - idx)  # mandatory zero/tail slots
        else:
            b = score_bits[row]
            if struct.unpack('<f', struct.pack('<I', b))[0] == 0.0:
                b = 0  # canonical signed zero; FTZ only widens this tie
            keys.append((ordered(b) << shift) | (invert - idx))
        ids.append(idx)
    return keys, ids

def selected(order):
    """The next two production steps: top-CAP unpack, ascending ID sort."""
    return sorted(m32 - (k & m32) for k in order[:cap])

def check(prefix, tail, vocab, score_bits):
    keys, ids = build(prefix, tail, vocab, score_bits)
    assert all(a < b for a, b in zip(ids, ids[1:])), \
        'original IDs must strictly increase: the stability lemma itself'
    full = sorted(keys, reverse=True)  # full 64-bit descending order
    word = sorted(keys, key=lambda k: k >> shift, reverse=True)  # stable, high word
    assert full == word, 'score-word order diverged from full-key order'
    assert selected(full) == selected(word), 'selected IDs diverged'
    return len(keys)

def finite_bits(r):
    return r & 0xff7fffff  # exponent field capped below 0xFF: always finite

specials = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff,
            0x807fffff, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
            0x7f7fffff, 0xff7fffff]
def pattern(name, width, rng):
    if name == 'all-zero': return [0] * width
    if name == 'signed-zero': return [rng.choice((0, 0x80000000)) for _ in range(width)]
    if name == 'specials': return [specials[i % len(specials)] for i in range(width)]
    if name == 'pool':  # small value pool: heavy ties across scored rows
        pool = specials + [finite_bits(rng.getrandbits(32)) for _ in range(64)]
        return [rng.choice(pool) for _ in range(width)]
    return [finite_bits(rng.getrandbits(32)) for _ in range(width)]  # 'random'

rng = random.Random(20260913)
cases = compared = 0
# The GPU screen shape, the minimal width past CAP, a CAP filled entirely by
# mandatory rows, and the maximal 2^20 width with the tail at the vocabulary end.
for prefix, tail, vocab in [(20000, 276, 21000), (2049, 1, 21000),
                            (2, 2047, 21000), ((1 << 20) - 276, 276, 1 << 20)]:
    width = prefix + tail
    for name in (['all-zero', 'signed-zero', 'specials', 'pool', 'random']
                 if width < 100000 else ['all-zero', 'pool']):
        compared += check(prefix, tail, vocab, pattern(name, width, rng))
        cases += 1
print(f"PASS {cases} cases, {compared} keys: stable bits [{score_lo},{score_hi}) order equals "
      "full-key order (complete sorted arrays and selected IDs), ties and mandatory rows included")
