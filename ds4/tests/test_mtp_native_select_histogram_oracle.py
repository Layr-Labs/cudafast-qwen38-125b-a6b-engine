"""Host oracle for radix-selection histogram aggregation and boundary election.

Run with Python only. This checks integer semantics against independent
Counter/sorted references; it does not compile or execute CUDA and makes no
claim about GPU synchronization, generated code or performance.
"""
from collections import Counter
from itertools import product
import random


def grouped_histogram(digits):
    hist = [0] * 256
    for start in range(0, len(digits), 32):
        warp = digits[start:start + 32]
        # A group leader adds the number of lanes with its digit. Include
        # arbitrary partial warps, not only multiples of the launch width.
        seen = set()
        for digit in warp:
            if digit not in seen:
                hist[digit] += sum(value == digit for value in warp)
                seen.add(digit)
    return hist


def boundary(hist, need):
    prefix = 0
    writers = []
    for digit in range(255, -1, -1):
        hb = hist[digit]
        prefix += hb
        if prefix >= need and prefix - hb < need:
            writers.append((digit, prefix - hb, hb))
    assert len(writers) == 1, (hist, need, writers)
    return writers[0]


def select(keys, cap, rng):
    candidates = list(keys)
    output = []
    need = cap
    for shift in range(56, -1, -8):
        if len(candidates) <= 256:
            output.extend(sorted(candidates, reverse=True)[:need])
            break
        digits = [(key >> shift) & 255 for key in candidates]
        hist = grouped_histogram(digits)
        expected = Counter(digits)
        assert hist == [expected[digit] for digit in range(256)]
        digit, above, count = boundary(hist, need)
        last = shift == 0 or above + count == need
        output.extend(key for key in candidates
                      if ((key >> shift) & 255) > digit
                      or (last and ((key >> shift) & 255) == digit))
        if last:
            break
        candidates = [key for key in candidates
                      if ((key >> shift) & 255) == digit]
        # GPU atomic compaction may permute the remaining candidates.
        rng.shuffle(candidates)
        need -= above
        assert 0 < need <= len(candidates)
    assert len(output) == cap
    assert sorted(output, reverse=True) == sorted(keys, reverse=True)[:cap]


def main():
    rng = random.Random(0x922C0DA)
    hist_cases = 0
    # Exhaust every eight-bin histogram with bin counts 0..2 and every
    # possible rank. Place the bins at both ends and across warp boundaries.
    for locations in ((0, 1, 2, 3, 4, 5, 6, 7),
                      (0, 31, 32, 127, 128, 223, 224, 255)):
        for counts in product(range(3), repeat=8):
            hist = [0] * 256
            for digit, count in zip(locations, counts):
                hist[digit] = count
            reference = sorted((d for d, c in enumerate(hist)
                                for _ in range(c)), reverse=True)
            for rank, expected in enumerate(reference, 1):
                digit, above, count = boundary(hist, rank)
                assert digit == expected and above < rank <= above + count
                hist_cases += 1
    # Every physical bucket can be the crossing bucket, with empty plateaus.
    for digit in range(256):
        hist = [0] * 256
        hist[digit] = 1025
        for need in (1, 31, 32, 256, 1024, 1025):
            assert boundary(hist, need) == (digit, 0, 1025)
            hist_cases += 1

    # All tail lengths, random digits and worst-case single-bucket contention.
    for n in range(1, 290):
        for digits in ([rng.randrange(256) for _ in range(n)], [137] * n):
            result = grouped_histogram(digits)
            expected = Counter(digits)
            assert result == [expected[d] for d in range(256)]

    selection_cases = 0
    for n in (257, 511, 1025, 2049, 4099, 8193):
        for mode in range(5):
            # The low word encodes unique descending IDs, as in production.
            high = [rng.getrandbits(32) if mode == 0 else
                    (0xBF800000 | rng.getrandbits(20)) if mode == 1 else
                    (0xC0000000 if mode == 2 else
                     rng.choice((0, 0x80000000, 0xFFFFFFFF)) if mode == 3 else
                     0xFFFFFFFF) for _ in range(n)]
            keys = [(word << 32) | (0xFFFFFFFF - idx)
                    for idx, word in enumerate(high)]
            rng.shuffle(keys)
            for cap in sorted({1, 31, 256, min(2048, n - 1), n - 1, n}):
                select(keys, cap, rng)
                selection_cases += 1
    # Wide screens with mandatory high keys, equal-score ties, and both
    # draft/target capacities. These are synthetic keys, not model outputs.
    for n in (65537, 131073):
        keys = [((0xFFFFFFFF if idx < 65 else
                  0xBF800000 | rng.getrandbits(21)) << 32) |
                (0xFFFFFFFF - idx) for idx in range(n)]
        for cap in (2048, 16384):
            select(keys, cap, rng)
            selection_cases += 1
    print(f"PASS: {hist_cases} histogram/rank cases, 578 warp-tail cases, "
          f"{selection_cases} full top-K comparisons; CUDA not executed")


if __name__ == "__main__":
    main()
