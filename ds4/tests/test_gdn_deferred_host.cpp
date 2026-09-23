/* CPU-only policy test. Ordered token histories form an independent oracle:
 * no floating point, CUDA kernel, model, or benchmark acceptance data needed.
 * Example: c++ -std=c++17 -O2 -Wall -Wextra -pedantic -Ids4
 *   ds4/tests/test_gdn_deferred_host.cpp -o /tmp/test_gdn_deferred_host */
#include "ds4_qwen4exp_gdn_deferred.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

using history = std::vector<uint64_t>;
static unsigned cases = 0, forwards = 0, inspections = 0;
[[noreturn]] static void fail(const char *message) {
    std::fprintf(stderr, "FAIL deferred host: %s (%u cases, %u forwards)\n",
                 message, cases, forwards);
    std::exit(1);
}
static void need(bool value, const char *message) { if (!value) fail(message); }

struct fixture {
    std::array<history, 2> storage;
    std::array<std::array<uint64_t, 4>, 2> tape{};
    std::array<history, 6> snapshots, conv_snapshots, expected_snapshots;
    history conv, expected, expected_conv, expected_final;
    unsigned live = 0, checkpoint = 1, prefix = 0, bank = 0;
    unsigned recurrent = 0, convolution = 0;
    bool previous = false, enabled = true;
    uint64_t next_token = 1;
    std::map<uint32_t, unsigned> graph_addresses;

    history materialize(unsigned rows, unsigned log_bank) const {
        need(rows <= 4 && log_bank < 2, "materialization bounds");
        history h = storage[checkpoint];
        for (unsigned r = 0; r < rows; ++r) h.push_back(tape[log_bank][r]);
        return h;
    }
    history final_state() const {
        if (!previous) return storage[live];
        const auto out = ds4_qwen4exp_gdn_deferred_output_plan(prefix, bank);
        need(out.valid, "final metadata");
        return materialize(out.final_rows, out.bank);
    }
    history snapshot(unsigned row) const {
        need(row < snapshots.size(), "snapshot bounds");
        if (!previous) return snapshots[row];
        need(row == 0, "only deferred row zero exists");
        const auto out = ds4_qwen4exp_gdn_deferred_output_plan(prefix, bank);
        need(out.valid, "snapshot metadata");
        return materialize(out.snapshot_rows, out.bank);
    }
    void inspect() {
        const unsigned p = prefix, b = bank, cp = checkpoint;
        const bool was_previous = previous;
        // Publishing final for diagnostics must not destroy snapshot ancestry.
        storage[live] = final_state();
        need(final_state() == expected_final,
             "inspection final state");
        need((recurrent ? snapshot(recurrent - 1) : final_state()) == expected,
             "inspection selected state");
        if (previous) {
            need(snapshot(0) == expected_snapshots[0], "inspection snapshot");
            need(final_state() == storage[live], "snapshot changed final");
        }
        need(prefix == p && bank == b && checkpoint == cp && previous == was_previous,
             "inspection changed controller");
        ++inspections;
    }
    void select(unsigned row, bool state, bool convolution_state) {
        need(row < snapshots.size(), "selection bounds");
        if (state) { recurrent = row + 1; expected = expected_snapshots[row]; }
        if (convolution_state) {
            convolution = row + 1; expected_conv = conv_snapshots[row];
        }
    }
    void forward(unsigned width = 2, unsigned armed = 1, bool inspect_after = true) {
        need(width > 0 && armed < width && armed <= snapshots.size(), "test width");
        const auto step = ds4_qwen4exp_gdn_deferred_plan(
            enabled, previous, prefix, bank, width, armed, recurrent, convolution);
        need(step.valid, "valid controller rejected");
        if (step.needs_final) storage[live] = final_state();
        if (step.settle) {
            if (recurrent) storage[live] = snapshot(recurrent - 1);
            if (convolution) conv = conv_snapshots[convolution - 1];
        } else if (convolution) conv = conv_snapshots[convolution - 1];
        if (step.swap) std::swap(live, checkpoint);
        prefix = step.prefix; bank = step.bank;
        const uint32_t key = width | (armed << 8) | (live << 16) |
                             (unsigned(step.active) << 17);
        auto inserted = graph_addresses.emplace(key, live);
        need(inserted.second || inserted.first->second == live, "graph pointer changed");

        history h = step.active ? materialize(prefix, bank) : storage[live];
        need(h == expected && conv == expected_conv, "selected input history");
        const bool fold = step.active && prefix >= 2;
        const unsigned destination = bank ^ unsigned(fold);
        for (unsigned r = 0; r < width; ++r) {
            const uint64_t token = next_token++;
            h.push_back(token); expected.push_back(token);
            conv.push_back(token + 1000000); expected_conv.push_back(token + 1000000);
            need(h == expected && conv == expected_conv, "current row output");
            if (r < armed) {
                expected_snapshots[r] = expected;
                conv_snapshots[r] = conv;
                if (!step.active) snapshots[r] = h;
            }
            if (step.active) {
                if (fold && r == 0) storage[checkpoint] = h;
                else {
                    const unsigned slot = fold ? 0 : prefix + r;
                    need(slot < 4, "log store bounds");
                    tape[destination][slot] = token;
                }
            }
        }
        if (!step.active) storage[live] = h;
        previous = step.active;
        recurrent = convolution = 0;
        expected_final = expected;
        ++forwards;
        need(final_state() == expected, "virtual final history");
        if (armed) need(snapshot(0) == expected_snapshots[0], "virtual row-zero history");
        if (inspect_after) { inspect(); inspect(); }
    }
    void reset() {
        const unsigned parity = live;
        storage[live].clear(); conv.clear(); expected.clear(); expected_conv.clear(); expected_final.clear();
        previous = false; prefix = bank = recurrent = convolution = 0;
        need(live == parity, "reset pointer parity");
    }
};

static void bounds_and_flags() {
    for (unsigned p = 0; p < 8; ++p) for (unsigned b = 0; b < 4; ++b) {
        const auto out = ds4_qwen4exp_gdn_deferred_output_plan(p,b);
        need(out.valid == (p < 4 && b < 2), "output validity");
        if (out.valid) {
            need(out.final_rows > 0 && out.final_rows <= 3, "final rows bounded");
            need(out.snapshot_rows + 1 == out.final_rows, "row-zero precedes final");
        }
        for (unsigned e = 0; e < 2; ++e) for (unsigned prev = 0; prev < 2; ++prev)
        for (unsigned w : {1u,2u,3u,8u}) for (unsigned n : {0u,1u,2u})
        for (unsigned rs = 0; rs < 3; ++rs) for (unsigned rc = 0; rc < 3; ++rc) {
            const auto s = ds4_qwen4exp_gdn_deferred_plan(e,prev,p,b,w,n,rs,rc);
            need(s.valid == (!prev || out.valid), "plan validity");
            if (!s.valid) {
                need(!s.active && !s.swap && !s.reuse && !s.needs_final && !s.settle,
                     "invalid plan requested mutation");
                continue;
            }
            const bool active = e && w == 2 && n == 1;
            const bool reuse = active && prev && rs == rc && rs <= 1;
            need(s.active == active && s.reuse == reuse, "dispatch eligibility");
            need(s.needs_final == (prev && !reuse), "exit final publication");
            need(s.settle == (!reuse && (rs || rc)), "pending selector settlement");
            need(s.swap == (active && !reuse), "checkpoint entry swap");
            need(s.prefix <= 3 && s.bank <= 1, "next metadata bounds");
            if (!reuse) need(!s.prefix && !s.bank, "fresh entry metadata");
        }
    }
    need(!ds4_qwen4exp_gdn_deferred_output_plan(UINT32_MAX,0).valid, "prefix overflow");
    need(!ds4_qwen4exp_gdn_deferred_output_plan(0,UINT32_MAX).valid, "bank overflow");
}

int main() {
    bounds_and_flags();
    // Every acceptance sequence of length ten, with and without diagnostic
    // publication. Repeated partial selects exercise last-selection-wins.
    for (unsigned mask = 0; mask < 1024; ++mask) for (unsigned inspect = 0; inspect < 2; ++inspect) {
        fixture f; f.forward(1,0);
        for (unsigned r = 0; r < 10; ++r) {
            f.forward(2,1,inspect != 0);
            if (!(mask & (1u << r))) {
                f.select(0,true,false); f.select(0,false,true); f.select(0,true,true);
            }
        }
        f.forward(1,0); f.reset(); f.reset(); f.forward(); ++cases;
    }
    // Reach each prefix/bank through real histories, then exercise every exit,
    // including conv-only rollback (which must retain FINAL recurrent state).
    for (unsigned mask = 0; mask < 64; ++mask) for (unsigned rs = 0; rs < 2; ++rs)
    for (unsigned rc = 0; rc < 2; ++rc) for (unsigned kind = 0; kind < 5; ++kind)
    for (unsigned inspect = 0; inspect < 2; ++inspect) {
        fixture f;
        for (unsigned r = 0; r < 6; ++r) {
            f.forward(2,1,inspect != 0);
            if (!(mask & (1u << r))) f.select(0,true,true);
        }
        // Complete one verify so earlier acceptance flags do not leak.
        f.forward(2,1,inspect != 0);
        if (rs) f.select(0,true,false);
        if (rc) f.select(0,false,true);
        if (inspect) { f.inspect(); f.inspect(); }
        if (kind == 0) f.forward(1,0);
        if (kind == 1) f.forward(2,0);
        if (kind == 2) f.forward(3,2);
        if (kind == 3) { f.enabled = false; f.forward(); }
        if (kind == 4) f.forward();
        f.enabled = true; f.forward();
        f.reset(); f.forward(); ++cases;
    }
    // Ordinary wider snapshots must settle before entering deferred mode.
    for (unsigned rs = 0; rs < 6; ++rs) for (unsigned rc = 0; rc < 6; ++rc) {
        fixture f; f.forward(7,6);
        f.select(rs,true,false); f.select(rc,false,true);
        f.forward(); f.forward(); ++cases;
    }
    std::printf("PASS deferred host: %u histories, %u forwards, %u inspections; bounds, exits, mixed adoptions, resets\n",
                cases, forwards, inspections);
}
