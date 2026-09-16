/* Native projection exactness and changed-input graph replays, no GGUF.
 * Reuse the broad exact-Q8 fixture with the new narrow kernel forced on.
 * Inference never sets this override; startup verification/timing selects it. */
#define main q8_fixture_main
#include "test_q8_decode_pairs.c"
#undef main
int main(void) {
    require(setenv("DS4_Q8_FORCE_HC_UP_MMA", "1", 1) == 0, "force new exact MMA");
    require(unsetenv("DS4_Q8_NO_HC_UP_MMA") == 0, "clear optout");
    return q8_fixture_main();
}
