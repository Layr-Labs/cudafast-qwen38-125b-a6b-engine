/* Native key, selection, full-refinement and nonfinite-fallback parity.
 * Reuse the broad original fixture with the exact one-warp screen forced on. */
#define main old_native_screen_fixture_main
#include "test_mtp_native_screen.c"
#undef main
int main(void) {
    need(setenv("DS4_MTP_FORCE_SCREEN_WARP", "1", 1) == 0, "force one-warp screen");
    need(unsetenv("DS4_MTP_NO_SCREEN_WARP") == 0, "clear MMA optout");
    return old_native_screen_fixture_main();
}
