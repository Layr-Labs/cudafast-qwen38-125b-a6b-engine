#!/usr/bin/env python3
"""Model-free regressions for descriptor checks and image size handling."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
build = root / '.build/security-io'
build.mkdir(parents=True, exist_ok=True)
source = (root / 'ds4/ds4.c').read_text()
def extract(name):
    start = source.index('static bool ' + name + '(')
    end = source.index('\n}\n', start) + 3
    return source[start:end]

code = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#define PNG_IMPLEMENTATION
#include "ds4/third_party/iris/png.h"
#define JPEG_IMPLEMENTATION
#include "ds4/third_party/iris/jpeg.h"
static void *xmalloc(size_t n) { void *p = malloc(n); assert(p); return p; }
static int swap_on_open;
static FILE *race_open(const char *path, const char *mode) {
    FILE *fp = fopen(path, mode);
    if (fp && swap_on_open) {
        swap_on_open = 0;
        assert(rename("replacement.bin", path) == 0);
    }
    return fp;
}
#define fopen race_open
'''
code += extract('read_f32_binary_file') + extract('imatrix_read_text_file')
code += r'''
#undef fopen
static void put(const char *path, const void *data, size_t n) {
    FILE *fp = fopen(path, "wb"); assert(fp);
    assert(fwrite(data, 1, n, fp) == n); assert(fclose(fp) == 0);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    float input[2] = {1.25f, -2.5f}, output[2] = {0};
    put("data.bin", input, sizeof(input));
    assert(read_f32_binary_file("data.bin", output, 2));
    assert(memcmp(input, output, sizeof(input)) == 0);
    assert(!read_f32_binary_file("data.bin", output, UINT64_MAX));
    assert(!read_f32_binary_file("data.bin", output, 1));
    // Replacing the path after open must not change which size is validated.
    put("replacement.bin", "x", 1); swap_on_open = 1;
    assert(read_f32_binary_file("data.bin", output, 2));
    assert(memcmp(input, output, sizeof(input)) == 0);
    put("data.bin", "original", 8); put("replacement.bin", "x", 1);
    swap_on_open = 1; char *text = NULL; size_t len = 0;
    assert(imatrix_read_text_file("data.bin", &text, &len));
    assert(len == 8 && strcmp(text, "original") == 0); free(text);
    put("data.bin", "", 0);
    assert(imatrix_read_text_file("data.bin", &text, &len));
    assert(len == 0 && text[0] == 0); free(text);
    for (int channels = 1; channels <= 4; channels++) {
        png_image *p = png_create(3, 2, channels); assert(p);
        for (int i = 0; i < 6*channels; i++) p->data[i] = i*13;
        assert(png_save(p, "roundtrip.png") == 0);
        png_image *q = png_load("roundtrip.png"); assert(q);
        assert(q->width == 3 && q->height == 2 && q->channels == channels);
        assert(memcmp(p->data, q->data, 6*channels) == 0);
        png_free(q); png_free(p);
    }
    uint8_t byte = 0; png_image bad = {INT_MAX, INT_MAX, 4, &byte};
    FILE *fp = tmpfile(); assert(fp);
    assert(png_save_internal(&bad, fp, NULL, NULL) == -1);
    assert(ftell(fp) == 0); fclose(fp);
    jpeg_image *j = jpeg_load(argv[1]); assert(j); jpeg_free(j);
    unlink("data.bin"); unlink("roundtrip.png");
    puts("PASS descriptor replacement, length guards, PNG round trips/rejection, JPEG fixture");
}
'''
(build / 'test.c').write_text(code)
subprocess.run(['cc', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O1', '-g',
    '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-I' + str(root),
    str(build / 'test.c'), '-lm', '-o', str(build / 'test')], check=True)
with tempfile.TemporaryDirectory(prefix='ds4-security-') as temporary:
    subprocess.run([str(build / 'test'), str(root / 'ds4/tests/vision-fixtures/glm53/earth.jpg')],
        cwd=temporary, check=True)
