#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ds4-cuda-spark-arch.XXXXXX")
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

cat >"$tmp/capture-make" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" >"$DS4_CAPTURE_ARGS"
EOF
chmod +x "$tmp/capture-make"

check_arch() {
    expected=$1
    mode=$2
    out="$tmp/$mode.args"
    case "$mode" in
        explicit)
            DS4_CAPTURE_ARGS="$out" make -s -C "$root" cuda-spark \
                MAKE="$tmp/capture-make" UNAME_S=Linux CUDA_ARCH=sm_90
            ;;
        absent)
            env -u CUDA_ARCH DS4_CAPTURE_ARGS="$out" make -s -C "$root" cuda-spark \
                MAKE="$tmp/capture-make" UNAME_S=Linux
            ;;
        empty)
            DS4_CAPTURE_ARGS="$out" make -s -C "$root" cuda-spark \
                MAKE="$tmp/capture-make" UNAME_S=Linux CUDA_ARCH=
            ;;
        *)
            echo "unknown mode: $mode" >&2
            exit 2
            ;;
    esac
    actual=$(sed -n 's/^CUDA_ARCH=//p' "$out")
    if [ "$actual" != "$expected" ]; then
        echo "FAIL: cuda-spark $mode propagated CUDA_ARCH=$actual; expected $expected" >&2
        echo "recursive make arguments:" >&2
        sed 's/^/  /' "$out" >&2
        return 1
    fi
}

check_arch sm_90 explicit
check_arch sm_121 absent
check_arch sm_121 empty
printf '%s\n' 'test_cuda_spark_arch PASS'
