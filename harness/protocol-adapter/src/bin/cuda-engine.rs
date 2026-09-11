//! `cuda-engine` — the Engine Protocol v1 stdio binary.
//!
//! Reads NDJSON requests on stdin, writes NDJSON responses on stdout.
//!
//! WHICH BACKEND IT SERVES, in the order it decides:
//!
//! 1. THE RESIDENT ENGINE, whenever `DS4_RESIDENT_SOCKET` names a socket. This
//!    is the SCORED path. benchd spawns one of these per phase, and every one
//!    of them attaches to the single `ds4-resident` process `tools/serve-up.sh`
//!    booted for the window: THE MODEL IS LOADED ONCE PER WINDOW, and a phase
//!    costs a connect. It needs no CUDA in this process and no `ds4-engine`
//!    feature; the resident holds the engine.
//! 2. the REAL ds4 backend in-process, under `--features ds4-engine` with no
//!    resident socket. It loads the model itself, before the startup hello, so
//!    the load never sits in a timed window -- but it loads it AGAIN for every
//!    phase, so this is a single-shot diagnostic path, not the scored one.
//! 3. the deterministic MOCK backend on a default build with no socket, so the
//!    protocol runs anywhere with no GPU. THE MOCK IS NOT INFERENCE, and this
//!    binary says so on stderr rather than leaving a reader to infer it from a
//!    suspiciously fast run.
//!
//! EACH OF THE THREE ANNOUNCES ITSELF. The hello's `backend`/`device` strings
//! are what benchd seals as `engine_backend`/`engine_device`, so they are the
//! only place a sealed artifact says which of the three produced the number.
//! The mock says `mock`/`none`, the resident-attached worker says
//! `ds4-resident` with the resident's `load_epoch`, and the in-process build
//! says the serve script's `DS4_ENGINE_IDENT`.

use std::io::{self, BufWriter};
use std::process::ExitCode;

use protocol_adapter::Adapter;

fn cuda_device() -> String {
    // tools/ds4/build.sh supplies DS4_CUDA_ARCH for CUDA builds. Keep the GB10
    // default for a direct Cargo build, but never label an explicitly targeted
    // sm_120 workstation build as sm_121 in the artifact benchd seals.
    format!("cuda {}", option_env!("DS4_CUDA_ARCH").unwrap_or("sm_121"))
}

fn main() -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let out = BufWriter::new(stdout.lock());

    let resident_socket = std::env::var(protocol_adapter::resident::SOCKET_ENV)
        .ok()
        .filter(|s| !s.is_empty());

    let result = if let Some(socket) = resident_socket {
        let (factory, hello, connect) =
            match protocol_adapter::ds4_backend::Ds4Factory::from_resident(&socket) {
                Ok(attached) => attached,
                Err(err) => {
                    eprintln!("cuda-engine: cannot attach to the resident engine: {err}");
                    return ExitCode::FAILURE;
                }
            };
        // The engine identity is the RESIDENT's, because the resident is the
        // process that holds the weights. benchd records it from the hello, and
        // the string names the topology and carries the resident's load_epoch,
        // so the one-load claim reaches the sealed artifact.
        eprintln!(
            "cuda-engine: attached to the resident engine at {socket} in {:.3} ms (load epoch \
             {}, nothing loaded in this process); serving {}",
            connect.as_secs_f64() * 1e3,
            hello.load_epoch,
            hello.ident
        );
        let backend = hello.backend_string();
        Adapter::with_backend(factory, backend, cuda_device()).run(stdin.lock(), out)
    } else {
        #[cfg(not(feature = "ds4-engine"))]
        {
            use protocol_adapter::mock::{MOCK_BACKEND, MOCK_DEVICE};
            eprintln!(
                "cuda-engine: serving the MOCK backend, announced as {MOCK_BACKEND}/{MOCK_DEVICE} \
                 in the hello. Its tokens are a fixed function of the input and its timings mean \
                 nothing. This binary exists to exercise the protocol. Build --features ds4-engine \
                 for the real ds4 backend, or export DS4_RESIDENT_SOCKET to attach to the \
                 window's resident engine."
            );
            let (factory, _log) = protocol_adapter::mock::MockFactory::new();
            Adapter::with_backend(factory, MOCK_BACKEND, MOCK_DEVICE).run(stdin.lock(), out)
        }

        #[cfg(feature = "ds4-engine")]
        {
            let factory = match protocol_adapter::ds4_backend::Ds4Factory::from_env() {
                Ok(factory) => factory,
                Err(err) => {
                    eprintln!("cuda-engine: cannot start the ds4 backend: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let backend = std::env::var("DS4_ENGINE_IDENT")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "ds4".to_string());
            eprintln!(
                "cuda-engine: serving the ds4 backend IN-PROCESS ({backend}); model loaded, \
                 session open. This loads the model per phase -- the scored path boots one \
                 ds4-resident per window (tools/serve-up.sh) and exports DS4_RESIDENT_SOCKET."
            );
            Adapter::with_backend(factory, backend, cuda_device()).run(stdin.lock(), out)
        }
    };

    // Protocol and engine errors are reported inline as `ok:false` responses;
    // run() returns Err only on a hard I/O failure.
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("cuda-engine: I/O error on the protocol stream: {err}");
            ExitCode::FAILURE
        }
    }
}
