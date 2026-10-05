# Phase 0 Spike Results

Device: RedMagic 10 Pro (NX789J, Snapdragon 8 Elite, 16 GB visible RAM, Android 15, qcom)
App: `com.mobilerag` debug build, minSdk 30, arm64-v8a only
Run date: 2026-09-12

## On-device results — ALL FIVE SPIKES PASS

| Spike | Verdict | Key metrics |
|---|---|---|
| A — ONNX Runtime embeddings (MiniLM-L6 int8 arm64) | **PASS** | model load 163 ms; avg inference 16 ms; sane cosine ordering (cat/kittens 0.326 > cat/dog 0.163 > cat/taxes 0.096); 384 dims |
| B — LadybugDB Cypher round-trip (our cross-compiled build) | **PASS** | native load + open db 43 ms; schema+insert 47 ms; 2-hop MATCH 12 ms |
| B-fallback — SQLite recursive CTE + FTS | **PASS** (with fixes) | insert 288 ms; 2-hop traversal 2 ms; FTS 'Acme*' works |
| C — llama.cpp Qwen3-0.6B Q8_0 (610 MB GGUF) | **PASS** | mmap load 3.85 s; TTFT 45 ms; **decode 78.2 tok/s**; 197 tokens streamed |
| ML Kit Gemini Nano probe | **PASS** (probe) | `FeatureStatus.UNAVAILABLE` — no AICore on this device, as the plan doc predicted |

## On-device incidents and fixes

1. **Stock Android SQLite has no FTS5** (`no such module: fts5`). Fell back to FTS4, which works. Consequence: if we ever go SQLite-first, keyword search must use FTS4, or we bundle our own SQLite build. A point in LadybugDB's favor (its FTS is native, version-controlled by us).
2. **adb-pushed files are invisible to the app** on this device: files pushed to `/sdcard/Android/data/<pkg>/` are owned by `shell`, and SELinux/FUSE blocks the app from reading them (chmod alone insufficient, shell cannot chown there, `run-as` cannot write to external storage). Working delivery path for models: `adb push <file> /data/local/tmp/` then `adb shell run-as com.mobilerag sh -c 'mkdir -p files/models && cp /data/local/tmp/<file> files/models/'` (internal files dir, app-owned).
3. SQLite spike test assertion was wrong (Q3 Report is 2 hops, not 3, from Alice) — fixed the assertion, not the engine.

## Build-time findings (verified 2026-09-12)

### LadybugDB — doc claim partially stale, self-build succeeded

- The plan doc (and LadybugDB docs) claim the Java API "works on Android ARMv8-A with binaries compiled for API level 21". **No such binary is published**: the Maven jar (`com.ladybugdb:lbug:0.20.3`) contains only glibc-linked `linux/osx/windows` .so files, and no CI workflow builds an Android variant. The `Native` loader detects Android but expects a `liblbug_java_native.so_android_arm64` resource that does not exist in the jar.
- **We cross-compiled it ourselves** (artifacts in `prebuilt/ladybug/arm64-v8a/`):
  - Core `liblbug.so` from tag `v0.20.3` with NDK 27.2 (API 26, arm64-v8a), OpenSSL 3.5.2 built for Android, NEON SIMD enabled. Two `std::atomic_ref` sites patched to `__atomic_exchange_n` (NDK libc++ lacks `atomic_ref`). Stripped to 23 MB.
  - JNI binding from `ladybug-java@f2fb39f` (the exact submodule commit at core tag v0.20.3), compiled directly with NDK clang++ against `liblbug.so` (186 JNI symbols, bionic-linked).
  - Packaged: `app/src/main/jniLibs/arm64-v8a/{liblbug.so,libc++_shared.so}` + `app/src/main/resources/liblbug_java_native.so_android_arm64`.
- Reproduction recipe at the bottom of this file. Upstream worthiness: the `atomic_ref` patch and an Android CI job would be a genuinely useful contribution.

### llama.cpp — official Android path works

- Vendored `examples/llama.android/lib` as the `:llamacpp` module (Arm's `InferenceEngine` API: `loadModel` / `sendUserPrompt` → `Flow<String>`).
- Adjustments: minSdk 30 (`__android_log_is_loggable` needs API 30), arm64-v8a only, NDK 27.2, CMake 3.31.6.
- 78 tok/s decode on Qwen3-0.6B Q8_0 confirms the CPU path is fast; OpenCL/Adreno variant not needed for 0.6B but worth testing at 1.7B+.

### Generation backends

- `GenerationBackend` interface implemented by `LlamaCppBackend` and `MlKitGenerationBackend` (Gemini Nano via `com.google.mlkit:genai-prompt:1.0.0-beta4`).
- **ML Kit is confirmed UNAVAILABLE on RedMagic** (AICore device list excludes nubia) — llama.cpp is the primary backend; ML Kit stays as opportunistic fallback for other devices.
- GenieX/QNN NPU benchmark deferred (requires Qualcomm Developer account). With 78 tok/s on 0.6B CPU, the NPU case must be made at 1.7B–8B model sizes.

## Decisions (confirmed 2026-09-12)

- **Storage: LadybugDB primary, SQLite interface-compatible fallback.** LadybugDB passed on-device with better traversal ergonomics (Cypher vs hand-written recursive CTEs), native FTS (stock SQLite lacks FTS5), and a native vector index for later phases. Cost: we now maintain a private native build — mitigated by the `GraphStore` interface (SQLite impl already exists and passes) and pinned artifacts in `prebuilt/`.
- **Generation: llama.cpp GGUF primary; ML Kit opportunistic where AICore exists; GenieX deferred to Phase 4.** 78 tok/s at 0.6B and 45 ms TTFT exceed the interactivity bar for Phase 1. Next model to benchmark: Qwen3-1.7B Q6_K (~1.4 GB) — the plan doc's "balanced" target.

## LadybugDB Android build recipe

```bash
# 1. OpenSSL 3.5.2 for Android (static)
export ANDROID_NDK_ROOT=~/Library/Android/sdk/ndk/27.2.12479018
export PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin:$PATH"
./Configure android-arm64 -D__ANDROID_API__=26 no-shared no-tests --prefix=$PWD/install
make -j8 build_libs && make install_sw

# 2. Core (tag v0.20.3); patch the two std::atomic_ref sites in src/c_api/{connection,database}.cpp
#    to __atomic_exchange_n(&ptr, nullptr, __ATOMIC_SEQ_CST)
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHELL=OFF -DBUILD_SINGLE_FILE_HEADER=OFF -DAUTO_UPDATE_GRAMMAR=OFF \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=OFF -DBUILD_EXTENSIONS="" \
  -DOPENSSL_INCLUDE_DIR=<openssl>/install/include \
  -DOPENSSL_CRYPTO_LIBRARY=<openssl>/install/lib/libcrypto.a \
  -DOPENSSL_SSL_LIBRARY=<openssl>/install/lib/libssl.a
cmake --build build-android -j8 --target lbug_shared   # NOT 'lbug' (static bundle uses macOS libtool)

# 3. JNI binding (ladybug-java @ f2fb39f)
javac -h jni-headers -cp "<arrow jars>" src/main/java/com/lbugdb/*.java
aarch64-linux-android26-clang++ -std=gnu++20 -fPIC -shared -O2 \
  -I jni-headers -I $JAVA_HOME/include -I $JAVA_HOME/include/darwin \
  -I <core>/src/include -I <core>/build-android/src/include \
  src/jni/lbug_java.cpp -L <core>/build-android/src -llbug -llog \
  -Wl,-soname,liblbug_java_native.so -o liblbug_java_native.so
```
