#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" {
void* ale_vec_create(int num_envs, const char* rom_path, int frame_skip,
                     int seed, int num_threads, float fraction, int capacity,
                     int depth, int warmup_episodes, float novelty,
                     long long snapshot_every, const char* snapshot_path);
void ale_vec_reset_all(void* handle, uint8_t* ram);
void ale_vec_step(void* handle, const int32_t* actions, float* rewards,
                  uint8_t* dones, uint8_t* warms, uint8_t* ram);
void ale_vec_destroy(void* handle);
}

namespace {

constexpr int kRamSize = 128;

using GpuStream = void*;

struct CudaRuntime {
    int (*memcpy_async)(void* dst, const void* src, size_t count, int kind, GpuStream stream);
    int (*stream_synchronize)(GpuStream stream);
    const char* (*error_string)(int error);
};

constexpr int kCudaMemcpyHostToDevice = 1;
constexpr int kCudaMemcpyDeviceToHost = 2;

const CudaRuntime* GetCudaRuntime() {
    static const CudaRuntime* runtime = []() -> const CudaRuntime* {
        void* lib = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) lib = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) return nullptr;
        auto* r = new CudaRuntime();
        r->memcpy_async = reinterpret_cast<decltype(r->memcpy_async)>(dlsym(lib, "cudaMemcpyAsync"));
        r->stream_synchronize =
            reinterpret_cast<decltype(r->stream_synchronize)>(dlsym(lib, "cudaStreamSynchronize"));
        r->error_string =
            reinterpret_cast<decltype(r->error_string)>(dlsym(lib, "cudaGetErrorString"));
        if (!r->memcpy_async || !r->stream_synchronize || !r->error_string) {
            delete r;
            return nullptr;
        }
        return r;
    }();
    return runtime;
}

struct Record {
    void* vec;
    int32_t num_envs;
    std::vector<int32_t> actions;
    std::vector<float> rewards;
    std::vector<uint8_t> dones;
    std::vector<uint8_t> warms;
    std::vector<uint8_t> ram;
};

std::mutex g_mutex;
std::unordered_map<int32_t, Record*> g_records;
int32_t g_next = 1;

Record* MakeRecord(int64_t num_envs, std::string_view rom, int64_t frame_skip,
                   int64_t seed, int64_t threads, int64_t fraction_ppm,
                   int64_t capacity, int64_t depth, int64_t warmup_episodes,
                   int64_t novelty_ppm, int64_t snapshot_every,
                   std::string_view snapshot) {
    auto* r = new Record();
    r->num_envs = (int32_t)num_envs;
    r->actions.assign(num_envs, 0);
    r->rewards.assign(num_envs, 0.0f);
    r->dones.assign(num_envs, 0);
    r->warms.assign(num_envs, 0);
    r->ram.assign(num_envs * kRamSize, 0);
    std::string rom_path(rom);
    std::string snapshot_path(snapshot);
    r->vec = ale_vec_create(
        (int)num_envs, rom_path.c_str(), (int)frame_skip, (int)seed,
        (int)threads, (float)fraction_ppm / 1e6f, (int)capacity, (int)depth,
        (int)warmup_episodes, (float)novelty_ppm / 1e6f,
        (long long)snapshot_every, snapshot_path.c_str());
    ale_vec_reset_all(r->vec, r->ram.data());
    return r;
}

ffi::Error Find(int32_t handle, Record** out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_records.find(handle);
    if (it == g_records.end())
        return ffi::Error(ffi::ErrorCode::kNotFound, "ale: bad handle");
    *out = it->second;
    return ffi::Error::Success();
}

int32_t Store(Record* r) {
    std::lock_guard<std::mutex> lock(g_mutex);
    int32_t handle = g_next++;
    g_records[handle] = r;
    return handle;
}

void Drop(int32_t handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_records.find(handle);
    if (it != g_records.end()) {
        ale_vec_destroy(it->second->vec);
        delete it->second;
        g_records.erase(it);
    }
}

ffi::Error InitCpu(ffi::AnyBuffer seed, int64_t num_envs, std::string_view rom,
                   int64_t frame_skip, int64_t threads, int64_t fraction_ppm,
                   int64_t capacity, int64_t depth, int64_t warmup_episodes,
                   int64_t novelty_ppm, int64_t snapshot_every,
                   std::string_view snapshot,
                   ffi::Result<ffi::AnyBuffer> handleOut,
                   ffi::Result<ffi::AnyBuffer> ram) {
    Record* r = MakeRecord(num_envs, rom, frame_skip,
                           *static_cast<const int32_t*>(seed.untyped_data()),
                           threads, fraction_ppm, capacity, depth,
                           warmup_episodes, novelty_ppm, snapshot_every,
                           snapshot);
    int32_t handle = Store(r);
    *(int32_t*)handleOut->untyped_data() = handle;
    std::memcpy(ram->untyped_data(), r->ram.data(), r->ram.size());
    return ffi::Error::Success();
}

ffi::Error StepCpu(ffi::AnyBuffer handleBuf, ffi::AnyBuffer actions,
                   ffi::Result<ffi::AnyBuffer> handleOut,
                   ffi::Result<ffi::AnyBuffer> ram,
                   ffi::Result<ffi::AnyBuffer> reward,
                   ffi::Result<ffi::AnyBuffer> done,
                   ffi::Result<ffi::AnyBuffer> warm) {
    int32_t handle = *(const int32_t*)handleBuf.untyped_data();
    Record* r = nullptr;
    if (auto err = Find(handle, &r); err.failure()) return err;
    ale_vec_step(r->vec, (const int32_t*)actions.untyped_data(),
                 (float*)reward->untyped_data(),
                 (uint8_t*)done->untyped_data(),
                 (uint8_t*)warm->untyped_data(),
                 (uint8_t*)ram->untyped_data());
    *(int32_t*)handleOut->untyped_data() = handle;
    return ffi::Error::Success();
}

ffi::Error CloseCpu(ffi::AnyBuffer handleBuf,
                    ffi::Result<ffi::AnyBuffer> handleOut) {
    int32_t handle = *(const int32_t*)handleBuf.untyped_data();
    Drop(handle);
    *(int32_t*)handleOut->untyped_data() = handle;
    return ffi::Error::Success();
}

ffi::Error MissingCuda() {
    return ffi::Error(ffi::ErrorCode::kUnavailable, "ale: libcudart not found");
}

ffi::Error Copy(const CudaRuntime* cuda, GpuStream stream, void* dst,
                const void* src, size_t bytes, int kind) {
    if (bytes == 0) return ffi::Error::Success();
    int rc = cuda->memcpy_async(dst, src, bytes, kind, stream);
    if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, cuda->error_string(rc));
    return ffi::Error::Success();
}

ffi::Error Sync(const CudaRuntime* cuda, GpuStream stream) {
    int rc = cuda->stream_synchronize(stream);
    if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, cuda->error_string(rc));
    return ffi::Error::Success();
}

ffi::Error InitCuda(GpuStream stream, ffi::AnyBuffer seedBuf, int64_t num_envs,
                    std::string_view rom, int64_t frame_skip, int64_t threads,
                    int64_t fraction_ppm, int64_t capacity, int64_t depth,
                    int64_t warmup_episodes, int64_t novelty_ppm,
                    int64_t snapshot_every, std::string_view snapshot,
                    ffi::Result<ffi::AnyBuffer> handleOut,
                    ffi::Result<ffi::AnyBuffer> ram) {
    const CudaRuntime* cuda = GetCudaRuntime();
    if (cuda == nullptr) return MissingCuda();
    int32_t seed = 0;
    if (auto e = Copy(cuda, stream, &seed, seedBuf.untyped_data(), sizeof(seed),
                      kCudaMemcpyDeviceToHost); e.failure())
        return e;
    if (auto e = Sync(cuda, stream); e.failure()) return e;
    Record* r = MakeRecord(num_envs, rom, frame_skip, seed, threads,
                           fraction_ppm, capacity, depth, warmup_episodes,
                           novelty_ppm, snapshot_every, snapshot);
    int32_t handle = Store(r);
    if (auto e = Copy(cuda, stream, handleOut->untyped_data(), &handle,
                      sizeof(handle), kCudaMemcpyHostToDevice); e.failure())
        return e;
    if (auto e = Copy(cuda, stream, ram->untyped_data(), r->ram.data(),
                      r->ram.size(), kCudaMemcpyHostToDevice); e.failure())
        return e;
    return Sync(cuda, stream);
}

ffi::Error StepCuda(GpuStream stream, ffi::AnyBuffer handleBuf,
                    ffi::AnyBuffer actions,
                    ffi::Result<ffi::AnyBuffer> handleOut,
                    ffi::Result<ffi::AnyBuffer> ram,
                    ffi::Result<ffi::AnyBuffer> reward,
                    ffi::Result<ffi::AnyBuffer> done,
                    ffi::Result<ffi::AnyBuffer> warm) {
    const CudaRuntime* cuda = GetCudaRuntime();
    if (cuda == nullptr) return MissingCuda();
    int32_t handle = 0;
    if (auto e = Copy(cuda, stream, &handle, handleBuf.untyped_data(),
                      sizeof(handle), kCudaMemcpyDeviceToHost); e.failure())
        return e;
    if (auto e = Sync(cuda, stream); e.failure()) return e;
    Record* r = nullptr;
    if (auto err = Find(handle, &r); err.failure()) return err;
    if (auto e = Copy(cuda, stream, r->actions.data(), actions.untyped_data(),
                      r->actions.size() * sizeof(int32_t),
                      kCudaMemcpyDeviceToHost); e.failure())
        return e;
    if (auto e = Sync(cuda, stream); e.failure()) return e;
    ale_vec_step(r->vec, r->actions.data(), r->rewards.data(),
                 r->dones.data(), r->warms.data(), r->ram.data());
    if (auto e = Copy(cuda, stream, handleOut->untyped_data(), &handle,
                      sizeof(handle), kCudaMemcpyHostToDevice); e.failure())
        return e;
    if (auto e = Copy(cuda, stream, ram->untyped_data(), r->ram.data(),
                      r->ram.size(), kCudaMemcpyHostToDevice); e.failure())
        return e;
    if (auto e = Copy(cuda, stream, reward->untyped_data(), r->rewards.data(),
                      r->rewards.size() * sizeof(float),
                      kCudaMemcpyHostToDevice); e.failure())
        return e;
    if (auto e = Copy(cuda, stream, done->untyped_data(), r->dones.data(),
                      r->dones.size(), kCudaMemcpyHostToDevice); e.failure())
        return e;
    if (auto e = Copy(cuda, stream, warm->untyped_data(), r->warms.data(),
                      r->warms.size(), kCudaMemcpyHostToDevice); e.failure())
        return e;
    return Sync(cuda, stream);
}

ffi::Error CloseCuda(GpuStream stream, ffi::AnyBuffer handleBuf,
                     ffi::Result<ffi::AnyBuffer> handleOut) {
    const CudaRuntime* cuda = GetCudaRuntime();
    if (cuda == nullptr) return MissingCuda();
    int32_t handle = 0;
    if (auto e = Copy(cuda, stream, &handle, handleBuf.untyped_data(),
                      sizeof(handle), kCudaMemcpyDeviceToHost); e.failure())
        return e;
    if (auto e = Sync(cuda, stream); e.failure()) return e;
    Drop(handle);
    if (auto e = Copy(cuda, stream, handleOut->untyped_data(), &handle,
                      sizeof(handle), kCudaMemcpyHostToDevice); e.failure())
        return e;
    return Sync(cuda, stream);
}

#define ALE_ATTRS                                                              \
    .Attr<int64_t>("num_envs")                                                 \
        .Attr<std::string_view>("rom")                                         \
        .Attr<int64_t>("frame_skip")                                           \
        .Attr<int64_t>("threads")                                              \
        .Attr<int64_t>("fraction_ppm")                                         \
        .Attr<int64_t>("capacity")                                             \
        .Attr<int64_t>("depth")                                                \
        .Attr<int64_t>("warmup_episodes")                                      \
        .Attr<int64_t>("novelty_ppm")                                          \
        .Attr<int64_t>("snapshot_every")                                       \
        .Attr<std::string_view>("snapshot")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kInitCpu, InitCpu,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>() ALE_ATTRS.Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kStepCpu, StepCpu,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kCloseCpu, CloseCpu,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kInitCuda, InitCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<GpuStream>>()
        .Arg<ffi::AnyBuffer>() ALE_ATTRS.Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kStepCuda, StepCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<GpuStream>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kCloseCuda, CloseCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<GpuStream>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

}  // namespace

extern "C" {
void* ale_ffi_init_cpu() { return (void*)kInitCpu; }
void* ale_ffi_step_cpu() { return (void*)kStepCpu; }
void* ale_ffi_close_cpu() { return (void*)kCloseCpu; }
void* ale_ffi_init_cuda() { return (void*)kInitCuda; }
void* ale_ffi_step_cuda() { return (void*)kStepCuda; }
void* ale_ffi_close_cuda() { return (void*)kCloseCuda; }
}
