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
void* pokemon_pool_create(const char* rom_path, int num_slots, int frame_skip, int hold,
                          int num_threads, int seed);
int64_t pokemon_pool_state_size(void* handle);
int pokemon_pool_num_actions(void* handle);
int pokemon_pool_obs_height(void* handle);
int pokemon_pool_obs_width(void* handle);
int pokemon_pool_advance(void* handle, int count, const uint8_t* states, const int32_t* actions,
                         uint8_t* states_out, uint8_t* frames, uint8_t* ram);
int pokemon_pool_press(void* handle, int slot, int action, int hold, int total);
int pokemon_pool_read(void* handle, int slot, int address);
int pokemon_pool_screen(void* handle, int slot, uint8_t* rgb);
int pokemon_pool_render(void* handle, int count, const uint8_t* states, uint8_t* rgb);
int pokemon_pool_save(void* handle, int slot, uint8_t* out);
int pokemon_pool_load(void* handle, int slot, const uint8_t* in);
int pokemon_pool_observe(void* handle, int slot, uint8_t* frame, uint8_t* ram);
int pokemon_pool_reset(void* handle, int slot);
int pokemon_pool_errors(void* handle, int slot);
void pokemon_pool_destroy(void* handle);
}

namespace {

constexpr int kRamSize = 8192;
constexpr size_t kRgbSize = 144 * 160 * 3;

struct Record {
    void* pool = nullptr;
    size_t state_size = 0;
    size_t frame_size = 0;
    std::vector<uint8_t> states;
    std::vector<uint8_t> states_out;
    std::vector<int32_t> actions;
    std::vector<uint8_t> frames;
    std::vector<uint8_t> ram;
    std::vector<uint8_t> shown;
    std::vector<uint8_t> rgb;
    size_t pinned = 0;

    void room(size_t count) {
        states.resize(count * state_size);
        states_out.resize(count * state_size);
        actions.resize(count);
        frames.resize(count * frame_size);
        ram.resize(count * kRamSize);
    }
};

std::mutex g_mutex;
std::unordered_map<int32_t, Record*> g_records;
int32_t g_next = 1;

Record* find(int32_t handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_records.find(handle);
    return it == g_records.end() ? nullptr : it->second;
}

int32_t Count(const ffi::AnyBuffer& states) {
    auto dims = states.dimensions();
    return dims.size() > 1 ? (int32_t)dims[0] : 1;
}

ffi::Error Missing() {
    return ffi::Error(ffi::ErrorCode::kNotFound, "pokemon: bad handle");
}

ffi::Error Run(Record* record, int32_t count, const uint8_t* states, const int32_t* actions,
               uint8_t* states_out, uint8_t* frames, uint8_t* ram) {
    int rc = pokemon_pool_advance(record->pool, count, states, actions, states_out, frames, ram);
    if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, "pokemon: state serialization failed");
    return ffi::Error::Success();
}

ffi::Error AdvanceCpu(ffi::AnyBuffer states, ffi::AnyBuffer actions,
                      ffi::Result<ffi::AnyBuffer> states_out, ffi::Result<ffi::AnyBuffer> frames,
                      ffi::Result<ffi::AnyBuffer> ram, int32_t handle) {
    Record* record = find(handle);
    if (record == nullptr) return Missing();
    return Run(record, Count(states), (const uint8_t*)states.untyped_data(),
               (const int32_t*)actions.untyped_data(), (uint8_t*)states_out->untyped_data(),
               (uint8_t*)frames->untyped_data(), (uint8_t*)ram->untyped_data());
}

using GpuStream = void*;

struct CudaRuntime {
    int (*memcpy_async)(void* dst, const void* src, size_t count, int kind, GpuStream stream);
    int (*stream_synchronize)(GpuStream stream);
    const char* (*error_string)(int error);
    int (*host_register)(void* ptr, size_t bytes, unsigned int flags);
    int (*host_unregister)(void* ptr);
};

constexpr int kCudaMemcpyHostToDevice = 1;
constexpr int kCudaMemcpyDeviceToHost = 2;

const CudaRuntime* GetCudaRuntime() {
    static const CudaRuntime* runtime = []() -> const CudaRuntime* {
        void* lib = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) lib = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) return nullptr;
        auto* r = new CudaRuntime();
        r->memcpy_async =
            reinterpret_cast<decltype(r->memcpy_async)>(dlsym(lib, "cudaMemcpyAsync"));
        r->stream_synchronize =
            reinterpret_cast<decltype(r->stream_synchronize)>(dlsym(lib, "cudaStreamSynchronize"));
        r->error_string =
            reinterpret_cast<decltype(r->error_string)>(dlsym(lib, "cudaGetErrorString"));
        r->host_register =
            reinterpret_cast<decltype(r->host_register)>(dlsym(lib, "cudaHostRegister"));
        r->host_unregister =
            reinterpret_cast<decltype(r->host_unregister)>(dlsym(lib, "cudaHostUnregister"));
        if (!r->memcpy_async || !r->stream_synchronize || !r->error_string ||
            !r->host_register || !r->host_unregister) {
            delete r;
            return nullptr;
        }
        return r;
    }();
    return runtime;
}

ffi::Error Copy(const CudaRuntime* cuda, GpuStream stream, void* dst, const void* src,
                size_t bytes, int kind) {
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

template <typename T>
void Unpin(const CudaRuntime* cuda, std::vector<T>& held) {
    if (!held.empty()) cuda->host_unregister(held.data());
}

template <typename T>
ffi::Error Pin(const CudaRuntime* cuda, std::vector<T>& held) {
    if (held.empty()) return ffi::Error::Success();
    int rc = cuda->host_register(held.data(), held.size() * sizeof(T), 0);
    if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, cuda->error_string(rc));
    return ffi::Error::Success();
}

ffi::Error Reserve(const CudaRuntime* cuda, Record* record, size_t count) {
    if (record->pinned) {
        Unpin(cuda, record->states);
        Unpin(cuda, record->states_out);
        Unpin(cuda, record->actions);
        Unpin(cuda, record->frames);
        Unpin(cuda, record->ram);
        record->pinned = 0;
    }
    record->room(count);
    for (auto e : {Pin(cuda, record->states), Pin(cuda, record->states_out), Pin(cuda, record->actions),
                   Pin(cuda, record->frames), Pin(cuda, record->ram)})
        if (e.failure()) return e;
    record->pinned = count;
    return ffi::Error::Success();
}

ffi::Error AdvanceCuda(GpuStream stream, ffi::AnyBuffer states, ffi::AnyBuffer actions,
                       ffi::Result<ffi::AnyBuffer> states_out, ffi::Result<ffi::AnyBuffer> frames,
                       ffi::Result<ffi::AnyBuffer> ram, int32_t handle) {
    const CudaRuntime* cuda = GetCudaRuntime();
    if (cuda == nullptr)
        return ffi::Error(ffi::ErrorCode::kUnavailable, "pokemon: libcudart not found");
    Record* record = find(handle);
    if (record == nullptr) return Missing();
    const size_t count = (size_t)Count(states);
    if (count != record->pinned)
        if (auto e = Reserve(cuda, record, count); e.failure()) return e;
    if (auto e = Copy(cuda, stream, record->states.data(), states.untyped_data(),
                      count * record->state_size, kCudaMemcpyDeviceToHost);
        e.failure())
        return e;
    if (auto e = Copy(cuda, stream, record->actions.data(), actions.untyped_data(),
                      count * sizeof(int32_t), kCudaMemcpyDeviceToHost);
        e.failure())
        return e;
    if (auto e = Sync(cuda, stream); e.failure()) return e;
    if (auto e = Run(record, (int32_t)count, record->states.data(), record->actions.data(),
                     record->states_out.data(), record->frames.data(), record->ram.data());
        e.failure())
        return e;
    struct Out {
        void* dst;
        const void* src;
        size_t bytes;
    };
    const Out outs[] = {
        {states_out->untyped_data(), record->states_out.data(), count * record->state_size},
        {frames->untyped_data(), record->frames.data(), count * record->frame_size},
        {ram->untyped_data(), record->ram.data(), count * kRamSize},
    };
    for (const Out& out : outs)
        if (auto e = Copy(cuda, stream, out.dst, out.src, out.bytes, kCudaMemcpyHostToDevice);
            e.failure())
            return e;
    return ffi::Error::Success();
}

ffi::Error RenderCpu(ffi::AnyBuffer states, ffi::Result<ffi::AnyBuffer> rgb, int32_t handle) {
    Record* record = find(handle);
    if (record == nullptr) return Missing();
    pokemon_pool_render(record->pool, Count(states), (const uint8_t*)states.untyped_data(),
                        (uint8_t*)rgb->untyped_data());
    return ffi::Error::Success();
}

ffi::Error RenderCuda(GpuStream stream, ffi::AnyBuffer states, ffi::Result<ffi::AnyBuffer> rgb,
                      int32_t handle) {
    const CudaRuntime* cuda = GetCudaRuntime();
    if (cuda == nullptr)
        return ffi::Error(ffi::ErrorCode::kUnavailable, "pokemon: libcudart not found");
    Record* record = find(handle);
    if (record == nullptr) return Missing();
    const size_t count = (size_t)Count(states);
    record->shown.resize(count * record->state_size);
    record->rgb.resize(count * kRgbSize);
    if (auto e = Copy(cuda, stream, record->shown.data(), states.untyped_data(),
                      record->shown.size(), kCudaMemcpyDeviceToHost);
        e.failure())
        return e;
    if (auto e = Sync(cuda, stream); e.failure()) return e;
    pokemon_pool_render(record->pool, (int32_t)count, record->shown.data(), record->rgb.data());
    if (auto e = Copy(cuda, stream, rgb->untyped_data(), record->rgb.data(), record->rgb.size(),
                      kCudaMemcpyHostToDevice);
        e.failure())
        return e;
    return Sync(cuda, stream);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kRenderCpu, RenderCpu,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int32_t>("handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(kRenderCuda, RenderCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<GpuStream>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int32_t>("handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(kAdvanceCpu, AdvanceCpu,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int32_t>("handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(kAdvanceCuda, AdvanceCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<GpuStream>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int32_t>("handle"));

}  // namespace

extern "C" {

int32_t pokemon_create(const char* rom_path, int32_t num_slots, int32_t frame_skip, int32_t hold,
                       int32_t num_threads, int32_t seed) {
    void* pool = pokemon_pool_create(rom_path, num_slots, frame_skip, hold, num_threads, seed);
    if (pool == nullptr) return -1;
    auto* record = new Record();
    record->pool = pool;
    record->state_size = (size_t)pokemon_pool_state_size(pool);
    record->frame_size = (size_t)pokemon_pool_obs_height(pool) * pokemon_pool_obs_width(pool);
    std::lock_guard<std::mutex> lock(g_mutex);
    int32_t handle = g_next++;
    g_records[handle] = record;
    return handle;
}

int64_t pokemon_state_size(int32_t handle) {
    Record* record = find(handle);
    return record == nullptr ? -1 : (int64_t)record->state_size;
}

int32_t pokemon_num_actions(int32_t handle) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_num_actions(record->pool);
}

int32_t pokemon_obs_height(int32_t handle) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_obs_height(record->pool);
}

int32_t pokemon_obs_width(int32_t handle) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_obs_width(record->pool);
}

int32_t pokemon_press(int32_t handle, int32_t slot, int32_t action, int32_t hold, int32_t total) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_press(record->pool, slot, action, hold, total);
}

int32_t pokemon_read(int32_t handle, int32_t slot, int32_t address) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_read(record->pool, slot, address);
}

int32_t pokemon_screen(int32_t handle, int32_t slot, uint8_t* rgb) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_screen(record->pool, slot, rgb);
}

int32_t pokemon_save(int32_t handle, int32_t slot, uint8_t* out) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_save(record->pool, slot, out);
}

int32_t pokemon_load(int32_t handle, int32_t slot, const uint8_t* in) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_load(record->pool, slot, in);
}

int32_t pokemon_observe(int32_t handle, int32_t slot, uint8_t* frame, uint8_t* ram) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_observe(record->pool, slot, frame, ram);
}

int32_t pokemon_reset(int32_t handle, int32_t slot) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_reset(record->pool, slot);
}

int32_t pokemon_errors(int32_t handle, int32_t slot) {
    Record* record = find(handle);
    return record == nullptr ? -1 : pokemon_pool_errors(record->pool, slot);
}

int32_t pokemon_destroy(int32_t handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_records.find(handle);
    if (it == g_records.end()) return -1;
    pokemon_pool_destroy(it->second->pool);
    delete it->second;
    g_records.erase(it);
    return 0;
}

void* pokemon_ffi_advance_cpu() { return (void*)kAdvanceCpu; }
void* pokemon_ffi_advance_cuda() { return (void*)kAdvanceCuda; }
void* pokemon_ffi_render_cpu() { return (void*)kRenderCpu; }
void* pokemon_ffi_render_cuda() { return (void*)kRenderCuda; }
}
