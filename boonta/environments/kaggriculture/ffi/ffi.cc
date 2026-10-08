#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cstdio>
#include <dlfcn.h>

#include "xla/ffi/api/ffi.h"
#include "kaggriculture.h"

namespace ffi = xla::ffi;

namespace {

using GpuStream = void*;

struct CudaRuntime {
	int (*memcpy_async)(void* dst, const void* src, size_t count, int kind, GpuStream stream);
	int (*stream_synchronize)(GpuStream stream);
	const char* (*error_string)(int error);
	int (*host_register)(void* ptr, size_t size, unsigned int flags);
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
		r->memcpy_async = reinterpret_cast<decltype(r->memcpy_async)>(dlsym(lib, "cudaMemcpyAsync"));
		r->stream_synchronize =
			reinterpret_cast<decltype(r->stream_synchronize)>(dlsym(lib, "cudaStreamSynchronize"));
		r->error_string =
			reinterpret_cast<decltype(r->error_string)>(dlsym(lib, "cudaGetErrorString"));
		r->host_register =
			reinterpret_cast<decltype(r->host_register)>(dlsym(lib, "cudaHostRegister"));
		r->host_unregister =
			reinterpret_cast<decltype(r->host_unregister)>(dlsym(lib, "cudaHostUnregister"));
		if (!r->memcpy_async || !r->stream_synchronize || !r->error_string) {
			delete r;
			return nullptr;
		}
		return r;
	}();
	return runtime;
}

template <typename T>
size_t bytes_of(const std::vector<T>& v) { return v.size() * sizeof(T); }

struct Record {
	std::vector<void*> pinned;
	kg_vec_t* vec;
	int32_t n_envs;
	std::vector<int8_t> tiles, lifespan, pos, shops, terminated, day, hour;
	std::vector<int16_t> inv, shed, seeds, hands, hires, quadrants, prices;
	std::vector<int32_t> market, heads;
	std::vector<float> money, reward;
};

std::mutex g_mutex;
std::unordered_map<int32_t, Record*> g_records;
int32_t g_next = 1;

kg_obs_t ObsOf(Record* r) {
	kg_obs_t obs;
	obs.tiles = r->tiles.data();
	obs.lifespan = r->lifespan.data();
	obs.pos = r->pos.data();
	obs.inv = r->inv.data();
	obs.shed = r->shed.data();
	obs.seeds = r->seeds.data();
	obs.money = r->money.data();
	obs.hands = r->hands.data();
	obs.hires = r->hires.data();
	obs.quadrants = r->quadrants.data();
	obs.market = r->market.data();
	obs.prices = r->prices.data();
	obs.shops = r->shops.data();
	obs.day = r->day.data();
	obs.hour = r->hour.data();
	obs.reward = r->reward.data();
	obs.terminated = r->terminated.data();
	return obs;
}

Record* MakeRecord(int32_t n_envs, int32_t threads, uint64_t seed, int32_t episode_steps) {
	Record* r = new Record();
	r->n_envs = n_envs;
	const size_t n = (size_t)n_envs, p = KG_PLAYERS, u = KG_OBS_UNITS;
	r->tiles.assign(n * p * KG_TILES * KG_OBS_TILE_FIELDS, 0);
	r->lifespan.assign(n * p * KG_TILES, 0);
	r->pos.assign(n * p * u * 2, 0);
	r->inv.assign(n * p * u * KG_NUM_ITEMS, 0);
	r->shed.assign(n * p * KG_NUM_ITEMS, 0);
	r->seeds.assign(n * p * KG_NUM_CROPS, 0);
	r->money.assign(n * p, 0.f);
	r->hands.assign(n * p, 0);
	r->hires.assign(n * p, 0);
	r->quadrants.assign(n * p, 0);
	r->market.assign(n * KG_NUM_PRODUCTS, 0);
	r->prices.assign(n * KG_NUM_PRODUCTS, 0);
	r->shops.assign(n * KG_NUM_SHOPS, 0);
	r->day.assign(n, 0);
	r->hour.assign(n, 0);
	r->reward.assign(n * p, 0.f);
	r->terminated.assign(n * p, 0);
	r->heads.assign(n * p * KG_NUM_HEADS, 0);
	const CudaRuntime* cuda = GetCudaRuntime();
	if (cuda && cuda->host_register) {
		int failures = 0, total = 0;
		int last = 0;
		auto pin = [&](void* ptr, size_t bytes) {
			total++;
			int rc = cuda->host_register(ptr, bytes, 0);
			if (rc != 0) { failures++; last = rc; } else { r->pinned.push_back(ptr); }
		};
		pin(r->tiles.data(), bytes_of(r->tiles));
		pin(r->lifespan.data(), bytes_of(r->lifespan));
		pin(r->pos.data(), bytes_of(r->pos));
		pin(r->inv.data(), bytes_of(r->inv));
		pin(r->shed.data(), bytes_of(r->shed));
		pin(r->seeds.data(), bytes_of(r->seeds));
		pin(r->money.data(), bytes_of(r->money));
		pin(r->market.data(), bytes_of(r->market));
		pin(r->heads.data(), bytes_of(r->heads));
		if (failures)
			std::fprintf(stderr, "kaggriculture: pinned %d/%d buffers, last error %d (%s)\n",
						 total - failures, total, last, cuda->error_string(last));
		else
			(void)0;
	}
	r->vec = kg_vec_create(n_envs, seed, threads, episode_steps);
	kg_vec_bind(r->vec, ObsOf(r), r->heads.data());
	kg_vec_reset(r->vec, seed);
	return r;
}

ffi::Error FindRecord(int32_t handle, Record** out) {
	std::lock_guard<std::mutex> lock(g_mutex);
	auto it = g_records.find(handle);
	if (it == g_records.end())
		return ffi::Error(ffi::ErrorCode::kNotFound, "kaggriculture: bad handle");
	*out = it->second;
	return ffi::Error::Success();
}

template <typename T>
void CopyOut(ffi::Result<ffi::AnyBuffer>& dst, const std::vector<T>& src) {
	std::memcpy(dst->untyped_data(), src.data(), src.size() * sizeof(T));
}

#define KG_COPY_ALL(r)                                                                   \
	CopyOut(tiles, r->tiles);        CopyOut(lifespan, r->lifespan);                     \
	CopyOut(pos, r->pos);            CopyOut(inv, r->inv);                               \
	CopyOut(shed, r->shed);          CopyOut(seeds, r->seeds);                           \
	CopyOut(money, r->money);        CopyOut(hands, r->hands);                           \
	CopyOut(hires, r->hires);        CopyOut(quadrants, r->quadrants);                   \
	CopyOut(market, r->market);      CopyOut(prices, r->prices);                         \
	CopyOut(shops, r->shops);        CopyOut(day, r->day);                               \
	CopyOut(hour, r->hour);          CopyOut(reward, r->reward);                         \
	CopyOut(terminated, r->terminated);

#define KG_RET_LIST                                                                      \
	ffi::Result<ffi::AnyBuffer> handleOut, ffi::Result<ffi::AnyBuffer> tiles,            \
		ffi::Result<ffi::AnyBuffer> lifespan, ffi::Result<ffi::AnyBuffer> pos,           \
		ffi::Result<ffi::AnyBuffer> inv, ffi::Result<ffi::AnyBuffer> shed,               \
		ffi::Result<ffi::AnyBuffer> seeds, ffi::Result<ffi::AnyBuffer> money,            \
		ffi::Result<ffi::AnyBuffer> hands, ffi::Result<ffi::AnyBuffer> hires,            \
		ffi::Result<ffi::AnyBuffer> quadrants, ffi::Result<ffi::AnyBuffer> market,       \
		ffi::Result<ffi::AnyBuffer> prices, ffi::Result<ffi::AnyBuffer> shops,           \
		ffi::Result<ffi::AnyBuffer> day, ffi::Result<ffi::AnyBuffer> hour,               \
		ffi::Result<ffi::AnyBuffer> reward, ffi::Result<ffi::AnyBuffer> terminated

#define KG_BIND_RETS                                                                     \
	.Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()                   \
		.Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()               \
		.Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()               \
		.Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()               \
		.Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()               \
		.Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()

ffi::Error InitCpu(int64_t num_envs, int64_t threads, int64_t seed, int64_t episode_steps, KG_RET_LIST) {
	Record* r = MakeRecord((int32_t)num_envs, (int32_t)threads, (uint64_t)seed, (int32_t)episode_steps);
	int32_t handle;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		handle = g_next++;
		g_records[handle] = r;
	}
	*(int32_t*)handleOut->untyped_data() = handle;
	KG_COPY_ALL(r);
	return ffi::Error::Success();
}

ffi::Error StepCpu(ffi::AnyBuffer handleBuf, ffi::AnyBuffer heads, KG_RET_LIST) {
	int32_t handle = *(const int32_t*)handleBuf.untyped_data();
	Record* r = nullptr;
	if (auto err = FindRecord(handle, &r); err.failure()) return err;
	std::memcpy(r->heads.data(), heads.untyped_data(), bytes_of(r->heads));
	kg_vec_step(r->vec);
	*(int32_t*)handleOut->untyped_data() = handle;
	KG_COPY_ALL(r);
	return ffi::Error::Success();
}

ffi::Error CloseCpu(ffi::AnyBuffer handleBuf, ffi::Result<ffi::AnyBuffer> handleOut) {
	int32_t handle = *(const int32_t*)handleBuf.untyped_data();
	std::lock_guard<std::mutex> lock(g_mutex);
	auto it = g_records.find(handle);
	if (it != g_records.end()) {
		const CudaRuntime* rt = GetCudaRuntime();
		if (rt && rt->host_unregister)
			for (void* ptr : it->second->pinned) rt->host_unregister(ptr);
		kg_vec_free(it->second->vec);
		delete it->second;
		g_records.erase(it);
	}
	*(int32_t*)handleOut->untyped_data() = handle;
	return ffi::Error::Success();
}


ffi::Error MissingCuda() {
	return ffi::Error(ffi::ErrorCode::kUnavailable, "kaggriculture: libcudart not found");
}

ffi::Error CudaCopy(const CudaRuntime* cuda, GpuStream stream, void* dst, const void* src,
					size_t bytes, int kind) {
	if (bytes == 0) return ffi::Error::Success();
	int rc = cuda->memcpy_async(dst, src, bytes, kind, stream);
	if (rc != 0)
		return ffi::Error(ffi::ErrorCode::kInternal, cuda->error_string(rc));
	return ffi::Error::Success();
}

ffi::Error CudaSync(const CudaRuntime* cuda, GpuStream stream) {
	int rc = cuda->stream_synchronize(stream);
	if (rc != 0)
		return ffi::Error(ffi::ErrorCode::kInternal, cuda->error_string(rc));
	return ffi::Error::Success();
}

struct Out { void* dst; const void* src; size_t bytes; };

ffi::Error PushOutputs(const CudaRuntime* cuda, GpuStream stream, int32_t* id, Record* r,
					   ffi::Result<ffi::AnyBuffer>** rets) {
	Out outs[] = {
		{(*rets[0])->untyped_data(), id, sizeof(int32_t)},
		{(*rets[1])->untyped_data(), r->tiles.data(), bytes_of(r->tiles)},
		{(*rets[2])->untyped_data(), r->lifespan.data(), bytes_of(r->lifespan)},
		{(*rets[3])->untyped_data(), r->pos.data(), bytes_of(r->pos)},
		{(*rets[4])->untyped_data(), r->inv.data(), bytes_of(r->inv)},
		{(*rets[5])->untyped_data(), r->shed.data(), bytes_of(r->shed)},
		{(*rets[6])->untyped_data(), r->seeds.data(), bytes_of(r->seeds)},
		{(*rets[7])->untyped_data(), r->money.data(), bytes_of(r->money)},
		{(*rets[8])->untyped_data(), r->hands.data(), bytes_of(r->hands)},
		{(*rets[9])->untyped_data(), r->hires.data(), bytes_of(r->hires)},
		{(*rets[10])->untyped_data(), r->quadrants.data(), bytes_of(r->quadrants)},
		{(*rets[11])->untyped_data(), r->market.data(), bytes_of(r->market)},
		{(*rets[12])->untyped_data(), r->prices.data(), bytes_of(r->prices)},
		{(*rets[13])->untyped_data(), r->shops.data(), bytes_of(r->shops)},
		{(*rets[14])->untyped_data(), r->day.data(), bytes_of(r->day)},
		{(*rets[15])->untyped_data(), r->hour.data(), bytes_of(r->hour)},
		{(*rets[16])->untyped_data(), r->reward.data(), bytes_of(r->reward)},
		{(*rets[17])->untyped_data(), r->terminated.data(), bytes_of(r->terminated)},
	};
	for (const Out& o : outs) {
		if (auto e = CudaCopy(cuda, stream, o.dst, o.src, o.bytes, kCudaMemcpyHostToDevice);
			e.failure())
			return e;
	}
	return ffi::Error::Success();
}

ffi::Error InitCuda(GpuStream stream, int64_t num_envs, int64_t threads, int64_t seed,
					int64_t episode_steps, KG_RET_LIST) {
	const CudaRuntime* cuda = GetCudaRuntime();
	if (cuda == nullptr) return MissingCuda();
	Record* r = MakeRecord((int32_t)num_envs, (int32_t)threads, (uint64_t)seed, (int32_t)episode_steps);
	int32_t handle;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		handle = g_next++;
		g_records[handle] = r;
	}
	ffi::Result<ffi::AnyBuffer>* rets[] = {&handleOut, &tiles, &lifespan, &pos, &inv, &shed,
		&seeds, &money, &hands, &hires, &quadrants, &market, &prices, &shops, &day, &hour,
		&reward, &terminated};
	return PushOutputs(cuda, stream, &handle, r, rets);
}

ffi::Error StepCuda(GpuStream stream, ffi::AnyBuffer handleBuf, ffi::AnyBuffer heads,
					KG_RET_LIST) {
	const CudaRuntime* cuda = GetCudaRuntime();
	if (cuda == nullptr) return MissingCuda();
	int32_t handle = 0;
	if (auto e = CudaCopy(cuda, stream, &handle, handleBuf.untyped_data(), sizeof(int32_t),
						  kCudaMemcpyDeviceToHost); e.failure())
		return e;
	if (auto e = CudaSync(cuda, stream); e.failure()) return e;
	Record* r = nullptr;
	if (auto err = FindRecord(handle, &r); err.failure()) return err;
	if (auto e = CudaCopy(cuda, stream, r->heads.data(), heads.untyped_data(),
						  bytes_of(r->heads), kCudaMemcpyDeviceToHost);
		e.failure())
		return e;
	if (auto e = CudaSync(cuda, stream); e.failure()) return e;
	kg_vec_step(r->vec);
	ffi::Result<ffi::AnyBuffer>* rets[] = {&handleOut, &tiles, &lifespan, &pos, &inv, &shed,
		&seeds, &money, &hands, &hires, &quadrants, &market, &prices, &shops, &day, &hour,
		&reward, &terminated};
	return PushOutputs(cuda, stream, &handle, r, rets);
}

ffi::Error CloseCuda(GpuStream stream, ffi::AnyBuffer handleBuf,
					 ffi::Result<ffi::AnyBuffer> handleOut) {
	const CudaRuntime* cuda = GetCudaRuntime();
	if (cuda == nullptr) return MissingCuda();
	int32_t handle = 0;
	if (auto e = CudaCopy(cuda, stream, &handle, handleBuf.untyped_data(), sizeof(int32_t),
						  kCudaMemcpyDeviceToHost); e.failure())
		return e;
	if (auto e = CudaSync(cuda, stream); e.failure()) return e;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		auto it = g_records.find(handle);
		if (it != g_records.end()) {
			const CudaRuntime* rt = GetCudaRuntime();
			if (rt && rt->host_unregister)
				for (void* ptr : it->second->pinned) rt->host_unregister(ptr);
			kg_vec_free(it->second->vec);
			delete it->second;
			g_records.erase(it);
		}
	}
	if (auto e = CudaCopy(cuda, stream, handleOut->untyped_data(), &handle, sizeof(int32_t),
						  kCudaMemcpyHostToDevice); e.failure())
		return e;
	return CudaSync(cuda, stream);
}

}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kKgInitCpu, InitCpu,
							  ffi::Ffi::Bind()
								  .Attr<int64_t>("num_envs")
								  .Attr<int64_t>("threads")
								  .Attr<int64_t>("seed")
								  .Attr<int64_t>("episode_steps")
									  KG_BIND_RETS);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kKgStepCpu, StepCpu,
							  ffi::Ffi::Bind()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
									  KG_BIND_RETS);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kKgCloseCpu, CloseCpu,
							  ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kKgInitCuda, InitCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("num_envs")
								  .Attr<int64_t>("threads")
								  .Attr<int64_t>("seed")
								  .Attr<int64_t>("episode_steps")
									  KG_BIND_RETS);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kKgStepCuda, StepCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
									  KG_BIND_RETS);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kKgCloseCuda, CloseCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

extern "C" {
void* kaggriculture_ffi_init_cuda() { return (void*)kKgInitCuda; }
void* kaggriculture_ffi_step_cuda() { return (void*)kKgStepCuda; }
void* kaggriculture_ffi_close_cuda() { return (void*)kKgCloseCuda; }
void* kaggriculture_ffi_init_cpu() { return (void*)kKgInitCpu; }
void* kaggriculture_ffi_step_cpu() { return (void*)kKgStepCpu; }
void* kaggriculture_ffi_close_cpu() { return (void*)kKgCloseCpu; }
}
