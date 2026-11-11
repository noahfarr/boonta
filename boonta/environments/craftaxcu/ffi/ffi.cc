#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" {
int32_t craftaxcu_observe(void* handle, void* stream, float* obs, float* rewards,
						  float* terminals);
int32_t craftaxcu_step(void* handle, void* stream, const int32_t* actions, float* obs,
					   float* rewards, float* terminals);
int32_t craftaxcu_reset(void* handle, void* stream, float* obs, float* rewards,
						float* terminals);
int32_t craftaxcu_reset_full(void* handle, void* stream, uint8_t* states,
							 uint8_t* soa, uint8_t* tail, float* obs, float* rewards,
							 float* terminals);
int32_t craftaxcu_step_full(void* handle, void* stream, uint8_t* states, uint8_t* soa,
							uint8_t* tail, const int32_t* actions, float* obs,
							float* rewards, float* terminals, float* tally);
int32_t craftaxcu_observe_full(void* handle, void* stream, uint8_t* states, uint8_t* soa,
							   uint8_t* tail, float* obs, float* rewards,
							   float* terminals);
int32_t craftaxcu_bank(void* handle, void* stream, uint8_t* states, uint8_t* soa,
					   uint8_t* tail, const int32_t* slots);
int32_t craftaxcu_revive(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						 uint8_t* tail, const int32_t* slots);
int32_t craftaxcu_save(void* handle, void* stream, const int32_t* index, int32_t count,
					   uint8_t* out);
int32_t craftaxcu_restore(void* handle, void* stream, const int32_t* index, int32_t count,
						  const uint8_t* in);
int32_t craftaxcu_cells(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						uint32_t* out);
int32_t craftaxcu_portray(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						  int32_t* out);
int32_t craftaxcu_worlds(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						 uint32_t* out);
int32_t craftaxcu_laurels(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						  float* out);
int32_t craftaxcu_action_mask(void* handle, void* stream, uint8_t* states, uint8_t* soa,
							  uint8_t* out);
int32_t craftaxcu_groups(void* handle, void* stream, int32_t* out);
int32_t craftaxcu_owners(void* handle, void* stream, int32_t* out);
int32_t craftaxcu_archive(void* handle, int32_t capacity, int32_t reservoir);
int32_t craftaxcu_step_archive(void* handle, void* stream, uint8_t* states,
							   uint8_t* soa, uint8_t* tail_leaf,
							   const int32_t* actions,
							   const float* weights, const float* cdf,
							   const float* share, const float* floor,
							   const float* leash, const float* refresh, float* obs,
							   float* rewards, float* terminals, int32_t* cells,
							   int32_t* origin, int32_t* banked, float* laurels,
							   int32_t* place, int32_t* store, const float* recent);
}

namespace {

using GpuStream = void*;

ffi::Error fault(int32_t code) {
	if (code == 0) return ffi::Error::Success();
	return ffi::Error(ffi::ErrorCode::kInternal, "craftax.cu: kernel launch failed");
}

ffi::Error ObserveCuda(GpuStream stream, int64_t pool, ffi::Result<ffi::AnyBuffer> obs,
					   ffi::Result<ffi::AnyBuffer> rewards,
					   ffi::Result<ffi::AnyBuffer> terminals) {
	return fault(craftaxcu_observe(reinterpret_cast<void*>(pool), stream,
								   reinterpret_cast<float*>(obs->untyped_data()),
								   reinterpret_cast<float*>(rewards->untyped_data()),
								   reinterpret_cast<float*>(terminals->untyped_data())));
}

ffi::Error ResetCuda(GpuStream stream, int64_t pool, ffi::Result<ffi::AnyBuffer> obs,
					 ffi::Result<ffi::AnyBuffer> rewards,
					 ffi::Result<ffi::AnyBuffer> terminals) {
	return fault(craftaxcu_reset(reinterpret_cast<void*>(pool), stream,
								 reinterpret_cast<float*>(obs->untyped_data()),
								 reinterpret_cast<float*>(rewards->untyped_data()),
								 reinterpret_cast<float*>(terminals->untyped_data())));
}

ffi::Error StepCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer actions,
					ffi::Result<ffi::AnyBuffer> obs, ffi::Result<ffi::AnyBuffer> rewards,
					ffi::Result<ffi::AnyBuffer> terminals) {
	return fault(craftaxcu_step(reinterpret_cast<void*>(pool), stream,
								reinterpret_cast<const int32_t*>(actions.untyped_data()),
								reinterpret_cast<float*>(obs->untyped_data()),
								reinterpret_cast<float*>(rewards->untyped_data()),
								reinterpret_cast<float*>(terminals->untyped_data())));
}

ffi::Error ObserveFullCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
						   ffi::AnyBuffer soa, ffi::AnyBuffer tail,
						   ffi::Result<ffi::AnyBuffer> obs,
						   ffi::Result<ffi::AnyBuffer> rewards,
						   ffi::Result<ffi::AnyBuffer> terminals) {
	return fault(craftaxcu_observe_full(
		reinterpret_cast<void*>(pool), stream,
		reinterpret_cast<uint8_t*>(states.untyped_data()),
		reinterpret_cast<uint8_t*>(soa.untyped_data()),
		reinterpret_cast<uint8_t*>(tail.untyped_data()),
		reinterpret_cast<float*>(obs->untyped_data()),
		reinterpret_cast<float*>(rewards->untyped_data()),
		reinterpret_cast<float*>(terminals->untyped_data())));
}

ffi::Error ResetFullCuda(GpuStream stream, int64_t pool,
						 ffi::Result<ffi::AnyBuffer> states,
						 ffi::Result<ffi::AnyBuffer> soa,
						 ffi::Result<ffi::AnyBuffer> tail,
						 ffi::Result<ffi::AnyBuffer> obs,
						 ffi::Result<ffi::AnyBuffer> rewards,
						 ffi::Result<ffi::AnyBuffer> terminals) {
	return fault(craftaxcu_reset_full(reinterpret_cast<void*>(pool), stream,
									  reinterpret_cast<uint8_t*>(states->untyped_data()),
									  reinterpret_cast<uint8_t*>(soa->untyped_data()),
									  reinterpret_cast<uint8_t*>(tail->untyped_data()),
									  reinterpret_cast<float*>(obs->untyped_data()),
									  reinterpret_cast<float*>(rewards->untyped_data()),
									  reinterpret_cast<float*>(terminals->untyped_data())));
}

// The three state leaves are aliased to the first three results, so XLA hands
// the same device buffer for input and output and stepping copies nothing.
ffi::Error StepFullCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
						ffi::AnyBuffer soa, ffi::AnyBuffer tail,
						ffi::AnyBuffer actions, ffi::Result<ffi::AnyBuffer> states_out,
						ffi::Result<ffi::AnyBuffer> soa_out,
						ffi::Result<ffi::AnyBuffer> tail_out,
						ffi::Result<ffi::AnyBuffer> obs,
						ffi::Result<ffi::AnyBuffer> rewards,
						ffi::Result<ffi::AnyBuffer> terminals,
						ffi::Result<ffi::AnyBuffer> tally) {
	(void)states;
	(void)soa;
	(void)tail;
	return fault(craftaxcu_step_full(reinterpret_cast<void*>(pool), stream,
									 reinterpret_cast<uint8_t*>(states_out->untyped_data()),
									 reinterpret_cast<uint8_t*>(soa_out->untyped_data()),
									 reinterpret_cast<uint8_t*>(tail_out->untyped_data()),
									 reinterpret_cast<const int32_t*>(actions.untyped_data()),
									 reinterpret_cast<float*>(obs->untyped_data()),
									 reinterpret_cast<float*>(rewards->untyped_data()),
									 reinterpret_cast<float*>(terminals->untyped_data()),
									 reinterpret_cast<float*>(tally->untyped_data())));
}

// The three state leaves are aliased to the three results, so banking and
// reviving mutate them in place and no env state crosses into XLA buffers.
ffi::Error BankCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
					ffi::AnyBuffer soa, ffi::AnyBuffer tail, ffi::AnyBuffer slots,
					ffi::Result<ffi::AnyBuffer> states_out,
					ffi::Result<ffi::AnyBuffer> soa_out,
					ffi::Result<ffi::AnyBuffer> tail_out) {
	(void)states;
	(void)soa;
	(void)tail;
	return fault(craftaxcu_bank(
		reinterpret_cast<void*>(pool), stream,
		reinterpret_cast<uint8_t*>(states_out->untyped_data()),
		reinterpret_cast<uint8_t*>(soa_out->untyped_data()),
		reinterpret_cast<uint8_t*>(tail_out->untyped_data()),
		reinterpret_cast<const int32_t*>(slots.untyped_data())));
}

ffi::Error ReviveCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
					  ffi::AnyBuffer soa, ffi::AnyBuffer tail, ffi::AnyBuffer slots,
					  ffi::Result<ffi::AnyBuffer> states_out,
					  ffi::Result<ffi::AnyBuffer> soa_out,
					  ffi::Result<ffi::AnyBuffer> tail_out) {
	(void)states;
	(void)soa;
	(void)tail;
	return fault(craftaxcu_revive(
		reinterpret_cast<void*>(pool), stream,
		reinterpret_cast<uint8_t*>(states_out->untyped_data()),
		reinterpret_cast<uint8_t*>(soa_out->untyped_data()),
		reinterpret_cast<uint8_t*>(tail_out->untyped_data()),
		reinterpret_cast<const int32_t*>(slots.untyped_data())));
}

ffi::Error SaveCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer index,
					ffi::Result<ffi::AnyBuffer> blob) {
	int32_t count = static_cast<int32_t>(index.element_count());
	return fault(craftaxcu_save(reinterpret_cast<void*>(pool), stream,
								reinterpret_cast<const int32_t*>(index.untyped_data()),
								count,
								reinterpret_cast<uint8_t*>(blob->untyped_data())));
}

ffi::Error RestoreCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer index,
					   ffi::AnyBuffer blob, ffi::Result<ffi::AnyBuffer> token) {
	int32_t count = static_cast<int32_t>(index.element_count());
	return fault(craftaxcu_restore(reinterpret_cast<void*>(pool), stream,
								   reinterpret_cast<const int32_t*>(index.untyped_data()),
								   count,
								   reinterpret_cast<const uint8_t*>(blob.untyped_data())));
}

ffi::Error ActionMaskCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
						  ffi::AnyBuffer soa, ffi::Result<ffi::AnyBuffer> mask) {
	return fault(craftaxcu_action_mask(reinterpret_cast<void*>(pool), stream,
									   reinterpret_cast<uint8_t*>(states.untyped_data()),
									   reinterpret_cast<uint8_t*>(soa.untyped_data()),
									   reinterpret_cast<uint8_t*>(mask->untyped_data())));
}

ffi::Error PortrayCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
					   ffi::AnyBuffer soa, ffi::Result<ffi::AnyBuffer> sheet) {
	return fault(craftaxcu_portray(reinterpret_cast<void*>(pool), stream,
								   reinterpret_cast<uint8_t*>(states.untyped_data()),
								   reinterpret_cast<uint8_t*>(soa.untyped_data()),
								   reinterpret_cast<int32_t*>(sheet->untyped_data())));
}

ffi::Error CellsCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
					 ffi::AnyBuffer soa, ffi::Result<ffi::AnyBuffer> tags) {
	return fault(craftaxcu_cells(reinterpret_cast<void*>(pool), stream,
								 reinterpret_cast<uint8_t*>(states.untyped_data()),
								 reinterpret_cast<uint8_t*>(soa.untyped_data()),
								 reinterpret_cast<uint32_t*>(tags->untyped_data())));
}

ffi::Error WorldsCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
					  ffi::AnyBuffer soa, ffi::Result<ffi::AnyBuffer> worlds) {
	return fault(craftaxcu_worlds(reinterpret_cast<void*>(pool), stream,
								  reinterpret_cast<uint8_t*>(states.untyped_data()),
								  reinterpret_cast<uint8_t*>(soa.untyped_data()),
								  reinterpret_cast<uint32_t*>(worlds->untyped_data())));
}

ffi::Error LaurelsCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
					   ffi::AnyBuffer soa, ffi::Result<ffi::AnyBuffer> laurels) {
	return fault(craftaxcu_laurels(reinterpret_cast<void*>(pool), stream,
								   reinterpret_cast<uint8_t*>(states.untyped_data()),
								   reinterpret_cast<uint8_t*>(soa.untyped_data()),
								   reinterpret_cast<float*>(laurels->untyped_data())));
}

ffi::Error GroupsCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer token,
					  ffi::Result<ffi::AnyBuffer> groups) {
	return fault(craftaxcu_groups(reinterpret_cast<void*>(pool), stream,
								  reinterpret_cast<int32_t*>(groups->untyped_data())));
}

ffi::Error OwnersCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer token,
					  ffi::Result<ffi::AnyBuffer> owners) {
	return fault(craftaxcu_owners(reinterpret_cast<void*>(pool), stream,
								  reinterpret_cast<int32_t*>(owners->untyped_data())));
}

ffi::Error StepArchiveCuda(GpuStream stream, int64_t pool, ffi::AnyBuffer states,
						   ffi::AnyBuffer soa, ffi::AnyBuffer tail,
						   ffi::AnyBuffer actions,
						   ffi::AnyBuffer weights, ffi::AnyBuffer cdf,
						   ffi::AnyBuffer share, ffi::AnyBuffer floor, ffi::AnyBuffer leash,
						   ffi::AnyBuffer refresh, ffi::AnyBuffer recent,
						   ffi::Result<ffi::AnyBuffer> states_out,
						   ffi::Result<ffi::AnyBuffer> soa_out,
						   ffi::Result<ffi::AnyBuffer> tail_out,
						   ffi::Result<ffi::AnyBuffer> obs,
						   ffi::Result<ffi::AnyBuffer> rewards,
						   ffi::Result<ffi::AnyBuffer> terminals,
						   ffi::Result<ffi::AnyBuffer> cells,
						   ffi::Result<ffi::AnyBuffer> origin,
						   ffi::Result<ffi::AnyBuffer> banked,
						   ffi::Result<ffi::AnyBuffer> laurels,
						   ffi::Result<ffi::AnyBuffer> place,
						   ffi::Result<ffi::AnyBuffer> store) {
	(void)states;
	(void)soa;
	(void)tail;
	return fault(craftaxcu_step_archive(
		reinterpret_cast<void*>(pool), stream,
		reinterpret_cast<uint8_t*>(states_out->untyped_data()),
		reinterpret_cast<uint8_t*>(soa_out->untyped_data()),
		reinterpret_cast<uint8_t*>(tail_out->untyped_data()),
		reinterpret_cast<const int32_t*>(actions.untyped_data()),
		reinterpret_cast<const float*>(weights.untyped_data()),
		reinterpret_cast<const float*>(cdf.untyped_data()),
		reinterpret_cast<const float*>(share.untyped_data()),
		reinterpret_cast<const float*>(floor.untyped_data()),
		reinterpret_cast<const float*>(leash.untyped_data()),
		reinterpret_cast<const float*>(refresh.untyped_data()),
		reinterpret_cast<float*>(obs->untyped_data()),
		reinterpret_cast<float*>(rewards->untyped_data()),
		reinterpret_cast<float*>(terminals->untyped_data()),
		reinterpret_cast<int32_t*>(cells->untyped_data()),
		reinterpret_cast<int32_t*>(origin->untyped_data()),
		reinterpret_cast<int32_t*>(banked->untyped_data()),
		reinterpret_cast<float*>(laurels->untyped_data()),
		reinterpret_cast<int32_t*>(place->untyped_data()),
		reinterpret_cast<int32_t*>(store->untyped_data()),
		reinterpret_cast<const float*>(recent.untyped_data())));
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxObserveCuda, ObserveCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxResetCuda, ResetCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxStepCuda, StepCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxObserveFullCuda, ObserveFullCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxResetFullCuda, ResetFullCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxStepFullCuda, StepFullCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxBankCuda, BankCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxReviveCuda, ReviveCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxSaveCuda, SaveCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxRestoreCuda, RestoreCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxGroupsCuda, GroupsCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxOwnersCuda, OwnersCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxActionMaskCuda, ActionMaskCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxPortrayCuda, PortrayCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxCellsCuda, CellsCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxWorldsCuda, WorldsCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxLaurelsCuda, LaurelsCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCraftaxStepArchiveCuda, StepArchiveCuda,
							  ffi::Ffi::Bind()
								  .Ctx<ffi::PlatformStream<GpuStream>>()
								  .Attr<int64_t>("pool")
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Arg<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>()
								  .Ret<ffi::AnyBuffer>());

extern "C" {
void* craftaxcu_ffi_step_archive_cuda() {
	return reinterpret_cast<void*>(kCraftaxStepArchiveCuda);
}
void* craftaxcu_ffi_cells_cuda() { return reinterpret_cast<void*>(kCraftaxCellsCuda); }
void* craftaxcu_ffi_portray_cuda() { return reinterpret_cast<void*>(kCraftaxPortrayCuda); }
void* craftaxcu_ffi_worlds_cuda() { return reinterpret_cast<void*>(kCraftaxWorldsCuda); }
void* craftaxcu_ffi_laurels_cuda() {
	return reinterpret_cast<void*>(kCraftaxLaurelsCuda);
}
void* craftaxcu_ffi_action_mask_cuda() {
	return reinterpret_cast<void*>(kCraftaxActionMaskCuda);
}
void* craftaxcu_ffi_groups_cuda() { return reinterpret_cast<void*>(kCraftaxGroupsCuda); }
void* craftaxcu_ffi_owners_cuda() { return reinterpret_cast<void*>(kCraftaxOwnersCuda); }
void* craftaxcu_ffi_save_cuda() { return reinterpret_cast<void*>(kCraftaxSaveCuda); }
void* craftaxcu_ffi_restore_cuda() { return reinterpret_cast<void*>(kCraftaxRestoreCuda); }
void* craftaxcu_ffi_observe_cuda() { return reinterpret_cast<void*>(kCraftaxObserveCuda); }
void* craftaxcu_ffi_step_cuda() { return reinterpret_cast<void*>(kCraftaxStepCuda); }
void* craftaxcu_ffi_bank_cuda() { return reinterpret_cast<void*>(kCraftaxBankCuda); }
void* craftaxcu_ffi_revive_cuda() { return reinterpret_cast<void*>(kCraftaxReviveCuda); }
void* craftaxcu_ffi_observe_full_cuda() {
	return reinterpret_cast<void*>(kCraftaxObserveFullCuda);
}
void* craftaxcu_ffi_reset_full_cuda() {
	return reinterpret_cast<void*>(kCraftaxResetFullCuda);
}
void* craftaxcu_ffi_step_full_cuda() {
	return reinterpret_cast<void*>(kCraftaxStepFullCuda);
}
void* craftaxcu_ffi_reset_cuda() { return reinterpret_cast<void*>(kCraftaxResetCuda); }
}
