#define CRAFTAX_CU_LIB
#include "vendor/craftax.cu"

static inline void cf_bind(CuVec* v, cudaStream_t st) { cu_vec_bind(v, st); }

#if CF_ENV_MAJOR
#define CF_AT(f, c, i) (g_cf_pool.f[(size_t)(i) * cf_row_##f + (size_t)(c)])
#else
#define CF_AT(f, c, i) (g_cf_pool.f[(size_t)(c) * (size_t)g_cf_n + (size_t)(i)])
#endif

namespace {

__global__ void k_bind(Craftax* envs, int n, CraftaxObs* obs, float* rewards,
                       float* terminals) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	envs[i].observations = obs + (size_t)i * CRAFTAX_OBS_SIZE;
	envs[i].rewards = rewards + i;
	envs[i].terminals = terminals + i;
}

__global__ void k_cast(const int32_t* src, float* dst, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	dst[i] = (float)src[i];
}

__device__ Craftax* g_cf_envs_for_snapshot = nullptr;

__global__ void k_set_envs(Craftax* envs) { g_cf_envs_for_snapshot = envs; }

__global__ void k_reset_all(Craftax* envs, int n, CraftaxResetRec* resets) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	Craftax* e = &envs[i];
	CraftaxThreefryKey carry, reset_key;
	craftax_threefry_split(e->rng_key, &carry, &reset_key);
	e->rng_key = carry;
	resets[i].env = i;
	resets[i].key0 = reset_key.word[0];
	resets[i].key1 = reset_key.word[1];
	if (i == 0) g_reset_count = n;
}

#define CF_BYTES(f, T, C) +(int)((sizeof(T) * (C) + 7) / 8 * 8)
static constexpr int kSoaBytes = 0 CF_SOA_FIELDS(CF_BYTES);
#undef CF_BYTES
static constexpr int kWorldBytes = (int)sizeof(CraftaxState);

struct EnvTail {
	unsigned int rng;
	uint64_t seed;
	CraftaxThreefryKey rng_key;
	float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
	float episode_return_accum;
	int32_t episode_length_accum;
	int32_t curriculum_episode;
};

static constexpr int kTailBytes = (int)sizeof(EnvTail);
static constexpr int kBlobBytes =
	(kWorldBytes + kSoaBytes + kTailBytes + 15) / 16 * 16;

__device__ void blob_save(int i, uint8_t* p) {
	const uint4* src = (const uint4*)&((CraftaxState*)g_cf_state_base)[i];
	uint4* dst = (uint4*)p;
	for (int w = threadIdx.x; w < kWorldBytes / 16; w += blockDim.x) dst[w] = src[w];
	uint8_t* q = p + kWorldBytes;
#define CF_SAVE(f, T, C)                                                     \
	{                                                                        \
		for (int c = threadIdx.x; c < (C); c += blockDim.x)                  \
			((T*)q)[c] = CF_AT(f, c, i); \
		q += (sizeof(T) * (C) + 7) / 8 * 8;                                  \
	}
	CF_SOA_FIELDS(CF_SAVE)
#undef CF_SAVE
	if (threadIdx.x != 0) return;
	const Craftax* e = &g_cf_envs_for_snapshot[i];
	EnvTail* tail = (EnvTail*)q;
	tail->rng = e->rng;
	tail->seed = e->seed;
	tail->rng_key = e->rng_key;
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		tail->achievements[c] = e->achievements[c];
	tail->episode_return_accum = e->episode_return_accum;
	tail->episode_length_accum = e->episode_length_accum;
	tail->curriculum_episode = e->curriculum_episode;
}

__device__ void blob_restore(int i, const uint8_t* p) {
	const uint4* src = (const uint4*)p;
	uint4* dst = (uint4*)&((CraftaxState*)g_cf_state_base)[i];
	for (int w = threadIdx.x; w < kWorldBytes / 16; w += blockDim.x) dst[w] = src[w];
	const uint8_t* q = p + kWorldBytes;
#define CF_LOAD(f, T, C)                                                     \
	{                                                                        \
		for (int c = threadIdx.x; c < (C); c += blockDim.x)                  \
			CF_AT(f, c, i) = ((const T*)q)[c]; \
		q += (sizeof(T) * (C) + 7) / 8 * 8;                                  \
	}
	CF_SOA_FIELDS(CF_LOAD)
#undef CF_LOAD
	if (threadIdx.x != 0) return;
	Craftax* e = &g_cf_envs_for_snapshot[i];
	const EnvTail* tail = (const EnvTail*)q;
	e->rng = tail->rng;
	e->seed = tail->seed;
	e->rng_key = tail->rng_key;
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		e->achievements[c] = tail->achievements[c];
	e->episode_return_accum = tail->episode_return_accum;
	e->episode_length_accum = tail->episode_length_accum;
	e->curriculum_episode = tail->curriculum_episode;
}

__global__ void k_save(int n, const int32_t* index, uint8_t* out) {
	int j = blockIdx.x;
	if (j >= n) return;
	int i = index == nullptr ? j : index[j];
	blob_save(i, out + (size_t)j * kBlobBytes);
}

// Env i banks into archive slot slots[i]; a negative slot means this env is not
// banking. The slot vector is chosen on the JAX side, so no policy lives here.
__global__ void k_bank_slots(int n, const int32_t* slots, uint8_t* archive) {
	int i = blockIdx.x;
	if (i >= n) return;
	int32_t slot = slots[i];
	if (slot < 0) return;
	blob_save(i, archive + (size_t)slot * kBlobBytes);
}

__global__ void k_revive_slots(int n, const int32_t* slots, const uint8_t* archive) {
	int i = blockIdx.x;
	if (i >= n) return;
	int32_t slot = slots[i];
	if (slot < 0) return;
	blob_restore(i, archive + (size_t)slot * kBlobBytes);
}

__global__ void k_restore(int n, const int32_t* index, const uint8_t* in) {
	int j = blockIdx.x;
	if (j >= n) return;
	int i = index == nullptr ? j : index[j];
	if (i < 0) return;
	blob_restore(i, in + (size_t)j * kBlobBytes);
}

#define CF_FNV_SEED 2166136261u
#define CF_FNV_PRIME 16777619u

__device__ inline uint32_t cf_mix(uint32_t h, uint32_t x) {
	return (h ^ x) * CF_FNV_PRIME;
}

__device__ float cf_vitality_state(int i) {
	const CraftaxState* st = &((CraftaxState*)g_cf_state_base)[i];
	const CraftaxWorldState* ws = (const CraftaxWorldState*)(const void*)st;
	(void)st;
	return fminf(
		fminf(CF(player_health, ws) * 0.1f, (float)CF(player_food, ws) * 0.1f),
		fminf((float)CF(player_drink, ws) * 0.1f, (float)CF(player_energy, ws) * 0.1f));
}

__device__ inline uint32_t cf_ach_tier(int c) {
	if (c < 25) return 0u;
	if (c < 30) return 1u;
	if (c < 33) return 2u;
	if (c < 36) return 3u;
	if (c < 40) return 1u;
	if (c < 44) return 2u;
	if (c < 50) return 3u;
	if (c < 55) return 1u;
	if (c < 59) return 2u;
	if (c < 63) return 1u;
	return 2u;
}

__device__ uint32_t cf_cell_state(int i) {
	const CraftaxState* st = &((CraftaxState*)g_cf_state_base)[i];
	const CraftaxWorldState* ws = (const CraftaxWorldState*)(const void*)st;
	uint32_t h = CF_FNV_SEED;
	uint32_t held = 0, deep = 0;
	uint32_t tier[4] = {0u, 0u, 0u, 0u};
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		if (CF2(achievements, c, st)) {
			if (c >= CRAFTAX_ACH_ENTER_GNOMISH_MINES && c != CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE) deep++; else held++;
			tier[cf_ach_tier(c)]++;
			if (!g_cf_pool.coarse || g_cf_pool.coarse == 7u) h = cf_mix(h, (uint32_t)(c + 1));
		}
	if (g_cf_pool.coarse == 7u) return h;
	if (g_cf_pool.coarse == 1u) h = cf_mix(h, held + deep + 1);
	if (g_cf_pool.coarse == 2u) h = cf_mix(cf_mix(h, held + 1), deep + 201u);
	if (g_cf_pool.coarse == 3u) {
		uint32_t pick = CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE, st) ? 4u
			: CF2(achievements, CRAFTAX_ACH_MAKE_IRON_PICKAXE, st) ? 3u
			: CF2(achievements, CRAFTAX_ACH_MAKE_STONE_PICKAXE, st) ? 2u
			: CF2(achievements, CRAFTAX_ACH_MAKE_WOOD_PICKAXE, st) ? 1u : 0u;
		uint32_t sword = CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_SWORD, st) ? 4u
			: CF2(achievements, CRAFTAX_ACH_MAKE_IRON_SWORD, st) ? 3u
			: CF2(achievements, CRAFTAX_ACH_MAKE_STONE_SWORD, st) ? 2u
			: CF2(achievements, CRAFTAX_ACH_MAKE_WOOD_SWORD, st) ? 1u : 0u;
		uint32_t bow = CF2(achievements, CRAFTAX_ACH_FIND_BOW, st) ? 1u : 0u;
		uint32_t armour = CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR, st) ? 2u
			: CF2(achievements, CRAFTAX_ACH_MAKE_IRON_ARMOUR, st) ? 1u : 0u;
		h = cf_mix(h, held + deep + 1);
		h = cf_mix(h, 301u + pick + 5u * sword + 25u * bow + 50u * armour);
	}
	if (g_cf_pool.coarse == 4u) {
		h = cf_mix(h, 401u + tier[0]);
		h = cf_mix(h, 501u + tier[1]);
		h = cf_mix(h, 601u + tier[2]);
		h = cf_mix(h, 701u + tier[3]);
	}
	if (g_cf_pool.coarse == 5u || g_cf_pool.coarse == 6u) {
		for (int t = 0; t < 4; t++) h = cf_mix(h, 401u + 100u * (uint32_t)t + tier[t]);
		if (g_cf_pool.coarse == 5u) return h;
		int here = (int)CF(player_level, ws);
		int felled = (int)CF2(monsters_killed, here < 0 ? 0 : (here > 8 ? 8 : here), st);
		return cf_mix(h, felled >= CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL ? 901u : 900u);
	}
	float vitality = fminf(
		fminf(CF(player_health, ws) * 0.1f, (float)CF(player_food, ws) * 0.1f),
		fminf((float)CF(player_drink, ws) * 0.1f, (float)CF(player_energy, ws) * 0.1f));
	uint32_t band = vitality < 0.3f ? 0u : (vitality < 0.7f ? 1u : 2u);
	int level = (int)CF(player_level, ws);
	int slain = (int)CF2(monsters_killed, level < 0 ? 0 : (level > 8 ? 8 : level), st);
	slain = slain < 0 ? 0 : (slain > CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL ? CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL : slain);
	h = cf_mix(h, (uint32_t)level);
	h = cf_mix(h, (uint32_t)slain + 101u);
	return cf_mix(h, band);
}

static constexpr int kSheetCols = 9;

__device__ void cf_describe(int i, int32_t* row) {
	const CraftaxState* st = &((CraftaxState*)g_cf_state_base)[i];
	const CraftaxWorldState* ws = (const CraftaxWorldState*)(const void*)st;
	uint32_t mask[3] = {0u, 0u, 0u};
	int count = 0;
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		if (CF2(achievements, c, st)) {
			mask[c >> 5] |= 1u << (c & 31);
			count++;
		}
	float vitality = cf_vitality_state(i);
	row[0] = (int32_t)CF(player_level, ws);
	row[1] = vitality < 0.3f ? 0 : (vitality < 0.7f ? 1 : 2);
	row[2] = count;
	row[3] = (int32_t)mask[0];
	row[4] = (int32_t)mask[1];
	row[5] = (int32_t)mask[2];
	row[6] = (int32_t)CF2(player_position, 0, st);
	row[7] = (int32_t)CF2(player_position, 1, st);
	int level = (int)CF(player_level, ws);
	row[8] = (int32_t)CF2(monsters_killed, level < 0 ? 0 : (level > 8 ? 8 : level), st);
}

__global__ void k_action_mask(int n, uint8_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const CraftaxState* st = &((CraftaxState*)g_cf_state_base)[i];
	uint8_t* m = out + (size_t)i * CRAFTAX_NUM_ACTIONS;
	for (int a = 0; a < CRAFTAX_NUM_ACTIONS; a++) m[a] = 1;
	if (CF(is_sleeping, st) || CF(is_resting, st)) return;

	int32_t step[2];
	craftax_step_direction(CF(player_direction, st), step);
	int32_t fr = CF2(player_position, 0, st) + step[0];
	int32_t fc = CF2(player_position, 1, st) + step[1];
	int32_t lvl = craftax_step_jax_index(CF(player_level, st), CRAFTAX_NUM_LEVELS);
	int32_t block = st->map[lvl][craftax_step_jax_index(fr, CRAFTAX_MAP_SIZE)]
						   [craftax_step_jax_index(fc, CRAFTAX_MAP_SIZE)];
	int32_t item = st->item_map[lvl][craftax_step_jax_index(fr, CRAFTAX_MAP_SIZE)]
								   [craftax_step_jax_index(fc, CRAFTAX_MAP_SIZE)];
	int32_t under = st->item_map[lvl]
			[craftax_step_jax_index(CF2(player_position, 0, st), CRAFTAX_MAP_SIZE)]
			[craftax_step_jax_index(CF2(player_position, 1, st), CRAFTAX_MAP_SIZE)];
	bool room = craftax_do_action_in_bounds(fr, fc);
	bool mob = craftax_step_is_in_mob(st, fr, fc);
	bool table = craftax_crafting_is_near_block(st, CRAFTAX_BLOCK_CRAFTING_TABLE);
	bool furnace = craftax_crafting_is_near_block(st, CRAFTAX_BLOCK_FURNACE);
	bool bare = room && !mob && !craftax_step_is_solid_block(block)
				&& item == CRAFTAX_ITEM_NONE;
	int32_t pick = CF(inv_pickaxe, st);
	int32_t wood = CF(inv_wood, st), stone = CF(inv_stone, st);
	int32_t iron = CF(inv_iron, st), coal = CF(inv_coal, st);
	int32_t gem = CF(inv_diamond, st);
	int32_t shots = craftax_medium_projectile_count(st);
	int32_t xp = CF(player_xp, st), mana = CF(player_mana, st);
	bool altar = block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE
					 ? CF(inv_ruby, st) >= 1
					 : (block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE
							? CF(inv_sapphire, st) >= 1
							: false);
	bool spell = mana >= 9 && altar;

	m[CRAFTAX_ACTION_DO] =
		!room || mob || block == CRAFTAX_BLOCK_TREE || block == CRAFTAX_BLOCK_FIRE_TREE
		|| block == CRAFTAX_BLOCK_ICE_SHRUB || block == CRAFTAX_BLOCK_FURNACE
		|| block == CRAFTAX_BLOCK_CRAFTING_TABLE || block == CRAFTAX_BLOCK_RIPE_PLANT
		|| block == CRAFTAX_BLOCK_WATER || block == CRAFTAX_BLOCK_FOUNTAIN
		|| block == CRAFTAX_BLOCK_CHEST || block == CRAFTAX_BLOCK_GRASS
		|| block == CRAFTAX_BLOCK_NECROMANCER
		|| ((block == CRAFTAX_BLOCK_STONE || block == CRAFTAX_BLOCK_COAL
			 || block == CRAFTAX_BLOCK_STALAGMITE) && pick >= 1)
		|| (block == CRAFTAX_BLOCK_IRON && pick >= 2)
		|| (block == CRAFTAX_BLOCK_DIAMOND && pick >= 3)
		|| ((block == CRAFTAX_BLOCK_SAPPHIRE || block == CRAFTAX_BLOCK_RUBY) && pick >= 4);
	m[CRAFTAX_ACTION_SLEEP] = CF(player_energy, st) < craftax_step_get_max_energy(st);
	m[CRAFTAX_ACTION_PLACE_STONE] =
		stone > 0 && room && !mob
		&& (block == CRAFTAX_BLOCK_WATER
			|| (!craftax_step_is_solid_block(block) && item == CRAFTAX_ITEM_NONE));
	m[CRAFTAX_ACTION_PLACE_TABLE] = wood >= 2 && bare;
	m[CRAFTAX_ACTION_PLACE_FURNACE] = stone > 0 && bare;
	m[CRAFTAX_ACTION_PLACE_PLANT] =
		CF(inv_sapling, st) > 0 && room && !mob && block == CRAFTAX_BLOCK_GRASS
		&& item == CRAFTAX_ITEM_NONE;
	m[CRAFTAX_ACTION_PLACE_TORCH] =
		CF(inv_torches, st) > 0 && room && !mob
		&& craftax_crafting_can_place_item(block) && item == CRAFTAX_ITEM_NONE;
	m[CRAFTAX_ACTION_MAKE_WOOD_PICKAXE] = wood >= 1 && table && pick < 1;
	m[CRAFTAX_ACTION_MAKE_STONE_PICKAXE] = wood >= 1 && stone >= 1 && table && pick < 2;
	m[CRAFTAX_ACTION_MAKE_IRON_PICKAXE] =
		wood >= 1 && stone >= 1 && iron >= 1 && coal >= 1 && table && furnace && pick < 3;
	m[CRAFTAX_ACTION_MAKE_DIAMOND_PICKAXE] = wood >= 1 && gem >= 3 && table && pick < 4;
	m[CRAFTAX_ACTION_MAKE_WOOD_SWORD] = wood >= 1 && table && CF(inv_sword, st) < 1;
	m[CRAFTAX_ACTION_MAKE_STONE_SWORD] =
		wood >= 1 && stone >= 1 && table && CF(inv_sword, st) < 2;
	m[CRAFTAX_ACTION_MAKE_IRON_SWORD] = wood >= 1 && stone >= 1 && iron >= 1 && coal >= 1
										&& table && furnace && CF(inv_sword, st) < 3;
	m[CRAFTAX_ACTION_MAKE_DIAMOND_SWORD] =
		wood >= 1 && gem >= 2 && table && CF(inv_sword, st) < 4;
	m[CRAFTAX_ACTION_MAKE_IRON_ARMOUR] =
		iron >= 3 && coal >= 3 && table && furnace
		&& (CF2(inv_armour, 0, st) < 1 || CF2(inv_armour, 1, st) < 1
			|| CF2(inv_armour, 2, st) < 1 || CF2(inv_armour, 3, st) < 1);
	m[CRAFTAX_ACTION_MAKE_DIAMOND_ARMOUR] =
		gem >= 3 && table
		&& (CF2(inv_armour, 0, st) < 2 || CF2(inv_armour, 1, st) < 2
			|| CF2(inv_armour, 2, st) < 2 || CF2(inv_armour, 3, st) < 2);
	m[CRAFTAX_ACTION_MAKE_ARROW] =
		stone >= 1 && wood >= 1 && table && CF(inv_arrows, st) < 99;
	m[CRAFTAX_ACTION_MAKE_TORCH] =
		coal >= 1 && wood >= 1 && table && CF(inv_torches, st) < 99;
	m[CRAFTAX_ACTION_DESCEND] =
		under == CRAFTAX_ITEM_LADDER_DOWN
		&& CF2(monsters_killed, lvl, st) >= CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL
		&& CF(player_level, st) < CRAFTAX_NUM_LEVELS - 1;
	m[CRAFTAX_ACTION_ASCEND] =
		under == CRAFTAX_ITEM_LADDER_UP && CF(player_level, st) > 0;
	m[CRAFTAX_ACTION_SHOOT_ARROW] = CF(inv_bow, st) >= 1 && CF(inv_arrows, st) >= 1
									&& shots < CRAFTAX_MAX_PLAYER_PROJECTILES;
	m[CRAFTAX_ACTION_CAST_FIREBALL] =
		mana >= 2 && shots < CRAFTAX_MAX_PLAYER_PROJECTILES && CF2(learned_spells, 0, st);
	m[CRAFTAX_ACTION_CAST_ICEBALL] =
		mana >= 2 && shots < CRAFTAX_MAX_PLAYER_PROJECTILES && CF2(learned_spells, 1, st);
	for (int p = 0; p < 6; p++)
		m[CRAFTAX_ACTION_DRINK_POTION_RED + p] = CF2(inv_potions, p, st) > 0;
	m[CRAFTAX_ACTION_READ_BOOK] = CF(inv_books, st) > 0;
	m[CRAFTAX_ACTION_ENCHANT_SWORD] = spell && CF(inv_sword, st) > 0;
	m[CRAFTAX_ACTION_ENCHANT_BOW] = spell && CF(inv_bow, st) > 0;
	m[CRAFTAX_ACTION_ENCHANT_ARMOUR] =
		spell
		&& (CF2(inv_armour, 0, st) + CF2(inv_armour, 1, st) + CF2(inv_armour, 2, st)
			+ CF2(inv_armour, 3, st)) > 0;
	m[CRAFTAX_ACTION_LEVEL_UP_DEXTERITY] =
		xp >= 1 && CF(player_dexterity, st) < CRAFTAX_MAX_ATTRIBUTE;
	m[CRAFTAX_ACTION_LEVEL_UP_STRENGTH] =
		xp >= 1 && CF(player_strength, st) < CRAFTAX_MAX_ATTRIBUTE;
	m[CRAFTAX_ACTION_LEVEL_UP_INTELLIGENCE] =
		xp >= 1 && CF(player_intelligence, st) < CRAFTAX_MAX_ATTRIBUTE;
}

__global__ void k_bank_cell(int n, int cells, int reservoir, const uint8_t* flags,
							int32_t* occupied, int32_t* claim, int32_t* last,
							int32_t* out, int32_t* banked, int32_t* origin,
							const float* weights, const float* floor,
							int32_t* res_tag, int32_t* seen, uint64_t salt,
							const float* refresh, int32_t* saves, int32_t* slots,
							int32_t* flagged, int32_t* counts, int32_t* sheet,
							int32_t* store, const float* recent) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	uint32_t tag = cf_cell_state(i);
	int32_t marked = (int32_t)(tag & 0x7fffffffu);
	int cell = (int)(tag % (uint32_t)cells);
	int base = cell * reservoir;
	out[i] = marked;
	banked[i] = 0;
	origin[i] = -1;
	store[i] = -1;
	if (flags != nullptr && flags[i]) {
		flagged[atomicAdd(&counts[1], 1)] = i;
		return;
	}
	if (refresh[0] > 0.0f && cell == (int)refresh[0] - 1) return;
	int arrived = (last[i] != cell);
	last[i] = cell;
	int want = 0;
	int j = -1;
	if (arrived) {
		int owner = res_tag[cell];
		if (owner == -1 || owner == marked) {
			int count = atomicAdd(&seen[cell], 1) + 1;
			int draw = count;
			if (recent[0] > 0.0f && (float)draw > recent[0])
				draw = max((int)recent[0], 1);
			if (count <= reservoir) {
				j = count - 1;
				want = count == 1 ? 1 : 3;
			} else {
				uint64_t h = salt ^ ((uint64_t)i * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)tag << 20);
				h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
				if ((int)(h % (uint64_t)draw) < reservoir) {
					j = (int)((h >> 40) % (uint64_t)reservoir);
					want = 3;
				}
			}
		} else if (floor[0] > 0.0f && weights[cell] < floor[0]) {
			j = 0;
			want = 2;
		}
	}
	int slot = base + (j < 0 ? 0 : j);
	int mine = (want && atomicCAS(&claim[slot], -1, i) == -1) ? want : 0;
	if (mine) {
		if (mine == 2) {
			for (int k = 1; k < reservoir; k++) occupied[base + k] = 0;
			seen[cell] = 1;
		}
		occupied[slot] = 1;
		if (j <= 0) {
			res_tag[cell] = marked;
			cf_describe(i, sheet + (size_t)cell * kSheetCols);
		}
		int at = atomicAdd(&counts[0], 1);
		saves[at] = i;
		slots[at] = slot;
		store[i] = slot;
	}
	banked[i] = mine;
}

__global__ void k_bank_save(const int32_t* counts, const int32_t* saves,
							const int32_t* slots, uint8_t* archive) {
	for (int j = blockIdx.x; j < counts[0]; j += gridDim.x)
		blob_save(saves[j], archive + (size_t)slots[j] * kBlobBytes);
}

__global__ void k_leash(int n, uint8_t* flags, float* terminals,
						int32_t* warm_steps, const float* leash) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	if (flags[i]) {
		warm_steps[i] = 0;
		return;
	}
	if (warm_steps[i] <= 0) return;
	warm_steps[i] += 1;
	float cap = leash[0];
	if (cap > 0.0f && (float)warm_steps[i] > cap) {
		flags[i] = 1;
		terminals[i] = 2.0f;
		warm_steps[i] = 0;
	}
}

__global__ void k_snap(int n, float* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const CraftaxState* st = &((CraftaxState*)g_cf_state_base)[i];
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		out[(size_t)i * CRAFTAX_NUM_ACHIEVEMENTS + c] =
			CF2(achievements, c, st) ? 1.0f : 0.0f;
}

__global__ void k_place_list(int cells, int reservoir, const int32_t* counts,
							 const int32_t* flagged, const float* cdf,
							 const int32_t* occupied, const uint8_t* archive,
							 int32_t* origin, uint64_t seed, const float* share,
							 int32_t* warm_steps, const float* refresh,
							 int32_t* place) {
	float total = cdf[cells];
	if (total <= 0.0f) return;
	for (int j = blockIdx.x; j < counts[1]; j += gridDim.x) {
		int i = flagged[j];
		uint64_t h = seed ^ ((uint64_t)i * 0x9E3779B97F4A7C15ull);
		h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
		float u = (float)((h >> 11) & 0xFFFFFFull) / (float)0x1000000;
		uint64_t g = h * 0xc4ceb9fe1a85ec53ull;
		float pick = (float)((g >> 11) & 0xFFFFFFull) / (float)0x1000000;
		if (pick >= share[0]) continue;
		float target = u * total;
		int lo = 0, hi = cells - 1;
		while (lo < hi) {
			int mid = (lo + hi) / 2;
			if (cdf[mid] < target) lo = mid + 1; else hi = mid;
		}
		if (refresh[0] > 0.0f && lo == (int)refresh[0] - 1) {
			if (threadIdx.x == 0) origin[i] = lo;
			continue;
		}
		int base = lo * reservoir;
		int held = 0;
		for (int k = 0; k < reservoir; k++) held += occupied[base + k] ? 1 : 0;
		if (held == 0) continue;
		int r = (int)((g >> 40) % (uint64_t)held);
		int slot = base;
		for (int k = 0; k < reservoir; k++) {
			if (!occupied[base + k]) continue;
			if (r == 0) { slot = base + k; break; }
			r--;
		}
		if (threadIdx.x == 0) {
			origin[i] = lo;
			warm_steps[i] = 1;
			place[i] = slot;
		}
		blob_restore(i, archive + (size_t)slot * kBlobBytes);
	}
}

struct Pool {
	CuVec v;
	CfPool* pristine;
	uint8_t* tail;
	int strayed;
	uint8_t* archive;
	int32_t* occupied;
	int32_t* claim;
	int32_t* last;
	int32_t* saves;
	int32_t* slots;
	int32_t* flagged;
	int32_t* counts;
	int32_t* sheet;
	int32_t* res_tag;
	float* res_vit;
	int32_t* seen;
	int32_t* warm_steps;
	int capacity;
	int cells;
	int reservoir;
	unsigned long long tick;
};

// Points the constant binding at a caller-owned CraftaxState array. The pointer
// arrives as a kernel argument, the way k_bind already takes XLA's obs buffer,
// so a rebind survives the CUDA graph capture that cu_vec_bind's device-side
// copy exists to satisfy.
// cf_slot derives an env index by subtracting state_base from a state pointer,
// so envs[i].state must move with the base or the difference is meaningless.
__global__ void k_rebind_states(CfPool* binding, char* states) {
	binding->state_base = states;
}

__global__ void k_rebind_envs(Craftax* envs, int n, char* states) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	envs[i].state = &((CraftaxState*)states)[i];
}

// The 62 SoA fields are separate allocations only by habit; the kernels reach
// them through g_cf_pool, so they can be slices of one caller-owned buffer.
__global__ void k_rebind_soa(CfPool* binding, char* base, int n) {
	(void)n;
#if CF_ENV_MAJOR
	// Every field starts at its own offset inside a per-env row; the stride from
	// one env to the next is cf_row_bytes, folded into cf_row_<field>.
	size_t off = 0;
#define CF_POINT(f, T, C)                       \
	binding->f = (T*)(base + off);              \
	off += (sizeof(T) * (C) + 7) / 8 * 8;
	CF_SOA_FIELDS(CF_POINT)
#undef CF_POINT
#else
	char* q = base;
#define CF_POINT(f, T, C)                                     \
	binding->f = (T*)q;                                       \
	q += ((size_t)sizeof(T) * (C) * (size_t)n + 255) / 256 * 256;
	CF_SOA_FIELDS(CF_POINT)
#undef CF_POINT
#endif
}

__global__ void k_rebind_pool(CfPool* binding, const CfPool* pristine) {
	*binding = *pristine;
}

// EnvTail's fields are members of Craftax, not a separate allocation, so they
// are copied rather than aliased. At 16384 envs that is ~5 MB a step against a
// ~1.3 GB state, so it stays in the noise.
__global__ void k_tail_save(const Craftax* envs, int n, uint8_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const Craftax* e = &envs[i];
	EnvTail* tail = &((EnvTail*)out)[i];
	tail->rng = e->rng;
	tail->seed = e->seed;
	tail->rng_key = e->rng_key;
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		tail->achievements[c] = e->achievements[c];
	tail->episode_return_accum = e->episode_return_accum;
	tail->episode_length_accum = e->episode_length_accum;
	tail->curriculum_episode = e->curriculum_episode;
}

__global__ void k_tail_load(Craftax* envs, int n, const uint8_t* in) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	Craftax* e = &envs[i];
	const EnvTail* tail = &((const EnvTail*)in)[i];
	e->rng = tail->rng;
	e->seed = tail->seed;
	e->rng_key = tail->rng_key;
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		e->achievements[c] = tail->achievements[c];
	e->episode_return_accum = tail->episode_return_accum;
	e->episode_length_accum = tail->episode_length_accum;
	e->curriculum_episode = tail->curriculum_episode;
}

void launch_step(CuVec* v, cudaStream_t st) {
	cf_bind(v, st);
	int n = v->num_envs;
	cudaMemsetAsync(v->d_reset_count, 0, sizeof(int), st);
	cudaMemsetAsync(v->d_spawn_count, 0, sizeof(int), st);
	cu_launch_step_run(v->d_envs, n, v->d_resets, v->d_reset_flags, st);
	int tail = (n * 32 + 255) / 256;
	if (tail > 512) tail = 512;
	k_spawn_tail<<<tail, 256, 0, st>>>();
	if (v->lazy) {
		cu_launch_reset_list_warp(v->d_envs, v->d_resets, n, st);
	} else {
		k_reset_list<<<(n + 63) / 64, 64, 0, st>>>(v->d_envs, v->d_resets);
	}
	dim3 block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
	int grid = (n + CRAFTAX_ENC_WARPS_PER_BLOCK - 1) / CRAFTAX_ENC_WARPS_PER_BLOCK;
	k_encode<<<grid, block, 0, st>>>(v->d_envs, n);
	k_encode_tail<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n);
}

// Undoes a rebind left behind by the caller-owned state path, so an entry point
// that works on the pool's own arrays never writes into a stale XLA buffer.
void repool(Pool* p, cudaStream_t st) {
	if (!p->strayed) return;
	int n = p->v.num_envs;
	k_rebind_pool<<<1, 1, 0, st>>>(p->v.d_binding, p->pristine);
	k_rebind_envs<<<(n + 255) / 256, 256, 0, st>>>(p->v.d_envs, n,
												   (char*)p->v.d_states);
	k_set_envs<<<1, 1, 0, st>>>(p->v.d_envs);
	k_tail_load<<<(n + 255) / 256, 256, 0, st>>>(p->v.d_envs, n, p->tail);
	p->strayed = 0;
}

// Parks the pool's own tail before the caller-owned path overwrites it. The
// tail fields are members of Craftax, which both paths share, so leaving is the
// only moment they can be saved.
void stray(Pool* p, cudaStream_t st) {
	if (p->strayed) return;
	int n = p->v.num_envs;
	k_set_envs<<<1, 1, 0, st>>>(p->v.d_envs);
	k_tail_save<<<(n + 255) / 256, 256, 0, st>>>(p->v.d_envs, n, p->tail);
	p->strayed = 1;
}

// Points every piece of per-env state at caller-owned memory. The pool's own
// arrays go dormant until a pool-owned entry point calls repool.
void rebind(Pool* p, uint8_t* states, uint8_t* soa, cudaStream_t st) {
	CuVec* v = &p->v;
	int n = v->num_envs;
	stray(p, st);
	k_rebind_states<<<1, 1, 0, st>>>(v->d_binding, (char*)states);
	k_rebind_soa<<<1, 1, 0, st>>>(v->d_binding, (char*)soa, n);
	k_rebind_envs<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (char*)states);
	cf_bind(v, st);
	k_set_envs<<<1, 1, 0, st>>>(v->d_envs);
}

}  // namespace

extern "C" {

void craftaxcu_dims(int32_t* out) {
	out[0] = CRAFTAX_OBS_SIZE;
	out[1] = CRAFTAX_NUM_ACTIONS;
	out[2] = (int32_t)sizeof(CraftaxState);
	out[3] = CRAFTAX_NUM_ACHIEVEMENTS;
	out[4] = kBlobBytes;
}

void* craftaxcu_create(int32_t num_envs, uint64_t seed, uint32_t fixed, uint32_t coarse) {
	Pool* p = new Pool();
	cudaDeviceSynchronize();
	cudaMemcpyToSymbol(g_craftax_fixed_world, &fixed, sizeof(uint32_t));
	p->v.coarse = coarse;
	cu_vec_init(&p->v, num_envs, seed);
	// The pool's own binding and tail, kept so an entry point stepping the pool's
	// arrays can undo whatever craftaxcu_step_full left behind. The tail is here
	// because its fields are members of Craftax, which both paths share.
	cudaMalloc(&p->pristine, sizeof(CfPool));
	cudaMemcpy(p->pristine, p->v.d_binding, sizeof(CfPool), cudaMemcpyDeviceToDevice);
	cudaMalloc(&p->tail, (size_t)num_envs * kTailBytes);
	cudaMemset(p->tail, 0, (size_t)num_envs * kTailBytes);
	p->strayed = 0;
	p->archive = nullptr;
	p->occupied = nullptr;
	p->saves = nullptr;
	p->slots = nullptr;
	p->flagged = nullptr;
	p->counts = nullptr;
	p->sheet = nullptr;
	p->claim = nullptr;
	p->last = nullptr;
	p->res_tag = nullptr;
	p->res_vit = nullptr;
	p->seen = nullptr;
	p->warm_steps = nullptr;
	p->capacity = 0;
	p->cells = 0;
	p->reservoir = 1;
	p->tick = 0;
	return p;
}

int32_t craftaxcu_archive(void* handle, int32_t capacity, int32_t reservoir) {
	Pool* p = (Pool*)handle;
	if (reservoir < 1) reservoir = 1;
	p->reservoir = reservoir;
	p->cells = capacity / reservoir;
	capacity = p->cells * reservoir;
	p->capacity = capacity;
	if (cudaMalloc(&p->archive, (size_t)capacity * kBlobBytes) != cudaSuccess) {
		p->archive = nullptr;
		return -1;
	}
	if (cudaMalloc(&p->occupied, (size_t)capacity * sizeof(int32_t)) != cudaSuccess)
		return -1;
	if (cudaMalloc(&p->saves, (size_t)p->v.num_envs * sizeof(int32_t)) != cudaSuccess)
		return -1;
	if (cudaMalloc(&p->slots, (size_t)p->v.num_envs * sizeof(int32_t)) != cudaSuccess)
		return -1;
	if (cudaMalloc(&p->flagged, (size_t)p->v.num_envs * sizeof(int32_t)) != cudaSuccess)
		return -1;
	if (cudaMalloc(&p->counts, 2 * sizeof(int32_t)) != cudaSuccess) return -1;
	if (cudaMalloc(&p->sheet, (size_t)p->cells * kSheetCols * sizeof(int32_t)) !=
		cudaSuccess)
		return -1;
	cudaMemset(p->sheet, 0xFF, (size_t)p->cells * kSheetCols * sizeof(int32_t));
	if (cudaMalloc(&p->claim, (size_t)capacity * sizeof(int32_t)) != cudaSuccess)
		return -1;
	if (cudaMalloc(&p->last, (size_t)p->v.num_envs * sizeof(int32_t)) != cudaSuccess)
		return -1;
	cudaMemset(p->last, 0xFF, (size_t)p->v.num_envs * sizeof(int32_t));
	cudaMemset(p->occupied, 0, (size_t)capacity * sizeof(int32_t));
	if (cudaMalloc(&p->res_tag, (size_t)p->cells * sizeof(int32_t)) != cudaSuccess)
		return -1;
	if (cudaMalloc(&p->res_vit, (size_t)p->cells * sizeof(float)) != cudaSuccess)
		return -1;
	if (cudaMalloc(&p->seen, (size_t)p->cells * sizeof(int32_t)) != cudaSuccess)
		return -1;
	cudaMemset(p->res_tag, 0xFF, (size_t)p->cells * sizeof(int32_t));
	cudaMemset(p->res_vit, 0, (size_t)p->cells * sizeof(float));
	cudaMemset(p->seen, 0, (size_t)p->cells * sizeof(int32_t));
	if (cudaMalloc(&p->warm_steps, (size_t)p->v.num_envs * sizeof(int32_t)) !=
		cudaSuccess)
		return -1;
	cudaMemset(p->warm_steps, 0, (size_t)p->v.num_envs * sizeof(int32_t));
	return 0;
}

int32_t craftaxcu_step_archive(void* handle, void* stream, uint8_t* states,
							   uint8_t* soa, uint8_t* tail_leaf,
							   const int32_t* actions,
							   const float* weights, const float* cdf,
							   const float* share, const float* floor,
							   const float* leash, const float* refresh, float* obs,
							   float* rewards, float* terminals, int32_t* cells,
							   int32_t* origin, int32_t* banked, float* laurels,
							   int32_t* place, int32_t* store, const float* recent) {
	Pool* p = (Pool*)handle;
	CuVec* v = &p->v;
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	rebind(p, states, soa, st);
	k_tail_load<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail_leaf);
	k_bind<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (CraftaxObs*)obs, rewards,
											terminals);
	k_cast<<<(n + 255) / 256, 256, 0, st>>>(actions, v->d_actions, n);
	cudaMemsetAsync(v->d_reset_count, 0, sizeof(int), st);
	cudaMemsetAsync(v->d_spawn_count, 0, sizeof(int), st);
	cu_launch_step_run(v->d_envs, n, v->d_resets, v->d_reset_flags, st);
	int tail = (n * 32 + 255) / 256;
	if (tail > 512) tail = 512;
	k_spawn_tail<<<tail, 256, 0, st>>>();
	k_leash<<<(n + 255) / 256, 256, 0, st>>>(n, v->d_reset_flags, terminals,
											 p->warm_steps, leash);
	k_snap<<<(n + 255) / 256, 256, 0, st>>>(n, laurels);
	cudaMemsetAsync(p->claim, 0xFF, (size_t)p->capacity * sizeof(int32_t), st);
	cudaMemsetAsync(place, 0xFF, (size_t)n * sizeof(int32_t), st);
	cudaMemsetAsync(p->counts, 0, 2 * sizeof(int32_t), st);
	int lists = n < 512 ? n : 512;
	k_bank_cell<<<(n + 127) / 128, 128, 0, st>>>(
		n, p->cells, p->reservoir, v->d_reset_flags, p->occupied, p->claim, p->last,
		cells, banked, origin, weights, floor, p->res_tag, p->seen,
		0xD1B54A32D192ED03ull * (uint64_t)(p->tick + 1), refresh, p->saves, p->slots,
		p->flagged, p->counts, p->sheet, store, recent);
	k_bank_save<<<lists, 128, 0, st>>>(p->counts, p->saves, p->slots, p->archive);
	if (v->lazy) {
		cu_launch_reset_list_warp(v->d_envs, v->d_resets, n, st);
	} else {
		k_reset_list<<<(n + 63) / 64, 64, 0, st>>>(v->d_envs, v->d_resets);
	}
	k_place_list<<<lists, 128, 0, st>>>(p->cells, p->reservoir, p->counts, p->flagged,
										cdf, p->occupied, p->archive, origin,
										0x9E3779B97F4A7C15ull * (uint64_t)(++p->tick),
										share, p->warm_steps, refresh, place);
	dim3 block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
	int grid = (n + CRAFTAX_ENC_WARPS_PER_BLOCK - 1) / CRAFTAX_ENC_WARPS_PER_BLOCK;
	k_encode<<<grid, block, 0, st>>>(v->d_envs, n);
	k_encode_tail<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n);
	k_tail_save<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail_leaf);
	return (int32_t)cudaPeekAtLastError();
}

__global__ void k_groups(int cells, const int32_t* sheet, int32_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= cells) return;
	const int32_t* row = sheet + (size_t)i * kSheetCols;
	out[i] = row[0] < 0 ? -1 : row[0] * 128 + row[2];
}

int32_t craftaxcu_groups(void* handle, void* stream, int32_t* out) {
	Pool* p = (Pool*)handle;
	if (p->sheet == nullptr) return -1;
	k_groups<<<(p->cells + 255) / 256, 256, 0, (cudaStream_t)stream>>>(p->cells, p->sheet, out);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_owners(void* handle, void* stream, int32_t* out) {
	Pool* p = (Pool*)handle;
	if (p->res_tag == nullptr) return -1;
	return (int32_t)cudaMemcpyAsync(out, p->res_tag, (size_t)p->cells * sizeof(int32_t),
									cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
}

int32_t craftaxcu_describe(void* handle, int32_t* out) {
	Pool* p = (Pool*)handle;
	if (p->sheet == nullptr) return -1;
	cudaDeviceSynchronize();
	return (int32_t)cudaMemcpy(out, p->sheet,
							   (size_t)p->cells * kSheetCols * sizeof(int32_t),
							   cudaMemcpyDeviceToHost);
}

int32_t craftaxcu_map(void* handle, int32_t env, uint8_t* out) {
	Pool* p = (Pool*)handle;
	if (env < 0 || env >= p->v.num_envs) return -1;
	cudaDeviceSynchronize();
	return (int32_t)cudaMemcpy(out, p->v.d_states[env].map, sizeof(p->v.d_states[env].map),
							   cudaMemcpyDeviceToHost);
}

void craftaxcu_close(void* handle) {
	Pool* p = (Pool*)handle;
	if (p->archive) cudaFree(p->archive);
	if (p->occupied) cudaFree(p->occupied);
	if (p->saves) cudaFree(p->saves);
	if (p->slots) cudaFree(p->slots);
	if (p->flagged) cudaFree(p->flagged);
	if (p->counts) cudaFree(p->counts);
	if (p->sheet) cudaFree(p->sheet);
	if (p->claim) cudaFree(p->claim);
	if (p->last) cudaFree(p->last);
	if (p->res_tag) cudaFree(p->res_tag);
	if (p->res_vit) cudaFree(p->res_vit);
	if (p->seen) cudaFree(p->seen);
	if (p->warm_steps) cudaFree(p->warm_steps);
	if (p->pristine) cudaFree(p->pristine);
	cu_vec_free(&p->v);
	delete p;
}

int32_t craftaxcu_observe(void* handle, void* stream, float* obs, float* rewards,
						  float* terminals) {
	CuVec* v = (CuVec*)handle;
	repool((Pool*)handle, (cudaStream_t)stream);
	cf_bind(v, (cudaStream_t)stream);
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	k_bind<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (CraftaxObs*)obs, rewards, terminals);
	dim3 block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
	int grid = (n + CRAFTAX_ENC_WARPS_PER_BLOCK - 1) / CRAFTAX_ENC_WARPS_PER_BLOCK;
	k_encode<<<grid, block, 0, st>>>(v->d_envs, n);
	k_encode_tail<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_reset(void* handle, void* stream, float* obs, float* rewards,
						float* terminals) {
	CuVec* v = (CuVec*)handle;
	repool((Pool*)handle, (cudaStream_t)stream);
	cf_bind(v, (cudaStream_t)stream);
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	k_bind<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (CraftaxObs*)obs, rewards, terminals);
	cudaMemsetAsync(rewards, 0, (size_t)n * sizeof(float), st);
	cudaMemsetAsync(terminals, 0, (size_t)n * sizeof(float), st);
	k_reset_all<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, v->d_resets);
	if (v->lazy) {
		cu_launch_reset_list_warp(v->d_envs, v->d_resets, n, st);
	} else {
		k_reset_list<<<(n + 63) / 64, 64, 0, st>>>(v->d_envs, v->d_resets);
	}
	cudaMemsetAsync(v->d_reset_count, 0, sizeof(int), st);
	dim3 block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
	int grid = (n + CRAFTAX_ENC_WARPS_PER_BLOCK - 1) / CRAFTAX_ENC_WARPS_PER_BLOCK;
	k_encode<<<grid, block, 0, st>>>(v->d_envs, n);
	k_encode_tail<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_step(void* handle, void* stream, const int32_t* actions, float* obs,
					   float* rewards, float* terminals) {
	CuVec* v = (CuVec*)handle;
	repool((Pool*)handle, (cudaStream_t)stream);
	cf_bind(v, (cudaStream_t)stream);
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	k_bind<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (CraftaxObs*)obs, rewards, terminals);
	k_cast<<<(n + 255) / 256, 256, 0, st>>>(actions, v->d_actions, n);
	launch_step(v, st);
	return (int32_t)cudaPeekAtLastError();
}

// Banks the caller-owned state of every env with a non-negative slot into the
// device archive. The slots come from the JAX side; nothing here decides.
int32_t craftaxcu_bank(void* handle, void* stream, uint8_t* states, uint8_t* soa,
					   uint8_t* tail, const int32_t* slots) {
	Pool* p = (Pool*)handle;
	CuVec* v = &p->v;
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	if (p->archive == nullptr) return -1;
	rebind(p, states, soa, st);
	k_tail_load<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail);
	k_bank_slots<<<n, 128, 0, st>>>(n, slots, p->archive);
	return (int32_t)cudaPeekAtLastError();
}

// Restores archived blobs into the caller-owned state for every env with a
// non-negative slot, then hands the tail back out so JAX sees the new state.
int32_t craftaxcu_revive(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						 uint8_t* tail, const int32_t* slots) {
	Pool* p = (Pool*)handle;
	CuVec* v = &p->v;
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	if (p->archive == nullptr) return -1;
	rebind(p, states, soa, st);
	k_tail_load<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail);
	k_revive_slots<<<n, 128, 0, st>>>(n, slots, p->archive);
	k_tail_save<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_soa_bytes(int32_t num_envs) {
#if CF_ENV_MAJOR
	return (int32_t)(cf_row_bytes * (size_t)num_envs);
#else
	size_t total = 0;
#define CF_SPAN(f, T, C) \
	total += ((size_t)sizeof(T) * (C) * (size_t)num_envs + 255) / 256 * 256;
	CF_SOA_FIELDS(CF_SPAN)
#undef CF_SPAN
	return (int32_t)total;
#endif
}

int32_t craftaxcu_soa_row_bytes(void) {
#if CF_ENV_MAJOR
	return (int32_t)cf_row_bytes;
#else
	return 0;
#endif
}

int32_t craftaxcu_tail_bytes(void) { return kTailBytes; }
int32_t craftaxcu_tail_key(void) { return (int32_t)offsetof(EnvTail, rng_key); }

int32_t craftaxcu_reset_full(void* handle, void* stream, uint8_t* states,
							 uint8_t* soa, uint8_t* tail, float* obs,
							 float* rewards, float* terminals) {
	Pool* p = (Pool*)handle;
	CuVec* v = &p->v;
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	rebind(p, states, soa, st);
	k_bind<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (CraftaxObs*)obs, rewards,
											terminals);
	cudaMemsetAsync(rewards, 0, (size_t)n * sizeof(float), st);
	cudaMemsetAsync(terminals, 0, (size_t)n * sizeof(float), st);
	k_reset_all<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, v->d_resets);
	if (v->lazy) {
		cu_launch_reset_list_warp(v->d_envs, v->d_resets, n, st);
	} else {
		k_reset_list<<<(n + 63) / 64, 64, 0, st>>>(v->d_envs, v->d_resets);
	}
	cudaMemsetAsync(v->d_reset_count, 0, sizeof(int), st);
	dim3 block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
	int grid = (n + CRAFTAX_ENC_WARPS_PER_BLOCK - 1) / CRAFTAX_ENC_WARPS_PER_BLOCK;
	k_encode<<<grid, block, 0, st>>>(v->d_envs, n);
	k_encode_tail<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n);
	k_tail_save<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail);
	return (int32_t)cudaPeekAtLastError();
}

// Encodes observations from caller-owned state without stepping, so a restart
// that scatters archived rows into the leaves can produce the matching obs.
int32_t craftaxcu_observe_full(void* handle, void* stream, uint8_t* states,
							   uint8_t* soa, uint8_t* tail, float* obs,
							   float* rewards, float* terminals) {
	Pool* p = (Pool*)handle;
	CuVec* v = &p->v;
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	rebind(p, states, soa, st);
	k_tail_load<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail);
	k_bind<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (CraftaxObs*)obs, rewards,
											terminals);
	dim3 block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
	int grid = (n + CRAFTAX_ENC_WARPS_PER_BLOCK - 1) / CRAFTAX_ENC_WARPS_PER_BLOCK;
	k_encode<<<grid, block, 0, st>>>(v->d_envs, n);
	k_encode_tail<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n);
	return (int32_t)cudaPeekAtLastError();
}

// The state and SoA leaves are aliased in place, so stepping copies neither;
// the tail is copied in and out because its fields are members of Craftax
// rather than an allocation the binding can be pointed at.
// One row per env: episodes finished since the last read, how many of them held
// each achievement at the end, and the summed episode step each was unlocked at.
// Reading empties the log.
__global__ void k_tally(Craftax* envs, int n, float* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	Log* log = &envs[i].log;
	float* row = &out[(size_t)i * (2 * CRAFTAX_NUM_ACHIEVEMENTS + 1)];
	row[0] = log->n;
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++) {
		row[c + 1] = log->achievements[c];
		row[c + 1 + CRAFTAX_NUM_ACHIEVEMENTS] = log->unlocked[c];
	}
	memset(log, 0, sizeof(Log));
}

__global__ void k_portray(int n, int32_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	cf_describe(i, &out[(size_t)i * kSheetCols]);
}

// Describes caller-owned rows, e.g. archived snapshots. Binding them is the same
// rebind every step makes, and the next step rebinds its own leaves.
int32_t craftaxcu_portray(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						  int32_t* out) {
	Pool* p = (Pool*)handle;
	cudaStream_t st = (cudaStream_t)stream;
	int n = p->v.num_envs;
	rebind(p, states, soa, st);
	k_portray<<<(n + 255) / 256, 256, 0, st>>>(n, out);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_step_full(void* handle, void* stream, uint8_t* states,
							uint8_t* soa, uint8_t* tail, const int32_t* actions,
							float* obs, float* rewards, float* terminals, float* tally) {
	Pool* p = (Pool*)handle;
	CuVec* v = &p->v;
	cudaStream_t st = (cudaStream_t)stream;
	int n = v->num_envs;
	rebind(p, states, soa, st);
	k_tail_load<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail);
	k_bind<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, (CraftaxObs*)obs, rewards,
											terminals);
	k_cast<<<(n + 255) / 256, 256, 0, st>>>(actions, v->d_actions, n);
	launch_step(v, st);
	k_tally<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tally);
	k_tail_save<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n, tail);
	return (int32_t)cudaPeekAtLastError();
}

// 1-3, 4-6, 7-9.  Zero is death, and a dead state is never banked.
__device__ inline uint32_t k_vital(int v) {
	if (v < 1) v = 1;
	if (v > 9) v = 9;
	return (uint32_t)((v - 1) / 3);
}

__global__ void k_cells(int n, uint32_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	out[i] = cf_cell_state(i);
}

int32_t craftaxcu_cells(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						uint32_t* out) {
	Pool* p = (Pool*)handle;
	cudaStream_t st = (cudaStream_t)stream;
	int n = p->v.num_envs;
	rebind(p, states, soa, st);
	k_cells<<<(n + 255) / 256, 256, 0, st>>>(n, out);
	return (int32_t)cudaPeekAtLastError();
}

__device__ inline uint32_t cf_world(int i) {
	const CraftaxState* st = &((CraftaxState*)g_cf_state_base)[i];
	uint32_t h = cf_mix(CF_FNV_SEED, st->lazy_floor_keys[0][0]);
	return cf_mix(h, st->lazy_floor_keys[0][1]);
}

__global__ void k_worlds(int n, uint32_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	out[i] = cf_world(i);
}

int32_t craftaxcu_worlds(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						 uint32_t* out) {
	Pool* p = (Pool*)handle;
	cudaStream_t st = (cudaStream_t)stream;
	int n = p->v.num_envs;
	rebind(p, states, soa, st);
	k_worlds<<<(n + 255) / 256, 256, 0, st>>>(n, out);
	return (int32_t)cudaPeekAtLastError();
}

__global__ void k_laurels(int n, float* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const CraftaxState* st = &((CraftaxState*)g_cf_state_base)[i];
	for (int c = 0; c < CRAFTAX_NUM_ACHIEVEMENTS; c++)
		out[i * CRAFTAX_NUM_ACHIEVEMENTS + c] = CF2(achievements, c, st) ? 1.0f : 0.0f;
}

int32_t craftaxcu_laurels(void* handle, void* stream, uint8_t* states, uint8_t* soa,
						  float* out) {
	Pool* p = (Pool*)handle;
	cudaStream_t st = (cudaStream_t)stream;
	int n = p->v.num_envs;
	rebind(p, states, soa, st);
	k_laurels<<<(n + 255) / 256, 256, 0, st>>>(n, out);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_action_mask(void* handle, void* stream, uint8_t* states, uint8_t* soa,
							  uint8_t* out) {
	Pool* p = (Pool*)handle;
	cudaStream_t st = (cudaStream_t)stream;
	int n = p->v.num_envs;
	rebind(p, states, soa, st);
	k_action_mask<<<(n + 255) / 256, 256, 0, st>>>(n, out);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_save(void* handle, void* stream, const int32_t* index, int32_t count,
					   uint8_t* out) {
	CuVec* v = (CuVec*)handle;
	cf_bind(v, (cudaStream_t)stream);
	int n = count > 0 ? count : v->num_envs;
	k_set_envs<<<1, 1, 0, (cudaStream_t)stream>>>(v->d_envs);
	k_save<<<n, 128, 0, (cudaStream_t)stream>>>(n, index, out);
	return (int32_t)cudaPeekAtLastError();
}

int32_t craftaxcu_restore(void* handle, void* stream, const int32_t* index, int32_t count,
						  const uint8_t* in) {
	CuVec* v = (CuVec*)handle;
	cf_bind(v, (cudaStream_t)stream);
	int n = count > 0 ? count : v->num_envs;
	k_set_envs<<<1, 1, 0, (cudaStream_t)stream>>>(v->d_envs);
	k_restore<<<n, 128, 0, (cudaStream_t)stream>>>(n, index, in);
	return (int32_t)cudaPeekAtLastError();
}

}  // extern "C"
