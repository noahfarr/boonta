#ifndef KAGGRICULTURE_H
#define KAGGRICULTURE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define KG_BOARD_SIZE 10
#define KG_TURNS_PER_DAY 24
#define KG_EPISODE_STEPS 720
#define KG_SHED_CAPACITY 100
#define KG_STARTING_MONEY 3000
#define KG_MAX_MARKET_ORDERS 10
#define KG_TOWN_SHOP_UNLOCK_INTERVAL 3
#define KG_TOWN_SHOP_SELL_INTERVAL 4
#define KG_TOWN_CENTER_SELL_INTERVAL 24
#define KG_MAX_SHOP_INSTANCES 8
#define KG_FARM_HAND_COST_MULT 1
#define KG_PRICE_FLOOR 1
#define KG_MARKET_I0 10000

#define KG_NUM_CROPS 5
#define KG_NUM_ANIMALS 3
#define KG_NUM_PRODUCTS 9
#define KG_NUM_ITEMS 12
#define KG_NUM_SHOPS 8
#define KG_NUM_QUADRANTS 4
#define KG_ANIMAL_ITEM_BASE KG_NUM_PRODUCTS
#define KG_PLAYERS 2
#define KG_TILES (KG_BOARD_SIZE * KG_BOARD_SIZE)
#define KG_MAX_UNITS 64

enum {
    KG_WHEAT = 0, KG_CARROT, KG_TOMATO, KG_STRAWBERRY, KG_MELON,
    KG_EGG, KG_MILK, KG_WOOL, KG_FERTILIZER
};

enum { KG_GOOSE = 0, KG_COW, KG_SHEEP };

enum { KG_EMPTY = 0, KG_LOCKED, KG_WEED, KG_PLANT, KG_COOP, KG_PASTURE };

enum {
    KG_OP_PASS = 0, KG_OP_NORTH, KG_OP_SOUTH, KG_OP_EAST, KG_OP_WEST,
    KG_OP_PICKUP, KG_OP_DROP, KG_OP_PLANT, KG_OP_WATER, KG_OP_HARVEST,
    KG_OP_FERTILIZE, KG_OP_DIG, KG_OP_BUILD_COOP, KG_OP_BUILD_PASTURE,
    KG_OP_PLACE, KG_OP_FEED, KG_OP_COLLECT_FERTILIZER, KG_OP_CARE,
    KG_NUM_UNIT_OPS
};

enum {
    KG_MK_NONE = 0, KG_MK_BUY_SEED, KG_MK_BUY_PRODUCT, KG_MK_BUY_ANIMAL,
    KG_MK_SELL, KG_MK_HIRE, KG_MK_BUY_LAND, KG_NUM_MARKET_OPS
};

enum {
    KG_CURVE_LINEAR = 0, KG_CURVE_SQ, KG_CURVE_SQRT, KG_CURVE_LOG, KG_CURVE_LOG10,
    KG_CURVE_HINGE
};

#define KG_HINGE_GAIN 8.0

typedef struct {
    int16_t seed;
    int16_t first_yield_day;
    int16_t max_yield_day;
    int16_t interval;
    int16_t max_yield;
    int16_t ongoing;
} kg_crop_t;

typedef struct {
    int16_t cost;
    int16_t structure;
    int16_t first_yield_day;
    int16_t interval;
    int16_t max_held;
    int16_t product;
} kg_animal_t;

typedef struct {
    float base;
    float i0;
    int32_t below_kind;
    int32_t above_kind;
    float below_amp;
    float above_amp;
    float t;
} kg_curve_t;

typedef struct {
    double base;
    double below_amp;
    double above_amp;
    int32_t below_kind;
    int32_t above_kind;
    double t;
} kg_price_t;

typedef struct {
    int8_t kind;
    int8_t crop;
    int8_t animal;
    int8_t watered_today;
    int8_t consecutive_unwatered;
    int8_t fed_today;
    int8_t consecutive_unfed;
    int8_t cared_today;
    int8_t fertilizer_available;
    int8_t pending_care_bonus;
    int8_t yield_units;
    int16_t planted_day;
    int16_t fertilized_until_day;
    int16_t max_lifespan_step;
} kg_tile_t;

typedef struct {
    float money;
    kg_tile_t tiles[KG_TILES];
    int8_t unit_x[KG_MAX_UNITS];
    int8_t unit_y[KG_MAX_UNITS];
    int32_t num_units;
    int32_t hires_today;
    int32_t quadrants;
    int32_t next_decay;
    int32_t shed[KG_NUM_ITEMS];
    int32_t seeds[KG_NUM_CROPS];
    int16_t inv[KG_MAX_UNITS][KG_NUM_ITEMS];
} kg_farm_t;

typedef struct {
    int32_t step;
    kg_farm_t farms[KG_PLAYERS];
    int32_t market[KG_NUM_PRODUCTS];
    int32_t prices[KG_NUM_PRODUCTS];
    int8_t shops[KG_NUM_SHOPS];
    int8_t shop_order[KG_MAX_SHOP_INSTANCES];
    int32_t shops_unlocked;
    uint32_t price_dirty;
    uint64_t rng;
    float weed_chance;
    int32_t done;
    int32_t episode_steps;
    float reward[KG_PLAYERS];
} kg_state_t;

typedef struct {
    int8_t op;
    int8_t arg;
    int16_t n;
} kg_unit_action_t;

typedef struct {
    int8_t op;
    int8_t item;
    int16_t n;
} kg_market_action_t;

typedef struct {
    kg_unit_action_t units[KG_MAX_UNITS];
    kg_market_action_t orders[KG_MAX_MARKET_ORDERS];
} kg_action_t;

extern const kg_crop_t KG_CROPS[KG_NUM_CROPS];
extern const kg_animal_t KG_ANIMALS[KG_NUM_ANIMALS];
extern kg_curve_t KG_CURVES[KG_NUM_PRODUCTS];
extern kg_price_t KG_PRICES[KG_NUM_PRODUCTS];
void kg_init_tables(void);
extern const int8_t KG_SHOP_DEMAND[KG_NUM_SHOPS][KG_NUM_PRODUCTS];
extern const int32_t KG_LAND_PRICES[3];

void kg_reset(kg_state_t *state, uint64_t seed);
void kg_step(kg_state_t *state, const kg_action_t actions[KG_PLAYERS]);
float kg_market_price(int item, int32_t inventory);
int64_t kg_hire_cost(int32_t already_today);


#define KG_OBS_UNITS 17
#define KG_OBS_TILE_FIELDS 13

typedef struct {
    int8_t *tiles;
    int8_t *lifespan;
    int8_t *pos;
    int16_t *inv;
    int16_t *shed;
    int16_t *seeds;
    float *money;
    int16_t *hands;
    int16_t *hires;
    int16_t *quadrants;
    int32_t *market;
    int16_t *prices;
    int8_t *shops;
    int8_t *day;
    int8_t *hour;
    float *reward;
    int8_t *terminated;
} kg_obs_t;

void kg_batch_reset(kg_state_t *states, int n, uint64_t seed);
void kg_batch_step(kg_state_t *states, const kg_action_t *actions, int n, int threads);
void kg_batch_encode(const kg_state_t *states, int n, kg_obs_t obs, int threads);

#define KG_NUM_HEADS (3 * KG_OBS_UNITS + 2 * KG_MAX_MARKET_ORDERS)

#define KG_NUM_QUANTITIES 25
extern const int32_t KG_QUANTITIES[KG_NUM_QUANTITIES];

#define KG_NUM_JOBS 23
extern const int8_t KG_JOB_OP[KG_NUM_JOBS];
extern const int8_t KG_JOB_ARG[KG_NUM_JOBS];
void kg_legal_jobs(const kg_state_t *state, int player, int8_t *out);

typedef struct {
    int n_envs;
    int threads;
    kg_state_t *states;
    kg_action_t *actions;
    kg_obs_t obs;
    const int32_t *heads;
} kg_vec_t;

kg_vec_t *kg_vec_create(int n_envs, uint64_t seed, int threads, int episode_steps);
void kg_vec_free(kg_vec_t *vec);
void kg_vec_bind(kg_vec_t *vec, kg_obs_t obs, const int32_t *heads);
void kg_vec_reset(kg_vec_t *vec, uint64_t seed);
void kg_vec_step(kg_vec_t *vec);
void kg_decode_heads(const int32_t *heads, kg_action_t actions[KG_PLAYERS]);

#ifdef __cplusplus
}
#endif

#endif
