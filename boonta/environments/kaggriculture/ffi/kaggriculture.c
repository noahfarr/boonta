#include "kaggriculture.h"

#include <limits.h>
#include <math.h>
#include <string.h>

const kg_crop_t KG_CROPS[KG_NUM_CROPS] = {
    {10, 2, 4, 0, 6, 0},
    {20, 2, 3, 0, 4, 0},
    {50, 8, 8, 1, 4, 1},
    {100, 10, 10, 2, 4, 1},
    {80, 10, 12, 0, 6, 0},
};

const kg_animal_t KG_ANIMALS[KG_NUM_ANIMALS] = {
    {300, KG_COOP, 4, 1, 4, KG_EGG},
    {400, KG_PASTURE, 8, 2, 6, KG_MILK},
    {500, KG_PASTURE, 6, 3, 6, KG_WOOL},
};

const int32_t KG_LAND_PRICES[3] = {1000, 2000, 4000};

static const struct {
    float base;
    float t;
    int below_kind;
    float below_target;
    int above_kind;
    float above_target;
} KG_MARKET_PARAMS[KG_NUM_PRODUCTS] = {
    { 25, 400, KG_CURVE_SQRT,   0.80f, KG_CURVE_LOG,    0.20f},
    { 35, 450, KG_CURVE_HINGE,  1.00f, KG_CURVE_SQRT,   0.70f},
    { 60, 200, KG_CURVE_HINGE,  0.40f, KG_CURVE_SQRT,   0.60f},
    {120, 100, KG_CURVE_SQRT,   0.70f, KG_CURVE_LINEAR, 1.60f},
    {250, 300, KG_CURVE_LOG,    0.20f, KG_CURVE_SQ,     3.60f},
    { 50, 332, KG_CURVE_HINGE,  0.40f, KG_CURVE_LOG,    0.20f},
    {160, 122, KG_CURVE_SQRT,   0.60f, KG_CURVE_LINEAR, 1.60f},
    {200, 105, KG_CURVE_LOG,    0.20f, KG_CURVE_SQ,     3.20f},
    {100, 200, KG_CURVE_LINEAR, 0.40f, KG_CURVE_LINEAR, 0.40f},
};

const int8_t KG_SHOP_DEMAND[KG_NUM_SHOPS][KG_NUM_PRODUCTS] = {
    {1, 0, 0, 0, 0, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 0, 1, 0, 0},
    {1, 0, 0, 1, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 2, 0},
    {1, 0, 0, 1, 0, 0, 1, 0, 0},
    {0, 2, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 1, 0, 0},
    {1, 1, 1, 1, 0, 0, 0, 0, 0},
};

kg_curve_t KG_CURVES[KG_NUM_PRODUCTS];

static double shape_of(int kind, double x, double t) {
    if (x < 0.0) x = 0.0;
    switch (kind) {
        case KG_CURVE_LINEAR: return x;
        case KG_CURVE_SQ:     return x * x;
        case KG_CURVE_SQRT:   return sqrt(x);
        case KG_CURVE_LOG:    return log(1.0 + x);
        case KG_CURVE_LOG10:  return log10(1.0 + x);
        case KG_CURVE_HINGE: {
            if (t <= 0.0) return x;
            double u = x / t;
            double over = u - 1.0;
            if (over < 0.0) over = 0.0;
            return u + KG_HINGE_GAIN * over * over;
        }
        default:              return x;
    }
}

static int64_t KG_FIB[92];
kg_price_t KG_PRICES[KG_NUM_PRODUCTS];
static int kg_tables_ready = 0;

static void ensure_tables(void) {
    if (kg_tables_ready) return;
    int64_t a = 1, b = 1;
    for (int i = 0; i < 92; i++) {
        KG_FIB[i] = a;
        int64_t next = a + b;
        a = b;
        b = next;
    }
    for (int i = 0; i < KG_NUM_PRODUCTS; i++) {
        double t = KG_MARKET_PARAMS[i].t;
        KG_PRICES[i].base = KG_MARKET_PARAMS[i].base;
        KG_PRICES[i].below_kind = KG_MARKET_PARAMS[i].below_kind;
        KG_PRICES[i].above_kind = KG_MARKET_PARAMS[i].above_kind;
        KG_PRICES[i].t = t;
        KG_PRICES[i].below_amp = KG_MARKET_PARAMS[i].below_target * KG_MARKET_PARAMS[i].base /
            shape_of(KG_MARKET_PARAMS[i].below_kind, t, t);
        KG_PRICES[i].above_amp = KG_MARKET_PARAMS[i].above_target * KG_MARKET_PARAMS[i].base /
            shape_of(KG_MARKET_PARAMS[i].above_kind, t, t);
    }
    kg_tables_ready = 1;
}

float kg_market_price(int item, int32_t inventory) {
    ensure_tables();
    const kg_price_t *p = &KG_PRICES[item];
    double price;
    if (inventory < KG_MARKET_I0) {
        price = p->base + p->below_amp *
            shape_of(p->below_kind, (double)(KG_MARKET_I0 - inventory), p->t);
    } else {
        price = p->base - p->above_amp *
            shape_of(p->above_kind, (double)(inventory - KG_MARKET_I0), p->t);
    }
    double rounded = nearbyint(price);
    if (rounded < KG_PRICE_FLOOR) rounded = KG_PRICE_FLOOR;
    return (float)rounded;
}

int64_t kg_hire_cost(int32_t already_today) {
    ensure_tables();
    if (already_today < 0) already_today = 0;
    if (already_today > 91) already_today = 91;
    return (int64_t)KG_FARM_HAND_COST_MULT * KG_FIB[already_today];
}

static int quadrant_of(int x, int y) {
    int half = KG_BOARD_SIZE / 2;
    if (y < half) return x < half ? 0 : 1;
    return x < half ? 2 : 3;
}

static void refresh_prices(kg_state_t *state) {
    uint32_t dirty = state->price_dirty;
    while (dirty) {
        int i = __builtin_ctz(dirty);
        dirty &= dirty - 1;
        state->prices[i] = (int32_t)kg_market_price(i, state->market[i]);
    }
    state->price_dirty = 0;
}

static inline void market_add(kg_state_t *state, int item, int32_t delta) {
    state->market[item] += delta;
    state->price_dirty |= 1u << item;
}

static uint32_t next_random(kg_state_t *state);

static void draw_shop_order(kg_state_t *state) {
    for (int i = 0; i < KG_MAX_SHOP_INSTANCES; i++) {
        state->shop_order[i] = (int8_t)(next_random(state) % (uint32_t)KG_NUM_SHOPS);
    }
}

void kg_reset(kg_state_t *state, uint64_t seed) {
    int32_t kept_episode_steps = state->episode_steps;
    ensure_tables();
    memset(state, 0, sizeof(*state));
    state->rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    state->weed_chance = 0.005f;
    state->episode_steps = kept_episode_steps;

    for (int p = 0; p < KG_PLAYERS; p++) {
        kg_farm_t *farm = &state->farms[p];
        farm->money = (float)KG_STARTING_MONEY;
        farm->num_units = 1;
        farm->quadrants = 0;
        farm->next_decay = INT32_MAX;
        for (int y = 0; y < KG_BOARD_SIZE; y++) {
            for (int x = 0; x < KG_BOARD_SIZE; x++) {
                kg_tile_t *tile = &farm->tiles[y * KG_BOARD_SIZE + x];
                tile->kind = quadrant_of(x, y) == 0 ? KG_EMPTY : KG_LOCKED;
                tile->crop = -1;
                tile->animal = -1;
                tile->planted_day = 0;
                tile->fertilized_until_day = -1;
                tile->max_lifespan_step = -1;
            }
        }
        int half = KG_BOARD_SIZE / 2;
        for (int u = 0; u < KG_MAX_UNITS; u++) {
            farm->unit_x[u] = (int8_t)(half - 1);
            farm->unit_y[u] = (int8_t)(half - 1);
        }
    }

    draw_shop_order(state);
    state->shops_unlocked = 0;
    for (int i = 0; i < KG_NUM_PRODUCTS; i++) state->market[i] = KG_MARKET_I0;
    state->price_dirty = (1u << KG_NUM_PRODUCTS) - 1u;
    refresh_prices(state);
}

void kg_init_tables(void) {
    for (int i = 0; i < KG_NUM_PRODUCTS; i++) {
        double t = KG_MARKET_PARAMS[i].t;
        KG_CURVES[i].base = KG_MARKET_PARAMS[i].base;
        KG_CURVES[i].i0 = (float)KG_MARKET_I0;
        KG_CURVES[i].below_kind = KG_MARKET_PARAMS[i].below_kind;
        KG_CURVES[i].above_kind = KG_MARKET_PARAMS[i].above_kind;
        KG_CURVES[i].t = (float)t;
        KG_CURVES[i].below_amp = (float)(KG_MARKET_PARAMS[i].below_target *
            KG_MARKET_PARAMS[i].base / shape_of(KG_MARKET_PARAMS[i].below_kind, t, t));
        KG_CURVES[i].above_amp = (float)(KG_MARKET_PARAMS[i].above_target *
            KG_MARKET_PARAMS[i].base / shape_of(KG_MARKET_PARAMS[i].above_kind, t, t));
    }
}

static const int8_t MOVE_DX[KG_NUM_UNIT_OPS] = {0, 0, 0, 1, -1};
static const int8_t MOVE_DY[KG_NUM_UNIT_OPS] = {0, -1, 1, 0, 0};

static int is_shed_adjacent(int x, int y) {
    int half = KG_BOARD_SIZE / 2;
    return (x == half - 1 || x == half) && (y == half - 1 || y == half);
}

static int32_t shed_total(const kg_farm_t *farm) {
    int32_t total = 0;
    for (int i = 0; i < KG_NUM_ITEMS; i++) total += farm->shed[i];
    return total;
}

static int inv_take(kg_farm_t *farm, int unit, int item, int32_t n) {
    if (farm->inv[unit][item] < n) return 0;
    farm->inv[unit][item] = (int16_t)(farm->inv[unit][item] - n);
    return 1;
}

static void note_decay(kg_farm_t *farm, int mls) {
    if (mls >= 0 && mls < farm->next_decay) farm->next_decay = mls;
}

static void new_plant(kg_tile_t *tile, int crop, int day) {
    const kg_crop_t *cd = &KG_CROPS[crop];
    tile->kind = KG_PLANT;
    tile->crop = (int8_t)crop;
    tile->animal = -1;
    tile->planted_day = (int16_t)day;
    tile->watered_today = 0;
    tile->consecutive_unwatered = 1;
    tile->yield_units = cd->ongoing ? 0 : 1;
    tile->max_lifespan_step = cd->ongoing
        ? -1 : (int16_t)((day + cd->max_yield_day + 1) * KG_TURNS_PER_DAY);
    tile->fertilized_until_day = -1;
    tile->fed_today = 0;
    tile->consecutive_unfed = 0;
    tile->cared_today = 0;
    tile->fertilizer_available = 0;
    tile->pending_care_bonus = 0;
}

static void new_animal(kg_tile_t *tile, int animal, int day) {
    tile->kind = (int8_t)KG_ANIMALS[animal].structure;
    tile->animal = (int8_t)animal;
    tile->crop = -1;
    tile->planted_day = (int16_t)day;
    tile->yield_units = 0;
    tile->consecutive_unfed = 0;
    tile->fed_today = 0;
    tile->cared_today = 0;
    tile->fertilizer_available = 0;
    tile->pending_care_bonus = 0;
    tile->watered_today = 0;
    tile->consecutive_unwatered = 0;
    tile->fertilized_until_day = -1;
    tile->max_lifespan_step = -1;
}

static void clear_tile(kg_tile_t *tile) {
    tile->kind = KG_EMPTY;
    tile->crop = -1;
    tile->animal = -1;
    tile->planted_day = 0;
    tile->watered_today = 0;
    tile->consecutive_unwatered = 0;
    tile->yield_units = 0;
    tile->fertilized_until_day = -1;
    tile->max_lifespan_step = -1;
    tile->fed_today = 0;
    tile->consecutive_unfed = 0;
    tile->cared_today = 0;
    tile->fertilizer_available = 0;
    tile->pending_care_bonus = 0;
}

static void apply_unit_action(kg_farm_t *farm, int unit, kg_unit_action_t action,
                              int day) {
    if (unit >= farm->num_units) return;
    int op = action.op;
    if (op < 0 || op >= KG_NUM_UNIT_OPS) return;

    int fx = farm->unit_x[unit];
    int fy = farm->unit_y[unit];

    if (op >= KG_OP_NORTH && op <= KG_OP_WEST) {
        int nx = fx + MOVE_DX[op];
        int ny = fy + MOVE_DY[op];
        if (nx < 0 || nx >= KG_BOARD_SIZE || ny < 0 || ny >= KG_BOARD_SIZE) return;
        farm->unit_x[unit] = (int8_t)nx;
        farm->unit_y[unit] = (int8_t)ny;
        return;
    }
    if (op == KG_OP_PASS) return;

    kg_tile_t *tile = &farm->tiles[fy * KG_BOARD_SIZE + fx];
    int has_animal = tile->animal >= 0;

    switch (op) {
    case KG_OP_DROP: {
        if (!is_shed_adjacent(fx, fy)) return;
        for (int item = 0; item < KG_NUM_ITEMS; item++) {
            int32_t n = farm->inv[unit][item];
            if (n <= 0) { farm->inv[unit][item] = 0; continue; }
            int32_t room = KG_SHED_CAPACITY - shed_total(farm);
            if (room < 0) room = 0;
            int32_t take = n < room ? n : room;
            if (take > 0) farm->shed[item] += take;
            farm->inv[unit][item] = 0;
        }
        return;
    }
    case KG_OP_PICKUP: {
        if (!is_shed_adjacent(fx, fy)) return;
        int item = action.arg;
        if (item < 0 || item >= KG_NUM_ITEMS) return;
        int32_t n = action.n;
        if (n <= 0) return;
        if (n > farm->shed[item]) n = farm->shed[item];
        if (n <= 0) return;
        farm->shed[item] -= n;
        farm->inv[unit][item] = (int16_t)(farm->inv[unit][item] + n);
        return;
    }
    case KG_OP_PLACE: {
        int item = action.arg;
        if (item < 0 || item >= KG_NUM_ITEMS) return;
        if (item >= KG_ANIMAL_ITEM_BASE) {
            int animal = item - KG_ANIMAL_ITEM_BASE;
            if (tile->kind == KG_ANIMALS[animal].structure && !has_animal) {
                if (inv_take(farm, unit, item, 1)) new_animal(tile, animal, day);
                return;
            }
        }
        if (is_shed_adjacent(fx, fy)) {
            int32_t n = action.n;
            if (n <= 0) return;
            if (n > farm->inv[unit][item]) n = farm->inv[unit][item];
            if (n <= 0) return;
            int32_t room = KG_SHED_CAPACITY - shed_total(farm);
            if (room < 0) room = 0;
            if (n > room) n = room;
            if (n <= 0) return;
            farm->inv[unit][item] = (int16_t)(farm->inv[unit][item] - n);
            farm->shed[item] += n;
        }
        return;
    }
    default:
        break;
    }

    if (tile->kind == KG_LOCKED) return;

    switch (op) {
    case KG_OP_PLANT: {
        int crop = action.arg;
        if (crop < 0 || crop >= KG_NUM_CROPS) return;
        if (tile->kind != KG_EMPTY) return;
        if (farm->seeds[crop] <= 0) return;
        farm->seeds[crop] -= 1;
        new_plant(tile, crop, day);
        note_decay(farm, tile->max_lifespan_step);
        return;
    }
    case KG_OP_WATER: {
        if (tile->kind != KG_PLANT) return;
        if (tile->watered_today) return;
        tile->watered_today = 1;
        const kg_crop_t *cd = &KG_CROPS[tile->crop];
        if (!cd->ongoing) {
            int age = day - tile->planted_day;
            int window = (cd->max_yield_day + 1) / 2;
            if (age >= window && age <= cd->max_yield_day) {
                int bonus = tile->fertilized_until_day >= day ? 2 : 1;
                int units = tile->yield_units + bonus;
                tile->yield_units = (int8_t)(units < cd->max_yield ? units : cd->max_yield);
            }
        }
        return;
    }
    case KG_OP_HARVEST: {
        if (tile->yield_units <= 0) return;
        if (tile->kind == KG_PLANT) {
            const kg_crop_t *cd = &KG_CROPS[tile->crop];
            if (day - tile->planted_day < cd->first_yield_day) return;
            int units = tile->yield_units;
            tile->yield_units = 0;
            farm->inv[unit][tile->crop] = (int16_t)(farm->inv[unit][tile->crop] + units);
            if (!cd->ongoing) clear_tile(tile);
        } else if (has_animal) {
            int units = tile->yield_units;
            tile->yield_units = 0;
            farm->inv[unit][KG_ANIMALS[tile->animal].product] = (int16_t)(farm->inv[unit][KG_ANIMALS[tile->animal].product] + units);
        }
        return;
    }
    case KG_OP_FERTILIZE: {
        if (tile->kind != KG_PLANT) return;
        if (!inv_take(farm, unit, KG_FERTILIZER, 1)) return;
        if (tile->fertilized_until_day < day + 2) {
            tile->fertilized_until_day = (int16_t)(day + 2);
        }
        return;
    }
    case KG_OP_DIG: {
        if (tile->kind == KG_EMPTY) return;
        if (has_animal) return;
        clear_tile(tile);
        return;
    }
    case KG_OP_BUILD_COOP: {
        if (tile->kind != KG_EMPTY) return;
        clear_tile(tile);
        tile->kind = KG_COOP;
        return;
    }
    case KG_OP_BUILD_PASTURE: {
        if (tile->kind != KG_EMPTY) return;
        clear_tile(tile);
        tile->kind = KG_PASTURE;
        return;
    }
    case KG_OP_FEED: {
        if (!has_animal) return;
        if (tile->fed_today) return;
        if (!inv_take(farm, unit, KG_WHEAT, 1)) return;
        tile->fed_today = 1;
        return;
    }
    case KG_OP_COLLECT_FERTILIZER: {
        if (!has_animal) return;
        if (!tile->fertilizer_available) return;
        tile->fertilizer_available = 0;
        farm->inv[unit][KG_FERTILIZER] = (int16_t)(farm->inv[unit][KG_FERTILIZER] + 1);
        return;
    }
    case KG_OP_CARE: {
        if (!has_animal) return;
        if (tile->cared_today) return;
        tile->cared_today = 1;
        return;
    }
    default:
        return;
    }
}

static void spawn_hand(kg_farm_t *farm, int8_t *out_x, int8_t *out_y) {
    int half = KG_BOARD_SIZE / 2;
    const int8_t ax[4] = {(int8_t)(half - 1), (int8_t)half, (int8_t)(half - 1), (int8_t)half};
    const int8_t ay[4] = {(int8_t)(half - 1), (int8_t)(half - 1), (int8_t)half, (int8_t)half};
    int occupants[4] = {0, 0, 0, 0};
    for (int u = 0; u < farm->num_units; u++) {
        for (int t = 0; t < 4; t++) {
            if (farm->unit_x[u] == ax[t] && farm->unit_y[u] == ay[t]) occupants[t]++;
        }
    }
    int best = 0;
    for (int t = 1; t < 4; t++) {
        if (occupants[t] < occupants[best]) best = t;
    }
    *out_x = ax[best];
    *out_y = ay[best];
}

static void do_hire(kg_farm_t *farm) {
    int64_t cost = kg_hire_cost(farm->hires_today);
    if (farm->money < (float)cost) return;
    if (farm->num_units >= KG_MAX_UNITS) return;
    farm->money -= (float)cost;
    farm->hires_today += 1;
    int unit = farm->num_units;
    spawn_hand(farm, &farm->unit_x[unit], &farm->unit_y[unit]);
    for (int i = 0; i < KG_NUM_ITEMS; i++) farm->inv[unit][i] = 0;
    farm->num_units += 1;
}

static void do_buy_land(kg_farm_t *farm) {
    if (farm->quadrants >= 3) return;
    int32_t cost = KG_LAND_PRICES[farm->quadrants];
    if (farm->money < (float)cost) return;
    farm->money -= (float)cost;
    static const int ORDER[3] = {1, 2, 3};
    int quadrant = ORDER[farm->quadrants];
    farm->quadrants += 1;
    for (int y = 0; y < KG_BOARD_SIZE; y++) {
        for (int x = 0; x < KG_BOARD_SIZE; x++) {
            kg_tile_t *tile = &farm->tiles[y * KG_BOARD_SIZE + x];
            if (quadrant_of(x, y) == quadrant && tile->kind == KG_LOCKED) {
                clear_tile(tile);
            }
        }
    }
}

static int commit_unit(int op, int item, int32_t price, kg_farm_t *farm,
                       kg_state_t *state) {
    switch (op) {
    case KG_MK_SELL:
        if (farm->shed[item] <= 0) return 0;
        farm->shed[item] -= 1;
        farm->money += (float)price;
        if (price > 1) market_add(state, item, 1);
        return 1;
    case KG_MK_BUY_PRODUCT:
        if (farm->money < (float)price) return 0;
        if (shed_total(farm) >= KG_SHED_CAPACITY) return 0;
        farm->money -= (float)price;
        farm->shed[item] += 1;
        market_add(state, item, -1);
        return 1;
    case KG_MK_BUY_SEED:
        if (farm->money < (float)price) return 0;
        farm->money -= (float)price;
        farm->seeds[item] += 1;
        return 1;
    case KG_MK_BUY_ANIMAL:
        if (farm->money < (float)price) return 0;
        if (shed_total(farm) >= KG_SHED_CAPACITY) return 0;
        farm->money -= (float)price;
        farm->shed[item] += 1;
        return 1;
    default:
        return 0;
    }
}

static void process_market(kg_state_t *state, const kg_action_t actions[KG_PLAYERS]) {
    int32_t remaining[KG_PLAYERS];
    int active[KG_PLAYERS];

    for (int slot = 0; slot < KG_MAX_MARKET_ORDERS; slot++) {
        for (int p = 0; p < KG_PLAYERS; p++) {
            const kg_market_action_t *order = &actions[p].orders[slot];
            active[p] = order->op > KG_MK_NONE && order->op < KG_NUM_MARKET_OPS;
            remaining[p] = order->n;
            if (order->op == KG_MK_HIRE) {
                do_hire(&state->farms[p]);
                active[p] = 0;
            } else if (order->op == KG_MK_BUY_LAND) {
                do_buy_land(&state->farms[p]);
                active[p] = 0;
            }
        }

        while (1) {
            int quoted_op[KG_PLAYERS] = {-1, -1};
            int quoted_item[KG_PLAYERS] = {0, 0};
            int32_t quoted_price[KG_PLAYERS] = {0, 0};
            int any = 0;

            for (int p = 0; p < KG_PLAYERS; p++) {
                if (!active[p] || remaining[p] <= 0) continue;
                const kg_market_action_t *order = &actions[p].orders[slot];
                int op = order->op;
                int item = order->item;
                if (op == KG_MK_SELL && item >= 0 && item < KG_NUM_PRODUCTS) {
                    quoted_price[p] = (int32_t)kg_market_price(item, state->market[item]);
                } else if (op == KG_MK_BUY_PRODUCT &&
                           (item == KG_WHEAT || item == KG_FERTILIZER)) {
                    quoted_price[p] = (int32_t)kg_market_price(item, state->market[item] - 1);
                } else if (op == KG_MK_BUY_SEED && item >= 0 && item < KG_NUM_CROPS) {
                    quoted_price[p] = KG_CROPS[item].seed;
                } else if (op == KG_MK_BUY_ANIMAL && item >= KG_ANIMAL_ITEM_BASE &&
                           item < KG_NUM_ITEMS) {
                    quoted_price[p] = KG_ANIMALS[item - KG_ANIMAL_ITEM_BASE].cost;
                } else {
                    active[p] = 0;
                    continue;
                }
                quoted_op[p] = op;
                quoted_item[p] = item;
                any = 1;
            }
            if (!any) break;

            int committed = 0;
            for (int p = 0; p < KG_PLAYERS; p++) {
                if (quoted_op[p] < 0) continue;
                if (commit_unit(quoted_op[p], quoted_item[p], quoted_price[p],
                                &state->farms[p], state)) {
                    remaining[p] -= 1;
                    committed = 1;
                } else {
                    active[p] = 0;
                }
            }
            if (!committed) break;
        }
    }
}

static void town_consume(kg_state_t *state, int step) {
    if (step % KG_TOWN_SHOP_SELL_INTERVAL == 0) {
        for (int s = 0; s < KG_NUM_SHOPS; s++) {
            int copies = state->shops[s];
            if (copies <= 0) continue;
            for (int i = 0; i < KG_NUM_PRODUCTS; i++) {
                if (KG_SHOP_DEMAND[s][i]) {
                    market_add(state, i, -KG_SHOP_DEMAND[s][i] * copies);
                }
            }
        }
    }
    if (step % KG_TOWN_CENTER_SELL_INTERVAL == 0) {
        for (int i = 0; i < KG_NUM_PRODUCTS; i++) {
            if (i == KG_FERTILIZER) continue;
            market_add(state, i, -1);
        }
    }
}

static void decay_plants(kg_farm_t *farm, int step) {
    if (step & 1) return;
    if (step < farm->next_decay) return;
    int32_t earliest = INT32_MAX;
    for (int t = 0; t < KG_TILES; t++) {
        kg_tile_t *tile = &farm->tiles[t];
        if (tile->kind != KG_PLANT) continue;
        int32_t mls = tile->max_lifespan_step;
        if (mls < 0) continue;
        if (step < mls) {
            if (mls < earliest) earliest = mls;
            continue;
        }
        tile->yield_units -= 1;
        if (tile->yield_units <= 0) {
            clear_tile(tile);
            tile->kind = KG_WEED;
        } else if (mls < earliest) {
            earliest = mls;
        }
    }
    farm->next_decay = earliest;
}

static void daily_refresh_plants(kg_farm_t *farm, int day) {
    int next_day = day + 1;
    for (int t = 0; t < KG_TILES; t++) {
        kg_tile_t *tile = &farm->tiles[t];
        if (tile->kind != KG_PLANT) continue;
        int was_watered = tile->watered_today;
        tile->consecutive_unwatered = was_watered
            ? 0 : (int8_t)(tile->consecutive_unwatered + 1);
        tile->watered_today = 0;
        if (tile->consecutive_unwatered >= 2) {
            clear_tile(tile);
            tile->kind = KG_WEED;
            continue;
        }
        const kg_crop_t *cd = &KG_CROPS[tile->crop];
        if (!cd->ongoing) continue;
        int since = next_day - tile->planted_day - cd->first_yield_day;
        if (since < 0) continue;
        if (cd->interval <= 0 || since % cd->interval != 0) continue;
        int count = since / cd->interval + 1;
        if (count > cd->max_yield) continue;
        int fertilized = was_watered && tile->fertilized_until_day >= day;
        int units = tile->yield_units + (fertilized ? 2 : 1);
        tile->yield_units = (int8_t)(units < cd->max_yield ? units : cd->max_yield);
        if (count == cd->max_yield) {
            tile->max_lifespan_step = (int16_t)((next_day + 1) * KG_TURNS_PER_DAY);
            note_decay(farm, tile->max_lifespan_step);
        }
    }
}

static void daily_refresh_animals(kg_farm_t *farm, int day) {
    int next_day = day + 1;
    for (int t = 0; t < KG_TILES; t++) {
        kg_tile_t *tile = &farm->tiles[t];
        if (tile->animal < 0) continue;
        if (tile->fed_today) tile->consecutive_unfed = 0;
        else tile->consecutive_unfed = (int8_t)(tile->consecutive_unfed + 1);
        if (tile->consecutive_unfed >= 2) {
            int structure = KG_ANIMALS[tile->animal].structure;
            clear_tile(tile);
            tile->kind = (int8_t)structure;
            continue;
        }
        const kg_animal_t *a = &KG_ANIMALS[tile->animal];
        int since = next_day - tile->planted_day - a->first_yield_day;
        if (since >= 0 && a->interval > 0 && since % a->interval == 0) {
            int bonus = tile->fed_today ? tile->pending_care_bonus : 0;
            int units = tile->yield_units + 1 + bonus;
            tile->yield_units = (int8_t)(units < a->max_held ? units : a->max_held);
            tile->pending_care_bonus = 0;
        }
        if (tile->cared_today && tile->fed_today) {
            tile->pending_care_bonus = (int8_t)(tile->pending_care_bonus + 1);
        }
        tile->fertilizer_available = 1;
        tile->fed_today = 0;
        tile->cared_today = 0;
    }
}

static void drop_inventories(kg_farm_t *farm) {
    for (int u = 0; u < farm->num_units; u++) {
        for (int item = 0; item < KG_NUM_ITEMS; item++) {
            int32_t n = farm->inv[u][item];
            if (n <= 0) { farm->inv[u][item] = 0; continue; }
            int32_t room = KG_SHED_CAPACITY - shed_total(farm);
            if (room < 0) room = 0;
            int32_t take = n < room ? n : room;
            if (take > 0) farm->shed[item] += take;
            farm->inv[u][item] = 0;
        }
    }
}

static uint32_t next_random(kg_state_t *state) {
    state->rng ^= state->rng << 13;
    state->rng ^= state->rng >> 7;
    state->rng ^= state->rng << 17;
    return (uint32_t)(state->rng >> 32);
}

static void spawn_weeds(kg_state_t *state, kg_farm_t *farm) {
    if (state->weed_chance <= 0.0f) return;
    for (int t = 0; t < KG_TILES; t++) {
        if (farm->tiles[t].kind != KG_EMPTY) continue;
        float u = (float)(next_random(state) / 4294967296.0);
        if (u < state->weed_chance) {
            clear_tile(&farm->tiles[t]);
            farm->tiles[t].kind = KG_WEED;
        }
    }
}

static void end_of_day(kg_state_t *state, int day) {
    int half = KG_BOARD_SIZE / 2;
    for (int p = 0; p < KG_PLAYERS; p++) {
        kg_farm_t *farm = &state->farms[p];
        daily_refresh_plants(farm, day);
        daily_refresh_animals(farm, day);
        spawn_weeds(state, farm);
        drop_inventories(farm);
        farm->num_units = 1;
        for (int u = 0; u < KG_MAX_UNITS; u++) {
            farm->unit_x[u] = (int8_t)(half - 1);
            farm->unit_y[u] = (int8_t)(half - 1);
            for (int i = 0; i < KG_NUM_ITEMS; i++) farm->inv[u][i] = 0;
        }
        farm->hires_today = 0;
    }

    int next_day = day + 1;
    if (next_day > 0 && next_day % KG_TOWN_SHOP_UNLOCK_INTERVAL == 0 &&
        state->shops_unlocked < KG_MAX_SHOP_INSTANCES) {
        state->shops[state->shop_order[state->shops_unlocked]] += 1;
        state->shops_unlocked += 1;
    }
}

void kg_step(kg_state_t *state, const kg_action_t actions[KG_PLAYERS]) {
    if (state->done) return;
    int step = state->step;
    int day = step / KG_TURNS_PER_DAY;

    for (int p = 0; p < KG_PLAYERS; p++) {
        kg_farm_t *farm = &state->farms[p];
        int32_t demand[KG_NUM_CROPS] = {0};
        for (int u = 0; u < farm->num_units; u++) {
            const kg_unit_action_t *a = &actions[p].units[u];
            if (a->op == KG_OP_PLANT && a->arg >= 0 && a->arg < KG_NUM_CROPS) {
                demand[a->arg] += 1;
            }
        }
        int blocked[KG_NUM_CROPS];
        for (int c = 0; c < KG_NUM_CROPS; c++) blocked[c] = demand[c] > farm->seeds[c];
        for (int u = 0; u < farm->num_units; u++) {
            kg_unit_action_t a = actions[p].units[u];
            if (a.op == KG_OP_PLANT && a.arg >= 0 && a.arg < KG_NUM_CROPS &&
                blocked[a.arg]) {
                a.op = KG_OP_PASS;
            }
            apply_unit_action(farm, u, a, day);
        }
    }

    process_market(state, actions);
    town_consume(state, step);
    for (int p = 0; p < KG_PLAYERS; p++) decay_plants(&state->farms[p], step);
    if ((step + 1) % KG_TURNS_PER_DAY == 0) end_of_day(state, day);

    refresh_prices(state);

    state->step = step + 1;
    int32_t horizon = state->episode_steps > 0 ? state->episode_steps : KG_EPISODE_STEPS;
    if (step >= horizon - 2) {
        state->done = 1;
        for (int p = 0; p < KG_PLAYERS; p++) state->reward[p] = state->farms[p].money;
    }
}

void kg_batch_reset(kg_state_t *states, int n, uint64_t seed) {
    for (int e = 0; e < n; e++) kg_reset(&states[e], seed + (uint64_t)e * 2654435761u);
}

void kg_batch_step(kg_state_t *states, const kg_action_t *actions, int n, int threads) {
    (void)threads;
    #pragma omp parallel for schedule(static) if (threads > 1)
    for (int e = 0; e < n; e++) {
        kg_state_t *s = &states[e];
        if (s->done) {
            kg_reset(s, s->rng);
        }
        kg_step(s, &actions[(size_t)e * KG_PLAYERS]);
    }
}

void kg_batch_encode(const kg_state_t *states, int n, kg_obs_t obs, int threads) {
    (void)threads;
    #pragma omp parallel for schedule(static) if (threads > 1)
    for (int e = 0; e < n; e++) {
        const kg_state_t *s = &states[e];
        obs.day[e] = (int8_t)(s->step / KG_TURNS_PER_DAY);
        obs.hour[e] = (int8_t)(s->step % KG_TURNS_PER_DAY);
        for (int i = 0; i < KG_NUM_PRODUCTS; i++) {
            obs.market[(size_t)e * KG_NUM_PRODUCTS + i] = s->market[i];
            obs.prices[(size_t)e * KG_NUM_PRODUCTS + i] = (int16_t)s->prices[i];
        }
        for (int i = 0; i < KG_NUM_SHOPS; i++) {
            obs.shops[(size_t)e * KG_NUM_SHOPS + i] = s->shops[i];
        }
        for (int p = 0; p < KG_PLAYERS; p++) {
            const kg_farm_t *farm = &s->farms[p];
            size_t base = ((size_t)e * KG_PLAYERS + p);
            obs.money[base] = farm->money;
            obs.hands[base] = (int16_t)(farm->num_units - 1);
            obs.hires[base] = (int16_t)farm->hires_today;
            obs.quadrants[base] = (int16_t)farm->quadrants;
            for (int i = 0; i < KG_NUM_ITEMS; i++) {
                obs.shed[base * KG_NUM_ITEMS + i] = (int16_t)farm->shed[i];
            }
            for (int i = 0; i < KG_NUM_CROPS; i++) {
                obs.seeds[base * KG_NUM_CROPS + i] = (int16_t)farm->seeds[i];
            }
            for (int u = 0; u < KG_OBS_UNITS; u++) {
                int live = u < farm->num_units;
                obs.pos[(base * KG_OBS_UNITS + u) * 2 + 0] = farm->unit_x[u];
                obs.pos[(base * KG_OBS_UNITS + u) * 2 + 1] = farm->unit_y[u];
                for (int i = 0; i < KG_NUM_ITEMS; i++) {
                    obs.inv[(base * KG_OBS_UNITS + u) * KG_NUM_ITEMS + i] =
                        live ? farm->inv[u][i] : 0;
                }
            }
            for (int t = 0; t < KG_TILES; t++) {
                const kg_tile_t *tile = &farm->tiles[t];
                int8_t *dst = &obs.tiles[(base * KG_TILES + t) * KG_OBS_TILE_FIELDS];
                dst[0] = tile->kind;
                dst[1] = tile->crop;
                dst[2] = tile->animal;
                dst[3] = (int8_t)tile->planted_day;
                dst[4] = tile->watered_today;
                dst[5] = tile->consecutive_unwatered;
                dst[6] = tile->yield_units;
                dst[7] = (int8_t)tile->fertilized_until_day;
                dst[8] = tile->fed_today;
                dst[9] = tile->consecutive_unfed;
                dst[10] = tile->cared_today;
                dst[11] = tile->fertilizer_available;
                dst[12] = tile->pending_care_bonus;
                obs.lifespan[base * KG_TILES + t] = tile->max_lifespan_step < 0
                    ? (int8_t)-1 : (int8_t)(tile->max_lifespan_step / KG_TURNS_PER_DAY);
            }
            obs.reward[base] = s->done ? s->farms[p].money : 0.0f;
            obs.terminated[base] = (int8_t)s->done;
        }
    }
}

#include <stdlib.h>

static void decode_heads(const int32_t *heads, kg_action_t *action);

const int32_t KG_QUANTITIES[KG_NUM_QUANTITIES] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 20, 24, 32, 40, 50, 64, 100,
};
const int8_t KG_JOB_OP[KG_NUM_JOBS] = {
    KG_OP_PASS,
    KG_OP_PLANT, KG_OP_PLANT, KG_OP_PLANT, KG_OP_PLANT, KG_OP_PLANT,
    KG_OP_WATER, KG_OP_HARVEST, KG_OP_FERTILIZE, KG_OP_DIG,
    KG_OP_BUILD_COOP, KG_OP_BUILD_PASTURE,
    KG_OP_PLACE, KG_OP_PLACE, KG_OP_PLACE,
    KG_OP_FEED, KG_OP_COLLECT_FERTILIZER, KG_OP_CARE,
    KG_OP_PICKUP, KG_OP_PICKUP, KG_OP_PICKUP, KG_OP_PICKUP, KG_OP_PICKUP,
};

const int8_t KG_JOB_ARG[KG_NUM_JOBS] = {
    0,
    0, 1, 2, 3, 4,
    0, 0, 0, 0,
    0, 0,
    KG_ANIMAL_ITEM_BASE, KG_ANIMAL_ITEM_BASE + 1, KG_ANIMAL_ITEM_BASE + 2,
    0, 0, 0,
    KG_WHEAT, KG_FERTILIZER,
    KG_ANIMAL_ITEM_BASE, KG_ANIMAL_ITEM_BASE + 1, KG_ANIMAL_ITEM_BASE + 2,
};

void kg_legal_jobs(const kg_state_t *state, int player, int8_t *out) {
    const kg_farm_t *farm = &state->farms[player];
    int day = state->step / KG_TURNS_PER_DAY;
    int32_t carried[KG_NUM_ITEMS];
    for (int i = 0; i < KG_NUM_ITEMS; i++) carried[i] = 0;
    for (int u = 0; u < farm->num_units; u++) {
        for (int i = 0; i < KG_NUM_ITEMS; i++) carried[i] += farm->inv[u][i];
    }

    for (int t = 0; t < KG_TILES; t++) {
        const kg_tile_t *tile = &farm->tiles[t];
        int8_t *row = out + (size_t)t * KG_NUM_JOBS;
        for (int j = 0; j < KG_NUM_JOBS; j++) row[j] = 0;

        int shed_tile = is_shed_adjacent(t % KG_BOARD_SIZE, t / KG_BOARD_SIZE);
        int animal = tile->animal >= 0;
        row[0] = 1;

        for (int j = 18; j < KG_NUM_JOBS; j++) {
            row[j] = shed_tile && farm->shed[KG_JOB_ARG[j]] > 0;
        }
        if (tile->kind == KG_LOCKED) continue;

        for (int c = 0; c < KG_NUM_CROPS; c++) {
            row[1 + c] = tile->kind == KG_EMPTY && farm->seeds[c] > 0;
        }
        row[6] = tile->kind == KG_PLANT && !tile->watered_today;
        row[7] = tile->yield_units > 0
            && (animal || (tile->kind == KG_PLANT
                           && day - tile->planted_day
                              >= KG_CROPS[tile->crop].first_yield_day));
        row[8] = tile->kind == KG_PLANT
            && tile->fertilized_until_day < day + 2
            && carried[KG_FERTILIZER] > 0;
        row[9] = tile->kind != KG_EMPTY && !animal;
        row[10] = tile->kind == KG_EMPTY;
        row[11] = tile->kind == KG_EMPTY;
        for (int a = 0; a < KG_NUM_ANIMALS; a++) {
            row[12 + a] = !animal && tile->kind == KG_ANIMALS[a].structure
                && farm->shed[KG_ANIMAL_ITEM_BASE + a] > 0;
        }
        row[15] = animal && !tile->fed_today && carried[KG_WHEAT] > 0;
        row[16] = animal && tile->fertilizer_available;
        row[17] = animal && !tile->cared_today;
    }
}

#define KG_NUM_MARKET_ACTIONS 25

static const int8_t KG_MARKET_ACTION_OP[KG_NUM_MARKET_ACTIONS] = {
    KG_MK_NONE,
    KG_MK_BUY_SEED, KG_MK_BUY_SEED, KG_MK_BUY_SEED, KG_MK_BUY_SEED, KG_MK_BUY_SEED,
    KG_MK_BUY_PRODUCT, KG_MK_BUY_PRODUCT,
    KG_MK_BUY_ANIMAL, KG_MK_BUY_ANIMAL, KG_MK_BUY_ANIMAL,
    KG_MK_SELL, KG_MK_SELL, KG_MK_SELL, KG_MK_SELL, KG_MK_SELL, KG_MK_SELL,
    KG_MK_SELL, KG_MK_SELL, KG_MK_SELL, KG_MK_SELL, KG_MK_SELL, KG_MK_SELL,
    KG_MK_HIRE, KG_MK_BUY_LAND,
};
static const int8_t KG_MARKET_ACTION_ITEM[KG_NUM_MARKET_ACTIONS] = {
    0, 0, 1, 2, 3, 4, KG_WHEAT, KG_FERTILIZER,
    KG_ANIMAL_ITEM_BASE, KG_ANIMAL_ITEM_BASE + 1, KG_ANIMAL_ITEM_BASE + 2,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0,
};

static inline int32_t quantity_of(int32_t index) {
    if (index < 0) index = 0;
    if (index >= KG_NUM_QUANTITIES) index = KG_NUM_QUANTITIES - 1;
    return KG_QUANTITIES[index];
}

void kg_decode_heads(const int32_t *heads, kg_action_t actions[KG_PLAYERS]) {
    for (int p = 0; p < KG_PLAYERS; p++)
        decode_heads(heads + (size_t)p * KG_NUM_HEADS, &actions[p]);
}

static void decode_heads(const int32_t *heads, kg_action_t *action) {
    const int units = KG_OBS_UNITS, orders = KG_MAX_MARKET_ORDERS;
    for (int u = 0; u < KG_MAX_UNITS; u++) {
        action->units[u].op = KG_OP_PASS;
        action->units[u].arg = 0;
        action->units[u].n = 1;
    }
    for (int u = 0; u < units; u++) {
        int32_t op = heads[u];
        action->units[u].op = (int8_t)(op < 0 ? 0 : (op >= KG_NUM_UNIT_OPS
                                                     ? KG_NUM_UNIT_OPS - 1 : op));
        action->units[u].arg = (int8_t)heads[units + u];
        action->units[u].n = (int16_t)quantity_of(heads[2 * units + u]);
    }
    const int base = 3 * units;
    for (int o = 0; o < orders; o++) {
        int32_t index = heads[base + o];
        if (index < 0) index = 0;
        if (index >= KG_NUM_MARKET_ACTIONS) index = KG_NUM_MARKET_ACTIONS - 1;
        action->orders[o].op = KG_MARKET_ACTION_OP[index];
        action->orders[o].item = KG_MARKET_ACTION_ITEM[index];
        action->orders[o].n = (int16_t)quantity_of(heads[base + orders + o]);
    }
}

kg_vec_t *kg_vec_create(int n_envs, uint64_t seed, int threads, int episode_steps) {
    kg_vec_t *vec = (kg_vec_t *)calloc(1, sizeof(kg_vec_t));
    vec->n_envs = n_envs;
    vec->threads = threads > 0 ? threads : 1;
    vec->states = (kg_state_t *)calloc((size_t)n_envs, sizeof(kg_state_t));
    vec->actions = (kg_action_t *)calloc((size_t)n_envs * KG_PLAYERS, sizeof(kg_action_t));
    for (int e = 0; e < n_envs; e++) vec->states[e].episode_steps = episode_steps;
    kg_batch_reset(vec->states, n_envs, seed);
    return vec;
}

void kg_vec_free(kg_vec_t *vec) {
    if (!vec) return;
    free(vec->states);
    free(vec->actions);
    free(vec);
}

void kg_vec_bind(kg_vec_t *vec, kg_obs_t obs, const int32_t *heads) {
    vec->obs = obs;
    vec->heads = heads;
}

void kg_vec_reset(kg_vec_t *vec, uint64_t seed) {
    kg_batch_reset(vec->states, vec->n_envs, seed);
    kg_batch_encode(vec->states, vec->n_envs, vec->obs, vec->threads);
}

void kg_vec_step(kg_vec_t *vec) {
    const int n = vec->n_envs;
    #pragma omp parallel for schedule(static) if (vec->threads > 1)
    for (int e = 0; e < n; e++) {
        kg_state_t *s = &vec->states[e];
        if (s->done) {
            kg_reset(s, s->rng);
        }
        for (int p = 0; p < KG_PLAYERS; p++) {
            const int32_t *heads = vec->heads +
                ((size_t)e * KG_PLAYERS + p) * KG_NUM_HEADS;
            decode_heads(heads, &vec->actions[(size_t)e * KG_PLAYERS + p]);
        }
        kg_step(s, &vec->actions[(size_t)e * KG_PLAYERS]);
    }
    kg_batch_encode(vec->states, n, vec->obs, vec->threads);
}
