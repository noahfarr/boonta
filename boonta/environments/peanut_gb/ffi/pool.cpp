#include <algorithm>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

extern "C" {
#define ENABLE_SOUND 0
#define ENABLE_LCD 1
#define PEANUT_GB_HEADER_ONLY
#include "vendor/peanut_gb.h"
uint8_t __gb_read(struct gb_s* gb, uint16_t addr);
extern const uint8_t* peanut_rom;
}

namespace {

constexpr int kScreenW = LCD_WIDTH;
constexpr int kScreenH = LCD_HEIGHT;
constexpr int kObsH = kScreenH / 2;
constexpr int kObsW = kScreenW / 2;
constexpr int kRamSize = 8192;
constexpr int kActions = 9;
constexpr size_t kCartRam = 32768;

struct Pool;

struct Slot {
    gb_s gb;
    uint8_t cram[kCartRam];
    uint8_t frame[kScreenH * kScreenW];
    Pool* pool = nullptr;
    std::vector<uint8_t> last;
    bool primed = false;
    int errors = 0;
    std::jmp_buf escape;
};

std::vector<uint8_t> rom_store;

struct Pool {
    std::vector<Slot*> slots;
    int frame_skip = 24;
    int hold = 8;
    int num_threads = 1;
    size_t cart_ram = 0;
    size_t state_size = 0;
};

Slot* owner(gb_s* gb) {
    return static_cast<Slot*>(gb->direct.priv);
}

uint8_t rom_read(gb_s* gb, const uint_fast32_t addr) {
    (void)gb;
    return peanut_rom[addr];
}

uint8_t cram_read(gb_s* gb, const uint_fast32_t addr) {
    return owner(gb)->cram[addr];
}

void cram_write(gb_s* gb, const uint_fast32_t addr, const uint8_t value) {
    owner(gb)->cram[addr] = value;
}

void on_error(gb_s* gb, const enum gb_error_e error, const uint16_t) {
    Slot* slot = owner(gb);
    slot->errors++;
    if (error == GB_INVALID_OPCODE) std::longjmp(slot->escape, 1);
}

void draw_line(gb_s* gb, const uint8_t* pixels, const uint_fast8_t line) {
    std::memcpy(owner(gb)->frame + line * kScreenW, pixels, kScreenW);
}

void wire(Slot& slot) {
    slot.gb.gb_rom_read = rom_read;
    slot.gb.gb_cart_ram_read = cram_read;
    slot.gb.gb_cart_ram_write = cram_write;
    slot.gb.gb_error = on_error;
    slot.gb.direct.priv = &slot;
    slot.gb.display.lcd_draw_line = nullptr;
}

uint8_t joypad(int action) {
    uint8_t held = 0xFF;
    switch (action) {
        case 1: held &= ~JOYPAD_UP; break;
        case 2: held &= ~JOYPAD_DOWN; break;
        case 3: held &= ~JOYPAD_LEFT; break;
        case 4: held &= ~JOYPAD_RIGHT; break;
        case 5: held &= ~JOYPAD_A; break;
        case 6: held &= ~JOYPAD_B; break;
        case 7: held &= ~JOYPAD_START; break;
        case 8: held &= ~JOYPAD_SELECT; break;
        default: break;
    }
    return held;
}

bool boot(Pool* pool, Slot& slot) {
    std::memset(&slot.gb, 0, sizeof(slot.gb));
    std::memset(slot.cram, 0, sizeof(slot.cram));
    std::memset(slot.frame, 0, sizeof(slot.frame));
    slot.pool = pool;
    slot.primed = false;
    slot.errors = 0;
    wire(slot);
    if (gb_init(&slot.gb, rom_read, cram_read, cram_write, on_error, &slot) != GB_INIT_NO_ERROR) return false;
    gb_init_lcd(&slot.gb, draw_line);
    slot.gb.display.lcd_draw_line = nullptr;
    return true;
}

void run(Slot& slot, int frames, bool render_last) {
    for (int i = 0; i < frames; i++) {
        slot.gb.display.lcd_draw_line = (render_last && i == frames - 1) ? draw_line : nullptr;
        if (setjmp(slot.escape) == 0) gb_run_frame(&slot.gb);
        else return;
    }
}

void press(Slot& slot, int action, int hold, int total) {
    hold = std::min(hold, total);
    slot.gb.direct.joypad = joypad(action);
    run(slot, hold, total == hold);
    slot.gb.direct.joypad = 0xFF;
    if (total > hold) run(slot, total - hold, true);
    slot.gb.display.lcd_draw_line = nullptr;
}

constexpr uint16_t kBattle = 0xD057;
constexpr uint16_t kWalk = 0xCFC5;
constexpr int kBrawl = 192;
constexpr int kStride = 120;

bool fighting(Slot& slot) {
    return __gb_read(&slot.gb, kBattle) != 0;
}

void stride(Pool* pool, Slot& slot, int action) {
    if (fighting(slot)) {
        press(slot, action, pool->hold, kBrawl);
        return;
    }
    press(slot, action, pool->hold, pool->frame_skip);
    for (int i = 0; i < kStride && __gb_read(&slot.gb, kWalk) != 0; i++) run(slot, 1, false);
    run(slot, 1, true);
    slot.gb.display.lcd_draw_line = nullptr;
}

uint8_t shade(uint8_t pixel) {
    return (uint8_t)(255 - 85 * (pixel & LCD_COLOUR));
}

void picture(const Slot& slot, uint8_t* frame) {
    for (int y = 0; y < kObsH; y++) {
        for (int x = 0; x < kObsW; x++) {
            int total = 0;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                    total += shade(slot.frame[(2 * y + dy) * kScreenW + 2 * x + dx]);
            frame[y * kObsW + x] = (uint8_t)(total / 4);
        }
    }
}

void paint(const Slot& slot, uint8_t* rgb) {
    for (int i = 0; i < kScreenW * kScreenH; i++) {
        uint8_t value = shade(slot.frame[i]);
        rgb[3 * i] = value;
        rgb[3 * i + 1] = value;
        rgb[3 * i + 2] = value;
    }
}

void observe(const Slot& slot, uint8_t* ram) {
    std::memcpy(ram, slot.gb.wram, kRamSize);
}

void stash(Pool* pool, Slot& slot, uint8_t* out) {
    std::memcpy(out, &slot.gb, sizeof(gb_s));
    gb_s* view = reinterpret_cast<gb_s*>(out);
    view->gb_rom_read = nullptr;
    view->gb_cart_ram_read = nullptr;
    view->gb_cart_ram_write = nullptr;
    view->gb_error = nullptr;
    view->gb_serial_tx = nullptr;
    view->gb_serial_rx = nullptr;
    view->gb_bootrom_read = nullptr;
    view->display.lcd_draw_line = nullptr;
    view->direct.priv = nullptr;
    std::memcpy(out + sizeof(gb_s), slot.cram, pool->cart_ram);
    std::memcpy(slot.last.data(), out, pool->state_size);
    slot.primed = true;
}

void serve(Pool* pool, Slot& slot, const uint8_t* in) {
    if (slot.primed && std::memcmp(in, slot.last.data(), pool->state_size) == 0) return;
    std::memcpy(&slot.gb, in, sizeof(gb_s));
    std::memcpy(slot.cram, in + sizeof(gb_s), pool->cart_ram);
    wire(slot);
    slot.primed = false;
}

}  // namespace

extern "C" {

void* pokemon_pool_create(const char* rom_path, int num_slots, int frame_skip, int hold,
                          int num_threads, int seed) {
    (void)seed;
    FILE* file = std::fopen(rom_path, "rb");
    if (file == nullptr) return nullptr;
    std::fseek(file, 0, SEEK_END);
    long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    std::vector<uint8_t> rom(size > 0 ? (size_t)size : 0);
    size_t got = rom.empty() ? 0 : std::fread(rom.data(), 1, rom.size(), file);
    std::fclose(file);
    if (got != rom.size() || rom.size() < 0x150) return nullptr;
    if (rom_store.empty()) {
        rom_store = std::move(rom);
        peanut_rom = rom_store.data();
    } else if (rom_store != rom) {
        return nullptr;
    }
    auto* pool = new Pool();
    pool->frame_skip = frame_skip > 0 ? frame_skip : 1;
    pool->hold = std::max(0, std::min(hold, pool->frame_skip));
    pool->num_threads = num_threads > 0 ? num_threads : 1;
    pool->slots.resize(num_slots);
    for (int i = 0; i < num_slots; i++) {
        pool->slots[i] = new Slot();
        if (!boot(pool, *pool->slots[i])) {
            for (Slot* slot : pool->slots) delete slot;
            delete pool;
            return nullptr;
        }
    }
    size_t cart_ram = 0;
    if (num_slots > 0 && gb_get_save_size_s(&pool->slots[0]->gb, &cart_ram) != 0) cart_ram = kCartRam;
    pool->cart_ram = std::min(cart_ram, kCartRam);
    pool->state_size = sizeof(gb_s) + pool->cart_ram;
    for (Slot* slot : pool->slots) slot->last.assign(pool->state_size, 0);
    return pool;
}

int64_t pokemon_pool_state_size(void* handle) {
    return (int64_t) static_cast<Pool*>(handle)->state_size;
}

int pokemon_pool_num_actions(void* handle) {
    (void)handle;
    return kActions;
}

int pokemon_pool_obs_height(void* handle) {
    (void)handle;
    return kObsH;
}

int pokemon_pool_obs_width(void* handle) {
    (void)handle;
    return kObsW;
}

int pokemon_pool_advance(void* handle, int count, const uint8_t* states, const int32_t* actions,
                         uint8_t* states_out, uint8_t* frames, uint8_t* ram) {
    auto* pool = static_cast<Pool*>(handle);
    const size_t ss = pool->state_size;
    const size_t fs = (size_t)kObsH * kObsW;
    const int width = (int)pool->slots.size();
    for (int base = 0; base < count; base += width) {
        const int n = std::min(width, count - base);
#pragma omp parallel for schedule(dynamic, 1) num_threads(pool->num_threads)
        for (int i = 0; i < n; i++) {
            const size_t j = (size_t)(base + i);
            Slot& slot = *pool->slots[i];
            serve(pool, slot, states + j * ss);
            stride(pool, slot, actions[j]);
            picture(slot, frames + j * fs);
            observe(slot, ram + j * kRamSize);
            stash(pool, slot, states_out + j * ss);
        }
    }
    return 0;
}

int pokemon_pool_press(void* handle, int slot_index, int action, int hold, int total) {
    auto* pool = static_cast<Pool*>(handle);
    Slot& slot = *pool->slots[slot_index];
    press(slot, action, hold, total);
    slot.primed = false;
    return 0;
}

int pokemon_pool_read(void* handle, int slot_index, int address) {
    auto* pool = static_cast<Pool*>(handle);
    return __gb_read(&pool->slots[slot_index]->gb, (uint16_t)address);
}

int pokemon_pool_screen(void* handle, int slot_index, uint8_t* rgb) {
    auto* pool = static_cast<Pool*>(handle);
    paint(*pool->slots[slot_index], rgb);
    return 0;
}

int pokemon_pool_render(void* handle, int count, const uint8_t* states, uint8_t* rgb) {
    auto* pool = static_cast<Pool*>(handle);
    const size_t ss = pool->state_size;
    const size_t rs = (size_t)kScreenH * kScreenW * 3;
    const int width = (int)pool->slots.size();
    for (int base = 0; base < count; base += width) {
        const int n = std::min(width, count - base);
#pragma omp parallel for schedule(dynamic, 1) num_threads(pool->num_threads)
        for (int i = 0; i < n; i++) {
            const size_t j = (size_t)(base + i);
            Slot& slot = *pool->slots[i];
            serve(pool, slot, states + j * ss);
            run(slot, 1, true);
            slot.gb.display.lcd_draw_line = nullptr;
            slot.primed = false;
            paint(slot, rgb + j * rs);
        }
    }
    return 0;
}

int pokemon_pool_save(void* handle, int slot_index, uint8_t* out) {
    auto* pool = static_cast<Pool*>(handle);
    stash(pool, *pool->slots[slot_index], out);
    return 0;
}

int pokemon_pool_load(void* handle, int slot_index, const uint8_t* in) {
    auto* pool = static_cast<Pool*>(handle);
    serve(pool, *pool->slots[slot_index], in);
    return 0;
}

int pokemon_pool_observe(void* handle, int slot_index, uint8_t* frame, uint8_t* ram) {
    auto* pool = static_cast<Pool*>(handle);
    picture(*pool->slots[slot_index], frame);
    observe(*pool->slots[slot_index], ram);
    return 0;
}

int pokemon_pool_reset(void* handle, int slot_index) {
    auto* pool = static_cast<Pool*>(handle);
    return boot(pool, *pool->slots[slot_index]) ? 0 : -1;
}

int pokemon_pool_errors(void* handle, int slot_index) {
    auto* pool = static_cast<Pool*>(handle);
    return pool->slots[slot_index]->errors;
}

void pokemon_pool_destroy(void* handle) {
    auto* pool = static_cast<Pool*>(handle);
    for (Slot* slot : pool->slots) delete slot;
    delete pool;
}

}  // extern "C"
