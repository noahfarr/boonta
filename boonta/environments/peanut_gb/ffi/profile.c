#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENABLE_SOUND 0
#define ENABLE_LCD 1
#define PEANUT_GB_HEADER_ONLY
#include "vendor/peanut_gb.h"

extern const uint8_t* peanut_rom;

static uint8_t cram[32768];
static uint8_t frame[LCD_HEIGHT * LCD_WIDTH];

static uint8_t rom_read(struct gb_s* gb, const uint_fast32_t addr) {
    (void)gb;
    return peanut_rom[addr];
}

static uint8_t cram_read(struct gb_s* gb, const uint_fast32_t addr) {
    (void)gb;
    return cram[addr];
}

static void cram_write(struct gb_s* gb, const uint_fast32_t addr, const uint8_t value) {
    (void)gb;
    cram[addr] = value;
}

static void on_error(struct gb_s* gb, const enum gb_error_e error, const uint16_t addr) {
    (void)gb;
    fprintf(stderr, "profile: emulator error %d at %04x\n", error, addr);
    exit(1);
}

static void draw_line(struct gb_s* gb, const uint8_t* pixels, const uint_fast8_t line) {
    (void)gb;
    memcpy(frame + line * LCD_WIDTH, pixels, LCD_WIDTH);
}

static const uint8_t pads[9] = {0xFF, (uint8_t)~JOYPAD_UP, (uint8_t)~JOYPAD_DOWN, (uint8_t)~JOYPAD_LEFT,
                                (uint8_t)~JOYPAD_RIGHT, (uint8_t)~JOYPAD_A, (uint8_t)~JOYPAD_B,
                                (uint8_t)~JOYPAD_START, (uint8_t)~JOYPAD_SELECT};

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    FILE* file = fopen(argv[1], "rb");
    if (file == NULL) return 1;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t* rom = malloc(size);
    if (fread(rom, 1, size, file) != (size_t)size) return 1;
    fclose(file);
    peanut_rom = rom;
    struct gb_s gb;
    memset(&gb, 0, sizeof(gb));
    if (gb_init(&gb, rom_read, cram_read, cram_write, on_error, NULL) != GB_INIT_NO_ERROR) return 1;
    gb_init_lcd(&gb, draw_line);
    int steps = atoi(argv[2]);
    unsigned seed = 1;
    for (int s = 0; s < steps; s++) {
        seed = seed * 1103515245u + 12345u;
        gb.direct.joypad = pads[(seed >> 16) % 9];
        gb.display.lcd_draw_line = NULL;
        for (int i = 0; i < 8; i++) gb_run_frame(&gb);
        gb.direct.joypad = 0xFF;
        for (int i = 0; i < 16; i++) {
            gb.display.lcd_draw_line = i == 15 ? draw_line : NULL;
            gb_run_frame(&gb);
        }
    }
    printf("profiled %d steps, map %d\n", steps, gb.wram[0x135E]);
    return 0;
}
