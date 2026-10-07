#include <stdint.h>

#define ENABLE_SOUND 0
#define ENABLE_LCD 1

const uint8_t* peanut_rom;
#define PEANUT_GB_ROM_READ(gb, addr) (peanut_rom[(addr)])

#include "vendor/peanut_gb.h"
