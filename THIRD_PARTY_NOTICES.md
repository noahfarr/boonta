# Third-party notices

These parts of boonta come from other projects and keep their own terms.

## whlo

The website in `docs/` runs training in the browser with [whlo](https://github.com/noahfarr/whlo), vendored in `docs/runtime/vendor/whlo` and licensed under Apache-2.0 (`docs/runtime/vendor/whlo/LICENSE`).

## Arcade Learning Environment

`boonta/environments/ale/ffi/vendor.sh` downloads the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) v0.12.0, which is licensed under GPL-2.0, and applies `rebase_frame_pointer.patch` to its Stella TIA emulation. The patch is distributed under GPL-2.0. ALE itself is not included, and a library built against it falls under GPL-2.0.

## Peanut-GB

`boonta/environments/peanut_gb/ffi/vendor/peanut_gb.h` is [Peanut-GB](https://github.com/deltabeard/Peanut-GB) at commit 8e656982f0, Copyright (c) 2018-2023 Mahyar Koshkouei, licensed under MIT (`vendor/LICENSE`). Parts of it come from [SameBoy](https://github.com/LIJI32/SameBoy), Copyright (c) 2015-2019 Lior Halphon, also MIT. `sprites.patch` and `rom_read.patch` in the same directory are boonta's changes to it.

## Game data

No game ROMs are included. The Pokémon Red environment needs a ROM you supply at `boonta/environments/peanut_gb/roms/pokemon_red.gb`, and `start_states/pokemon_red.bin` is a Peanut-GB save state for that ROM. The event flag names in `pokemon_red/flags.py` and the map names and sizes in `pokemon_red/maps.py` were read from the [pret/pokered](https://github.com/pret/pokered) disassembly. Pokémon is a trademark of Nintendo, Creatures and GAME FREAK.
