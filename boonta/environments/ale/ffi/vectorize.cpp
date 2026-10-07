#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "ale/ale_interface.hpp"
#include "ale/common/Log.hpp"

namespace {

constexpr int kRamSize = 128;
constexpr int kStateBytes = 40000;

struct Vec {
    std::vector<ale::ALEInterface*> envs;
    std::vector<ale::Action> actions;
    int num_envs;
    int frame_skip;

    float fraction;
    float novelty;
    int capacity;
    int depth;
    int warmup_episodes;
    int64_t snapshot_every;
    std::string snapshot_path;
    int64_t step_count = 0;
    int64_t next_snapshot = 0;
    std::mt19937 rng;
    std::vector<uint8_t> bank;
    std::vector<int64_t> write;
    std::vector<int64_t> touches;
    std::vector<uint8_t> room_of;
    std::vector<double> best;
    std::vector<double> accrued;
    std::vector<uint8_t> warm;
    std::vector<uint8_t> ram;
    int64_t episodes = 0;
    int64_t served = 0;
};

int cell_of(const uint8_t* ram, int capacity) {
    uint32_t h = ram[3];
    h = h * 31u + ram[65];
    h = h * 31u + (ram[42] >> 4);
    h = h * 31u + (ram[43] >> 4);
    h *= 2654435761u;
    return h % (uint32_t)capacity;
}

void observe_env(Vec* vec, int i) {
    const auto& memory = vec->envs[i]->getRAM();
    std::memcpy(vec->ram.data() + i * kRamSize, memory.array(), kRamSize);
}

int stash_env(Vec* vec, int i, uint8_t* buffer) {
    ale::ALEState state = vec->envs[i]->cloneSystemState();
    std::string blob = state.serialize();
    if (blob.size() + sizeof(uint32_t) > kStateBytes) return -1;
    uint32_t size = static_cast<uint32_t>(blob.size());
    std::memcpy(buffer, &size, sizeof(size));
    std::memcpy(buffer + sizeof(size), blob.data(), blob.size());
    return static_cast<int>(blob.size());
}

bool serve_env(Vec* vec, int i, const uint8_t* buffer) {
    uint32_t size;
    std::memcpy(&size, buffer, sizeof(size));
    if (size == 0 || size + sizeof(size) > kStateBytes) return false;
    std::string blob(reinterpret_cast<const char*>(buffer + sizeof(size)), size);
    ale::ALEState state(blob);
    vec->envs[i]->restoreSystemState(state);
    return true;
}

void restart(Vec* vec, int i) {
    if (vec->fraction <= 0.0f || vec->episodes < vec->warmup_episodes) return;
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    if (coin(vec->rng) >= vec->fraction) return;
    std::vector<double> weight(vec->capacity, 0.0);
    double total = 0.0;
    for (int c = 0; c < vec->capacity; c++) {
        if (vec->write[c] > 0) {
            weight[c] = (std::max(vec->best[c], 0.0) + 1.0) /
                        std::pow(1.0 + (double)vec->touches[c], (double)vec->novelty);
            total += weight[c];
        }
    }
    if (total <= 0.0) return;
    std::uniform_real_distribution<double> draw(0.0, total);
    double mark = draw(vec->rng);
    int cell = -1;
    for (int c = 0; c < vec->capacity; c++) {
        if (weight[c] > 0.0) {
            cell = c;
            mark -= weight[c];
            if (mark <= 0.0) break;
        }
    }
    if (cell < 0) return;
    int64_t filled = std::min(vec->write[cell], (int64_t)vec->depth);
    if (filled <= 0) return;
    std::uniform_int_distribution<int64_t> pick(0, filled - 1);
    int64_t slot = pick(vec->rng);
    if (!serve_env(vec, i,
                   vec->bank.data() + ((int64_t)cell * vec->depth + slot) * kStateBytes))
        return;
    observe_env(vec, i);
    vec->warm[i] = 1;
    vec->served++;
}

}  // namespace

extern "C" {

void* ale_vec_create(int num_envs, const char* rom_path, int frame_skip,
                     int seed, int num_threads, float fraction, int capacity,
                     int depth, int warmup_episodes, float novelty,
                     long long snapshot_every, const char* snapshot_path) {
    auto* vec = new Vec();
    vec->num_envs = num_envs;
    vec->frame_skip = frame_skip;
    vec->fraction = fraction;
    vec->novelty = novelty;
    vec->snapshot_every = snapshot_every;
    vec->snapshot_path = snapshot_path ? snapshot_path : "";
    vec->capacity = capacity;
    vec->depth = depth;
    vec->warmup_episodes = warmup_episodes;
    vec->rng.seed(seed);
    vec->bank.assign((int64_t)capacity * depth * kStateBytes, 0);
    vec->write.assign(capacity, 0);
    vec->touches.assign(capacity, 0);
    vec->room_of.assign(capacity, 0);
    vec->best.assign(capacity, -1e18);
    vec->accrued.assign(num_envs, 0.0);
    vec->warm.assign(num_envs, 0);
    vec->ram.assign((int64_t)num_envs * kRamSize, 0);
    omp_set_num_threads(num_threads);
    ale::Logger::setMode(ale::Logger::Error);
    for (int i = 0; i < num_envs; i++) {
        auto* env = new ale::ALEInterface();
        env->setInt("random_seed", seed + i);
        env->setFloat("repeat_action_probability", 0.0f);
        env->setInt("frame_skip", 1);
        env->setInt("max_num_frames_per_episode", 108000);
        env->loadROM(rom_path);
        vec->envs.push_back(env);
        vec->actions = env->getMinimalActionSet();
        observe_env(vec, i);
    }
    return vec;
}

int ale_vec_num_actions(void* handle) {
    return static_cast<int>(static_cast<Vec*>(handle)->actions.size());
}

void ale_vec_reset_all(void* handle, uint8_t* ram) {
    auto* vec = static_cast<Vec*>(handle);
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < vec->num_envs; i++) {
        vec->envs[i]->reset_game();
        observe_env(vec, i);
    }
    std::fill(vec->accrued.begin(), vec->accrued.end(), 0.0);
    std::fill(vec->warm.begin(), vec->warm.end(), 0);
    std::memcpy(ram, vec->ram.data(), vec->ram.size());
}

void ale_vec_step(void* handle, const int32_t* actions, float* rewards,
                  uint8_t* dones, uint8_t* warms, uint8_t* ram) {
    auto* vec = static_cast<Vec*>(handle);
    std::vector<int> prior_cell(vec->num_envs);
    for (int i = 0; i < vec->num_envs; i++)
        prior_cell[i] = cell_of(vec->ram.data() + i * kRamSize, vec->capacity);

#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < vec->num_envs; i++) {
        auto* env = vec->envs[i];
        float reward = 0.0f;
        for (int k = 0; k < vec->frame_skip && !env->game_over(); k++) {
            reward += env->act(vec->actions[actions[i]]);
        }
        bool done = env->game_over();
        if (done) env->reset_game();
        rewards[i] = reward;
        dones[i] = done ? 1 : 0;
        observe_env(vec, i);
    }

    for (int i = 0; i < vec->num_envs; i++) {
        vec->accrued[i] += rewards[i];
        int cell = cell_of(vec->ram.data() + i * kRamSize, vec->capacity);
        bool entered = !dones[i] && cell != prior_cell[i];
        int touched = entered ? cell : prior_cell[i];
        if (vec->accrued[i] > vec->best[touched]) vec->best[touched] = vec->accrued[i];
        if (entered) {
            vec->touches[cell]++;
            vec->room_of[cell] = vec->ram[i * kRamSize + 3];
        }
        if (entered && (vec->write[cell] < vec->depth ||
                        std::uniform_int_distribution<int>(0, 63)(vec->rng) == 0)) {
            int64_t slot = vec->write[cell] % vec->depth;
            if (stash_env(vec, i, vec->bank.data() +
                          ((int64_t)cell * vec->depth + slot) * kStateBytes) > 0)
                vec->write[cell]++;
        }
        warms[i] = vec->warm[i];
        if (dones[i]) {
            vec->episodes++;
            vec->accrued[i] = 0.0;
            vec->warm[i] = 0;
            restart(vec, i);
        }
    }
    std::memcpy(ram, vec->ram.data(), vec->ram.size());

    vec->step_count += vec->num_envs;
    if (vec->snapshot_every > 0 && !vec->snapshot_path.empty() &&
        vec->step_count >= vec->next_snapshot) {
        vec->next_snapshot = vec->step_count + vec->snapshot_every;
        FILE* f = std::fopen(vec->snapshot_path.c_str(), "ab");
        if (f) {
            std::fwrite(&vec->step_count, sizeof(int64_t), 1, f);
            std::fwrite(vec->room_of.data(), sizeof(uint8_t), vec->capacity, f);
            std::fwrite(vec->best.data(), sizeof(double), vec->capacity, f);
            std::fwrite(vec->write.data(), sizeof(int64_t), vec->capacity, f);
            std::fwrite(vec->touches.data(), sizeof(int64_t), vec->capacity, f);
            std::fclose(f);
        }
    }
}

int64_t ale_vec_served(void* handle) { return static_cast<Vec*>(handle)->served; }

void ale_vec_destroy(void* handle) {
    auto* vec = static_cast<Vec*>(handle);
    for (auto* env : vec->envs) delete env;
    delete vec;
}

}  // extern "C"
