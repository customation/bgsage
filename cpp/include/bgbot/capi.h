// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Mark Higgins (engine), Customation AS (C API)
//
// bgsage_capi — a flat C ABI over the bgbot engine core, for embedders
// that cannot (or should not) use the Python package: engine daemons,
// FFI bindings, GUIs. One bgsage_engine handle corresponds to one
// evaluation level (a strategy stack); handles are cheap to keep and
// reuse, expensive to create (weights load from disk).
//
// Thread-safety: calls on ONE handle must be serialized by the caller.
// Different handles may be used from different threads concurrently,
// with one caveat inherited from the core: MultiPlyStrategy instances
// share a thread-local position cache, which this API clears after
// every N-ply evaluation (matching the Python analyzer's behavior).
#ifndef BGBOT_CAPI_H
#define BGBOT_CAPI_H

#include <stddef.h>

#ifdef _WIN32
#  ifdef BGSAGE_CAPI_BUILD
#    define BGSAGE_API __declspec(dllexport)
#  else
#    define BGSAGE_API __declspec(dllimport)
#  endif
#else
#  define BGSAGE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ── Errors ──────────────────────────────────────────────────────────
// Every fallible function returns 0 on success, nonzero on failure and,
// when err/errlen are supplied, writes a NUL-terminated message.
#define BGSAGE_OK              0
#define BGSAGE_E_INVALID_ARG   1
#define BGSAGE_E_LOAD_FAILED   2
#define BGSAGE_E_EVAL_FAILED   3
#define BGSAGE_E_CANCELLED     4
#define BGSAGE_E_BUFFER_SMALL  5

// ── Board convention ────────────────────────────────────────────────
// int[26], always from the perspective of the player on roll:
// index 0 = opponent's bar (>= 0), 1..24 = points (positive = player on
// roll), 25 = the on-roll player's bar (>= 0).
#define BGSAGE_BOARD_SIZE 26

// ── Engine construction ─────────────────────────────────────────────

typedef struct bgsage_engine bgsage_engine;

#define BGSAGE_KIND_PLY     1  // n_plies 1..4; 1 = raw NN (XG convention)
#define BGSAGE_KIND_ROLLOUT 2  // (truncated) rollout per bgsage_rollout_config

// Per-purpose trial evaluation config (mirrors core TrialEvalConfig).
// All zeros = unset/inherit.
typedef struct {
    int ply;             // N-ply depth (0 = unset, 1 = raw NN)
    int rollout_trials;  // > 0 = inner truncated rollout instead of N-ply
    int rollout_depth;   // inner rollout truncation depth (default 5)
    int rollout_ply;     // inner rollout decision ply (default 1)
} bgsage_trial_eval;

// Rollout configuration (mirrors core RolloutConfig; defaults match the
// unified create_rollout factory). The named app levels are expressible:
//   1T: trials=72,  truncation=5, decision_ply=1, ultra_late_threshold=2
//   2T: trials=360, truncation=7, decision_ply=2, truncation_ply=2,
//       late_ply=1, late_threshold=1, cube.ply=2, cube_late.ply=2,
//       ultra_late_threshold=9999
//   3T: trials=360, truncation=7, decision_ply=3, late_ply=2,
//       late_threshold=2, ultra_late_threshold=9999
//   full rollout: trials=1296, truncation=0, ultra_late_threshold=9999
typedef struct {
    int n_trials;               // default 1296
    int truncation_depth;       // default 0 (play to completion)
    int decision_ply;           // default 1
    int truncation_ply;         // default -1 (= decision_ply)
    int late_ply;               // default -1
    int late_threshold;         // default 20
    int ultra_late_threshold;   // default 2; 9999 disables ply drop
    int enable_vr;              // default 1
    int cubeful_trial_moves;    // default 1
    int cubeful_late_threshold; // default 0
    unsigned int seed;          // default 42
    double target_se;           // default 0 (off)
    int max_batches;            // default 50
    double prefilter_threshold; // checker-play stage-1 loose cull; 0 = off
    bgsage_trial_eval checker, checker_late, cube, cube_late;
} bgsage_rollout_config;

// Fill a config with the factory defaults above.
BGSAGE_API void bgsage_rollout_config_init(bgsage_rollout_config* config);

typedef struct {
    // Strategy family: "5nn" | "pair" | "backgame_pair" (stage9).
    const char* strategy_type;
    // Weight files + hidden sizes, one per NN slot (19 for backgame_pair,
    // aliases already resolved by the caller — pass the same file twice
    // where the canonical map collapses slots).
    const char* const* weight_paths;
    const int* hidden_sizes;
    int n_weights;
    // Bearoff database path, or NULL to run without one.
    const char* bearoff_db_path;

    int kind;                       // BGSAGE_KIND_*
    int n_plies;                    // KIND_PLY: 1..4
    bgsage_rollout_config rollout;  // KIND_ROLLOUT
    int filter_max_moves;           // move filter (default 5 when 0)
    double filter_threshold;        // default 0.08 when 0
    int threads;                    // parallel threads; 0 = auto
} bgsage_engine_config;

BGSAGE_API bgsage_engine* bgsage_engine_create(
    const bgsage_engine_config* config, char* err, size_t errlen);
BGSAGE_API void bgsage_engine_destroy(bgsage_engine* engine);

// ── Per-call cube/match context ─────────────────────────────────────

#define BGSAGE_OWNER_CENTERED 0
#define BGSAGE_OWNER_MOVER    1
#define BGSAGE_OWNER_OPPONENT 2

typedef struct {
    int cube_value;   // 1, 2, 4, ...
    int owner;        // BGSAGE_OWNER_*
    int away1;        // mover's away score; 0/0 = money game
    int away2;
    int is_crawford;
    int jacoby;       // money only; forced off in match play internally
    int beaver;       // money only; forced off in match play internally
} bgsage_cube_ctx;

// ── Results ─────────────────────────────────────────────────────────

typedef struct {
    double win, win_gammon, win_backgammon, lose_gammon, lose_backgammon;
} bgsage_probs;

typedef struct {
    bgsage_probs probs;   // post-move, mover's perspective
    double cubeless;
    double cubeful;
} bgsage_eval;

typedef struct {
    bgsage_probs probs;   // pre-roll, mover's perspective
    double cubeless;
    double equity_nd, equity_dt, equity_dp;  // offerer's POV, normalized
    int should_double, should_take, is_beaver;
} bgsage_cube_result;

typedef struct {
    int board[BGSAGE_BOARD_SIZE];  // post-move board, mover's perspective
    bgsage_probs probs;            // post-move, mover's perspective
    double cubeless;
    double cubeful;                // ranking equity (cubeful)
    double equity_diff;            // 0 for best, negative below
} bgsage_move;

// ── Evaluation ──────────────────────────────────────────────────────

// Static evaluation of a post-move board (mover just moved; opponent on
// roll next). Cubeful via the engine's level.
BGSAGE_API int bgsage_post_move(
    bgsage_engine* engine, const int board[BGSAGE_BOARD_SIZE],
    const bgsage_cube_ctx* cube, bgsage_eval* out, char* err, size_t errlen);

// Pre-roll evaluation of a position from the on-roll player's
// perspective (the chance node before the mover rolls): flip, evaluate,
// invert — the identity used across the platform.
BGSAGE_API int bgsage_pre_roll(
    bgsage_engine* engine, const int board[BGSAGE_BOARD_SIZE],
    const bgsage_cube_ctx* cube, bgsage_eval* out, char* err, size_t errlen);

// Cube action for the player on roll.
BGSAGE_API int bgsage_cube_action(
    bgsage_engine* engine, const int board[BGSAGE_BOARD_SIZE],
    const bgsage_cube_ctx* cube, bgsage_cube_result* out,
    char* err, size_t errlen);

// Ranked checker plays for the given dice, best first. Writes up to
// max_moves entries; *n_out receives the count. A full legal-move list
// never exceeds a few dozen entries; 64 is a safe buffer size.
BGSAGE_API int bgsage_checker_play(
    bgsage_engine* engine, const int board[BGSAGE_BOARD_SIZE],
    int die1, int die2, const bgsage_cube_ctx* cube,
    bgsage_move* out, int max_moves, int* n_out, char* err, size_t errlen);

// ── Progress and cancellation (rollout engines; no-ops otherwise) ───

typedef void (*bgsage_progress_fn)(void* user, int done, int total);
BGSAGE_API void bgsage_engine_set_progress(
    bgsage_engine* engine, bgsage_progress_fn fn, void* user);
BGSAGE_API void bgsage_engine_cancel(bgsage_engine* engine);
BGSAGE_API void bgsage_engine_reset_cancel(bgsage_engine* engine);

// ── Introspection ───────────────────────────────────────────────────

// The C API's own semantic version, e.g. "0.1.0".
BGSAGE_API const char* bgsage_capi_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // BGBOT_CAPI_H
