/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Smoke test: load stage9 weights, evaluate the starting position.
 * Usage: capi_smoke <models_dir> <bearoff_db_path>
 */
#include <stdio.h>
#include <string.h>

#include "bgbot/capi.h"

/* stage9 slot order with the canonical map already applied (19 slots,
 * 15 distinct files; slots 11,12,15,16 alias slot 12's file). */
static const char* kPlanFiles[19] = {
    "sl_s9_purerace.weights.best",  "sl_s9_race_race.weights.best",
    "sl_s9_race_att.weights.best",  "sl_s9_race_prim.weights.best",
    "sl_s9_race_anch.weights.best", "sl_s9_att_race.weights.best",
    "sl_s9_att_att.weights.best",   "sl_s9_att_prim.weights.best",
    "sl_s9_att_anch.weights.best",  "sl_s9_prim_race.weights.best",
    "sl_s9_prim_att.weights.best",  "sl_s9_prim_anch.weights.best",
    "sl_s9_prim_anch.weights.best", "sl_s9_anch_race.weights.best",
    "sl_s9_anch_att.weights.best",  "sl_s9_prim_anch.weights.best",
    "sl_s9_prim_anch.weights.best", "sl_s9_player_bg.weights.best",
    "sl_s9_opponent_bg.weights.best",
};

static const int kStartingBoard[BGSAGE_BOARD_SIZE] = {
    0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: capi_smoke <models_dir> <bearoff_db>\n");
        return 2;
    }
    static char paths[19][1024];
    const char* path_ptrs[19];
    int hiddens[19];
    for (int i = 0; i < 19; ++i) {
        snprintf(paths[i], sizeof(paths[i]), "%s/%s", argv[1], kPlanFiles[i]);
        path_ptrs[i] = paths[i];
        hiddens[i] = i == 0 ? 100 : 400;
    }

    char err[512] = {0};
    bgsage_engine_config config;
    memset(&config, 0, sizeof(config));
    config.strategy_type = "backgame_pair";
    config.weight_paths = path_ptrs;
    config.hidden_sizes = hiddens;
    config.n_weights = 19;
    config.bearoff_db_path = argv[2];
    config.kind = BGSAGE_KIND_PLY;
    config.n_plies = 1;
    config.threads = 1;

    bgsage_engine* engine = bgsage_engine_create(&config, err, sizeof(err));
    if (engine == NULL) {
        fprintf(stderr, "create failed: %s\n", err);
        return 1;
    }
    printf("capi %s: engine created\n", bgsage_capi_version());

    bgsage_cube_ctx cube = {1, BGSAGE_OWNER_CENTERED, 0, 0, 0, 1, 1};

    bgsage_eval pre;
    if (bgsage_pre_roll(engine, kStartingBoard, &cube, &pre, err, sizeof(err)) != BGSAGE_OK) {
        fprintf(stderr, "pre_roll failed: %s\n", err);
        return 1;
    }
    printf("start pre-roll: win=%.4f eq=%+.4f cf=%+.4f\n",
           pre.probs.win, pre.cubeless, pre.cubeful);

    bgsage_move moves[64];
    int n_moves = 0;
    if (bgsage_checker_play(engine, kStartingBoard, 3, 1, &cube,
                            moves, 64, &n_moves, err, sizeof(err)) != BGSAGE_OK) {
        fprintf(stderr, "checker_play failed: %s\n", err);
        return 1;
    }
    printf("3-1: %d moves; best eq=%+.4f win=%.4f (p8=%d p5=%d p6=%d)\n",
           n_moves, moves[0].cubeful, moves[0].probs.win,
           moves[0].board[8], moves[0].board[5], moves[0].board[6]);

    bgsage_cube_result cr;
    if (bgsage_cube_action(engine, kStartingBoard, &cube, &cr, err, sizeof(err)) != BGSAGE_OK) {
        fprintf(stderr, "cube_action failed: %s\n", err);
        return 1;
    }
    printf("cube: ND=%+.4f DT=%+.4f DP=%+.4f double=%d take=%d\n",
           cr.equity_nd, cr.equity_dt, cr.equity_dp, cr.should_double, cr.should_take);

    /* The 3-1 best play must make the 5-point: two checkers on 5, and
     * 8/6 reduced by one each. */
    if (moves[0].board[5] != 2 || moves[0].board[8] != 2 || moves[0].board[6] != 4) {
        fprintf(stderr, "FAIL: best 3-1 is not 8/5 6/5\n");
        return 1;
    }
    if (pre.probs.win < 0.5 || pre.probs.win > 0.58) {
        fprintf(stderr, "FAIL: start win prob out of range\n");
        return 1;
    }
    bgsage_engine_destroy(engine);
    printf("SMOKE OK\n");
    return 0;
}
