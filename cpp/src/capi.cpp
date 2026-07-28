// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Mark Higgins (engine), Customation AS (C API)
//
// Implementation notes: every evaluation flow here is a verbatim port of
// an existing reference consumer — the pybind bindings (batch_checker_play
// 1-ply/N-ply lambdas, evaluate_cube_decision_unified,
// cube_decision_nply_unified, RolloutStrategy.cube_decision) and the
// Python analyzer's rollout checker-play orchestration. Divergence from
// those references is a bug; the parity harness in the engine-server
// repository compares this API's output against BgBotAnalyzer.

#include "bgbot/capi.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/match_equity.h"
#include "bgbot/moves.h"
#include "bgbot/multipy.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"
#include "bgbot/rollout.h"
#include "bgbot/strategy.h"

using namespace bgbot;

namespace {

constexpr const char* kVersion = "0.1.0";

void set_error(char* err, size_t errlen, const std::string& message) {
    if (err == nullptr || errlen == 0) return;
    std::snprintf(err, errlen, "%s", message.c_str());
}

CubeOwner to_owner(int owner) {
    switch (owner) {
        case BGSAGE_OWNER_MOVER: return CubeOwner::PLAYER;
        case BGSAGE_OWNER_OPPONENT: return CubeOwner::OPPONENT;
        default: return CubeOwner::CENTERED;
    }
}

// Match play force-disables jacoby/beaver (analyzer.py:1336-1338).
struct CallCtx {
    CubeInfo ci;
    bool match_play;
};

CallCtx make_cube_info(const bgsage_cube_ctx* cube) {
    const bool match_play = cube->away1 > 0 || cube->away2 > 0;
    const bool jacoby = match_play ? false : cube->jacoby != 0;
    const bool beaver = match_play ? false : cube->beaver != 0;
    CubeInfo ci{cube->cube_value, to_owner(cube->owner),
                MatchInfo{cube->away1, cube->away2, cube->is_crawford != 0},
                -1.0f, jacoby, beaver, 0};
    return {ci, match_play};
}

Board to_board(const int values[BGSAGE_BOARD_SIZE]) {
    Board board{};
    for (int i = 0; i < BGSAGE_BOARD_SIZE; ++i) board[i] = values[i];
    return board;
}

void from_board(const Board& board, int out[BGSAGE_BOARD_SIZE]) {
    for (int i = 0; i < BGSAGE_BOARD_SIZE; ++i) out[i] = board[i];
}

bgsage_probs to_probs(const std::array<float, NUM_OUTPUTS>& probs) {
    return {probs[0], probs[1], probs[2], probs[3], probs[4]};
}

std::array<float, NUM_OUTPUTS> inverted(const std::array<float, NUM_OUTPUTS>& probs) {
    return {1.0f - probs[0], probs[3], probs[4], probs[1], probs[2]};
}

std::shared_ptr<Strategy> make_base_strategy(
    const std::string& strategy_type,
    const std::vector<std::string>& weight_paths,
    const std::vector<int>& hidden_sizes) {
    // Mirrors bindings.cpp make_strategy_from_type.
    if (strategy_type == "5nn")
        return std::make_shared<GamePlanStrategy>(weight_paths, hidden_sizes);
    if (strategy_type == "pair")
        return std::make_shared<GamePlanPairStrategy>(weight_paths, hidden_sizes);
    if (strategy_type == "backgame_pair")
        return std::make_shared<BackgameAwarePairStrategy>(weight_paths, hidden_sizes);
    throw std::invalid_argument("unknown strategy_type: " + strategy_type);
}

}  // namespace

struct bgsage_engine {
    int kind = BGSAGE_KIND_PLY;
    int n_plies = 1;
    MoveFilter filter{5, 0.08f};
    double prefilter_threshold = 0.0;
    int threads = 0;

    std::shared_ptr<Strategy> base;             // raw 1-ply strategy
    std::shared_ptr<Strategy> base_eval;        // base, bearoff-wrapped when db loaded
    std::shared_ptr<MultiPlyStrategy> multipy;  // kind PLY, n_plies >= 2
    std::shared_ptr<RolloutStrategy> rollout;   // kind ROLLOUT
    std::shared_ptr<MultiPlyStrategy> rollout_2ply;  // rollout checker prefilter stage
    std::unique_ptr<BearoffDB> bearoff;

    bgsage_progress_fn progress_fn = nullptr;
    void* progress_user = nullptr;

    void report_progress(int done, int total) const {
        if (progress_fn != nullptr) progress_fn(progress_user, done, total);
    }
};

void bgsage_rollout_config_init(bgsage_rollout_config* config) {
    if (config == nullptr) return;
    std::memset(config, 0, sizeof(*config));
    config->n_trials = 1296;
    config->truncation_depth = 0;
    config->decision_ply = 1;
    config->truncation_ply = -1;
    config->late_ply = -1;
    config->late_threshold = 20;
    config->ultra_late_threshold = 2;
    config->enable_vr = 1;
    config->cubeful_trial_moves = 1;
    config->cubeful_late_threshold = 0;
    config->seed = 42;
    config->target_se = 0.0;
    config->max_batches = 50;
    config->checker.rollout_depth = 5;
    config->checker.rollout_ply = 1;
    config->checker_late = config->checker;
    config->cube = config->checker;
    config->cube_late = config->checker;
}

bgsage_engine* bgsage_engine_create(
    const bgsage_engine_config* config, char* err, size_t errlen) {
    if (config == nullptr || config->strategy_type == nullptr ||
        config->weight_paths == nullptr || config->hidden_sizes == nullptr ||
        config->n_weights <= 0) {
        set_error(err, errlen, "invalid engine config");
        return nullptr;
    }
    try {
        auto engine = std::make_unique<bgsage_engine>();
        engine->kind = config->kind;
        engine->n_plies = config->n_plies;
        engine->filter = MoveFilter{
            config->filter_max_moves > 0 ? config->filter_max_moves : 5,
            config->filter_threshold > 0 ? (float)config->filter_threshold : 0.08f};
        engine->threads = config->threads;

        std::vector<std::string> paths;
        std::vector<int> hiddens;
        paths.reserve(config->n_weights);
        hiddens.reserve(config->n_weights);
        for (int i = 0; i < config->n_weights; ++i) {
            paths.emplace_back(config->weight_paths[i]);
            hiddens.push_back(config->hidden_sizes[i]);
        }

        engine->base = make_base_strategy(config->strategy_type, paths, hiddens);
        engine->base_eval = engine->base;

        if (config->bearoff_db_path != nullptr && config->bearoff_db_path[0] != '\0') {
            engine->bearoff = std::make_unique<BearoffDB>();
            if (!engine->bearoff->load(config->bearoff_db_path)) {
                set_error(err, errlen,
                          std::string("failed to load bearoff db: ") + config->bearoff_db_path);
                return nullptr;
            }
            engine->base_eval = std::make_shared<BearoffStrategy>(
                engine->base, engine->bearoff.get());
        }

        if (config->kind == BGSAGE_KIND_PLY) {
            if (config->n_plies < 1 || config->n_plies > 4) {
                set_error(err, errlen, "n_plies must be 1..4");
                return nullptr;
            }
            if (config->n_plies >= 2) {
                engine->multipy = std::make_shared<MultiPlyStrategy>(
                    engine->base, config->n_plies, engine->filter,
                    /*full_depth_opponent=*/false,
                    /*parallel_evaluate=*/true, config->threads);
                // PubEval prefilter, exactly as create_multipy wires it.
                static auto pubeval = std::make_shared<PubEval>();
                engine->multipy->set_move_prefilter(pubeval);
                if (engine->bearoff)
                    engine->multipy->set_bearoff_db(engine->bearoff.get());
            }
        } else if (config->kind == BGSAGE_KIND_ROLLOUT) {
            const bgsage_rollout_config& rc_in = config->rollout;
            engine->prefilter_threshold = rc_in.prefilter_threshold;
            RolloutConfig rc;
            rc.n_trials = rc_in.n_trials;
            rc.truncation_depth = rc_in.truncation_depth;
            rc.decision_ply = rc_in.decision_ply;
            rc.truncation_ply = rc_in.truncation_ply;
            rc.enable_vr = rc_in.enable_vr != 0;
            rc.parallelize_trials = true;
            rc.filter = engine->filter;
            rc.n_threads = config->threads;
            rc.seed = rc_in.seed;
            rc.late_ply = rc_in.late_ply;
            rc.late_threshold = rc_in.late_threshold;
            rc.ultra_late_threshold = rc_in.ultra_late_threshold;
            rc.cubeful_trial_moves = rc_in.cubeful_trial_moves != 0;
            rc.cubeful_late_threshold = rc_in.cubeful_late_threshold;
            rc.target_se = rc_in.target_se;
            rc.max_batches = rc_in.max_batches;
            auto to_trial = [](const bgsage_trial_eval& t) {
                TrialEvalConfig config;
                config.ply = t.ply;
                config.rollout_trials = t.rollout_trials;
                config.rollout_depth = t.rollout_depth > 0 ? t.rollout_depth : 5;
                config.rollout_ply = t.rollout_ply > 0 ? t.rollout_ply : 1;
                return config;
            };
            rc.checker = to_trial(rc_in.checker);
            rc.checker_late = to_trial(rc_in.checker_late);
            rc.cube = to_trial(rc_in.cube);
            rc.cube_late = to_trial(rc_in.cube_late);

            engine->rollout = std::make_shared<RolloutStrategy>(engine->base, rc);
            if (engine->bearoff)
                engine->rollout->set_bearoff_db(engine->bearoff.get());

            // 2-ply rescore stage for checker-play prefiltering, mirroring
            // _RolloutAnalyzer._strategy_2ply.
            engine->rollout_2ply = std::make_shared<MultiPlyStrategy>(
                engine->base, 2, engine->filter, false, true, config->threads);
            static auto pubeval2 = std::make_shared<PubEval>();
            engine->rollout_2ply->set_move_prefilter(pubeval2);
            if (engine->bearoff)
                engine->rollout_2ply->set_bearoff_db(engine->bearoff.get());
        } else {
            set_error(err, errlen, "unknown engine kind");
            return nullptr;
        }
        return engine.release();
    } catch (const std::exception& ex) {
        set_error(err, errlen, ex.what());
        return nullptr;
    }
}

void bgsage_engine_destroy(bgsage_engine* engine) { delete engine; }

void bgsage_engine_set_progress(bgsage_engine* engine, bgsage_progress_fn fn, void* user) {
    if (engine == nullptr) return;
    engine->progress_fn = fn;
    engine->progress_user = user;
}

void bgsage_engine_cancel(bgsage_engine* engine) {
    if (engine != nullptr && engine->rollout) engine->rollout->cancel();
}

void bgsage_engine_reset_cancel(bgsage_engine* engine) {
    if (engine != nullptr && engine->rollout) engine->rollout->reset_cancel();
}

const char* bgsage_capi_version(void) { return kVersion; }

namespace {

// Scored candidate used by the checker-play flows.
struct Scored {
    Board board;
    std::array<float, NUM_OUTPUTS> probs;
    float cubeless;
    float cubeful;
    bool is_survivor = false;
};

// 1-ply scoring of every candidate — verbatim from the batch_checker_play
// 1-ply lambda (evaluate → compute_equity → cube_efficiency → cl2cf),
// sorted by cubeful equity descending.
std::vector<Scored> score_candidates_1ply(
    const Strategy& strategy, const std::vector<Board>& candidates,
    const Board& pre_board, const CubeInfo& ci) {
    std::vector<Scored> scored(candidates.size());
    for (size_t j = 0; j < candidates.size(); ++j) {
        auto post_probs = strategy.evaluate_probs(candidates[j], pre_board);
        float cl_eq = NeuralNetwork::compute_equity(post_probs);
        bool race = is_race(candidates[j]);
        auto [pp, op] = pip_counts(candidates[j]);
        float x = cube_efficiency(post_probs, race, pp, op);
        float cf_eq = cl2cf(post_probs, ci, x);
        scored[j] = {candidates[j], post_probs, cl_eq, cf_eq, false};
    }
    std::sort(scored.begin(), scored.end(),
              [](const Scored& a, const Scored& b) { return a.cubeful > b.cubeful; });
    return scored;
}

// The reference filter: top max_moves within threshold of best; >= 1 kept.
int filter_keep_count(const std::vector<Scored>& scored, const MoveFilter& filter) {
    if (scored.empty()) return 0;
    float best = scored[0].cubeful;
    int keep = 0;
    for (size_t j = 0; j < scored.size() && keep < filter.max_moves; ++j) {
        if ((best - scored[j].cubeful) < filter.threshold) ++keep;
        else break;
    }
    return keep == 0 ? 1 : keep;
}

// Opponent's-optimal cubeful equity for a post-move board at N-ply,
// sign-flipped — verbatim from the batch_checker_play N-ply compute_cubeful.
float compute_cubeful_nply(const Board& post_move_board, const CubeInfo& ci,
                           Strategy& strategy_1ply, int n_plies,
                           const MoveFilter& filter) {
    Board opp_pre_roll = flip(post_move_board);
    CubeOwner opp_owner;
    switch (ci.owner) {
        case CubeOwner::PLAYER: opp_owner = CubeOwner::OPPONENT; break;
        case CubeOwner::OPPONENT: opp_owner = CubeOwner::PLAYER; break;
        default: opp_owner = CubeOwner::CENTERED; break;
    }
    if (ci.is_money()) {
        CubeInfo opp_ci{1, opp_owner, MatchInfo{}, -1.0f, ci.jacoby, ci.beaver};
        return -cubeful_equity_nply(opp_pre_roll, opp_ci, strategy_1ply,
                                    n_plies, filter, /*n_threads=*/1);
    }
    CubeInfo opp_ci{1, opp_owner, ci.match.flip(), -1.0f, ci.jacoby, ci.beaver};
    return -cubeful_equity_nply(opp_pre_roll, opp_ci, strategy_1ply,
                                n_plies, filter, /*n_threads=*/1);
}

// Thread count for the standalone cubeful analytics entries. The capi config
// documents `threads` as 0 = auto, which the strategy constructors resolve
// themselves; these entries take a concrete count instead, so 0 means serial.
int analytics_threads(const bgsage_engine* engine) {
    return engine->threads > 0 ? engine->threads : 1;
}

int fill_moves(const std::vector<Scored>& scored, bgsage_move* out,
               int max_moves, int* n_out) {
    const int count = std::min<int>((int)scored.size(), max_moves);
    const double best = scored.empty() ? 0.0 : scored[0].cubeful;
    for (int j = 0; j < count; ++j) {
        from_board(scored[j].board, out[j].board);
        out[j].probs = to_probs(scored[j].probs);
        out[j].cubeless = scored[j].cubeless;
        out[j].cubeful = scored[j].cubeful;
        out[j].equity_diff = scored[j].cubeful - best;
    }
    if (n_out != nullptr) *n_out = count;
    return BGSAGE_OK;
}

}  // namespace

int bgsage_checker_play(
    bgsage_engine* engine, const int board_in[BGSAGE_BOARD_SIZE],
    int die1, int die2, const bgsage_cube_ctx* cube,
    bgsage_move* out, int max_moves, int* n_out, char* err, size_t errlen) {
    if (engine == nullptr || board_in == nullptr || cube == nullptr ||
        out == nullptr || max_moves <= 0) {
        set_error(err, errlen, "invalid argument");
        return BGSAGE_E_INVALID_ARG;
    }
    try {
        const Board pre_board = to_board(board_in);
        const auto call = make_cube_info(cube);
        auto candidates = possible_boards(pre_board, die1, die2);
        if (candidates.empty()) {
            if (n_out != nullptr) *n_out = 0;
            return BGSAGE_OK;
        }

        if (engine->kind == BGSAGE_KIND_PLY && engine->n_plies == 1) {
            auto scored = score_candidates_1ply(*engine->base_eval, candidates,
                                                pre_board, call.ci);
            return fill_moves(scored, out, max_moves, n_out);
        }

        if (engine->kind == BGSAGE_KIND_PLY) {
            // Analyzer parity (_CubefulAnalyzer.checker_play_analytics with a
            // multi-ply inner, use_cube_aware_probs branch): EVERY candidate's
            // probs, cubeless AND cubeful equity come from ONE cube-aware
            // N-ply traversal of its flipped (opponent pre-roll) node —
            // cubeful_probs_and_equity_nply — then sort by cubeful. The
            // cubeless inner analyzer's survivor filtering only fills fields
            // this overwrite replaces, so it drops out of the observable
            // result.
            //
            // The traversal spreads its 21 top-level rolls over the shared
            // multipy pool (cube_eval.cpp enables that only at n_plies > 2).
            // Parity is unaffected: rolls accumulate into arCf in fixed index
            // order whatever the thread count, so the equities are identical
            // to the serial walk — this is throughput only.
            CubeInfo opp_ci = call.ci;
            if (opp_ci.owner == CubeOwner::PLAYER) opp_ci.owner = CubeOwner::OPPONENT;
            else if (opp_ci.owner == CubeOwner::OPPONENT) opp_ci.owner = CubeOwner::PLAYER;
            std::swap(opp_ci.match.away1, opp_ci.match.away2);

            std::vector<Scored> scored;
            scored.reserve(candidates.size());
            for (const Board& candidate : candidates) {
                auto r = cubeful_probs_and_equity_nply(
                    flip(candidate), opp_ci, *engine->base_eval, engine->n_plies,
                    engine->filter, analytics_threads(engine));
                Scored entry;
                entry.board = candidate;
                entry.probs = inverted(r.probs);
                entry.cubeless = NeuralNetwork::compute_equity(entry.probs);
                entry.cubeful = -r.equity;
                scored.push_back(entry);
            }
            std::sort(scored.begin(), scored.end(),
                      [](const Scored& a, const Scored& b) { return a.cubeful > b.cubeful; });
            clear_cubeful_eval_cache();
            return fill_moves(scored, out, max_moves, n_out);
        }

        // Rollout flow — port of _RolloutAnalyzer.checker_play_analytics:
        // stage-1 loose cull (prefilter_threshold) or TINY filter at 1-ply;
        // optional 2-ply rescore + TINY filter; >= 2 survivors guaranteed;
        // each survivor rolled out via cubeful_rollout_position.
        engine->rollout->reset_cancel();
        CubeInfo ci = call.ci;
        auto scored_1ply = score_candidates_1ply(*engine->base_eval, candidates,
                                                 pre_board, ci);
        std::vector<Scored> survivors;
        if (engine->prefilter_threshold > 0 && scored_1ply.size() > 1) {
            MoveFilter loose{(int)scored_1ply.size(), (float)engine->prefilter_threshold};
            int stage1_keep = filter_keep_count(scored_1ply, loose);
            std::vector<Board> stage1_boards;
            for (int j = 0; j < stage1_keep; ++j) stage1_boards.push_back(scored_1ply[j].board);
            auto scored_2ply = score_candidates_1ply(*engine->rollout_2ply, stage1_boards,
                                                     pre_board, ci);
            engine->rollout_2ply->clear_cache();
            int keep = filter_keep_count(scored_2ply, engine->filter);
            survivors.assign(scored_2ply.begin(), scored_2ply.begin() + keep);
            // Backfill to >= 2 from the 2-ply pool.
            for (size_t j = keep; survivors.size() < 2 && j < scored_2ply.size(); ++j)
                survivors.push_back(scored_2ply[j]);
        } else {
            int keep = filter_keep_count(scored_1ply, engine->filter);
            survivors.assign(scored_1ply.begin(), scored_1ply.begin() + keep);
            for (size_t j = keep; survivors.size() < 2 && j < scored_1ply.size(); ++j)
                survivors.push_back(scored_1ply[j]);
        }

        const int total_moves = (int)survivors.size();
        const int n_trials = engine->rollout->config().n_trials;
        for (int i = 0; i < total_moves; ++i) {
            auto progress = [engine, i, n_trials, total_moves](int done, int) {
                engine->report_progress(i * n_trials + done, total_moves * n_trials);
            };
            auto result = engine->rollout->cubeful_rollout_position(
                survivors[i].board, ci, progress);
            survivors[i].cubeful = result.cubeful_equity;
            survivors[i].cubeless = result.cubeless.equity;
            survivors[i].probs = result.cubeless.mean_probs;
        }
        std::sort(survivors.begin(), survivors.end(),
                  [](const Scored& a, const Scored& b) { return a.cubeful > b.cubeful; });
        return fill_moves(survivors, out, max_moves, n_out);
    } catch (const RolloutCancelled&) {
        set_error(err, errlen, "cancelled");
        return BGSAGE_E_CANCELLED;
    } catch (const std::exception& ex) {
        set_error(err, errlen, ex.what());
        return BGSAGE_E_EVAL_FAILED;
    }
}

int bgsage_post_move(
    bgsage_engine* engine, const int board_in[BGSAGE_BOARD_SIZE],
    const bgsage_cube_ctx* cube, bgsage_eval* out, char* err, size_t errlen) {
    if (engine == nullptr || board_in == nullptr || cube == nullptr || out == nullptr) {
        set_error(err, errlen, "invalid argument");
        return BGSAGE_E_INVALID_ARG;
    }
    try {
        const Board board = to_board(board_in);
        const auto call = make_cube_info(cube);

        std::array<float, NUM_OUTPUTS> probs;
        if (engine->kind == BGSAGE_KIND_PLY && engine->n_plies >= 2) {
            // Analyzer parity (post_move_analytics, n_plies > 1 with cube
            // info): probs come from the N-ply CUBE-AWARE tree at the
            // opponent's pre-roll node — flip the board, flip the cube
            // perspective, cubeful_probs_nply, invert back. NOT the
            // dead-cube evaluate_probs walk: its cubeless interior picks
            // diverge from the analyzer's match-aware picks.
            Board opp_pre_roll = flip(board);
            CubeInfo opp_ci = call.ci;
            if (opp_ci.owner == CubeOwner::PLAYER) opp_ci.owner = CubeOwner::OPPONENT;
            else if (opp_ci.owner == CubeOwner::OPPONENT) opp_ci.owner = CubeOwner::PLAYER;
            std::swap(opp_ci.match.away1, opp_ci.match.away2);
            auto opp_probs = cubeful_probs_nply(
                opp_pre_roll, opp_ci, *engine->base_eval, engine->n_plies,
                engine->filter, analytics_threads(engine));
            probs = inverted(opp_probs);
            clear_cubeful_eval_cache();
        } else {
            probs = engine->base_eval->evaluate_probs(board, is_race(board));
        }
        float cl_eq = NeuralNetwork::compute_equity(probs);
        bool race = is_race(board);
        auto [pp, op] = pip_counts(board);
        float x = cube_efficiency(probs, race, pp, op);
        float cf_eq = cl2cf(probs, call.ci, x);

        out->probs = to_probs(probs);
        out->cubeless = cl_eq;
        out->cubeful = cf_eq;
        return BGSAGE_OK;
    } catch (const std::exception& ex) {
        set_error(err, errlen, ex.what());
        return BGSAGE_E_EVAL_FAILED;
    }
}

int bgsage_pre_roll(
    bgsage_engine* engine, const int board_in[BGSAGE_BOARD_SIZE],
    const bgsage_cube_ctx* cube, bgsage_eval* out, char* err, size_t errlen) {
    if (engine == nullptr || board_in == nullptr || cube == nullptr || out == nullptr) {
        set_error(err, errlen, "invalid argument");
        return BGSAGE_E_INVALID_ARG;
    }
    try {
        // Flip to the just-moved player's perspective, evaluate post-move,
        // invert back to the mover — with the cube context flipped alongside.
        bgsage_cube_ctx flipped_ctx = *cube;
        if (cube->owner == BGSAGE_OWNER_MOVER) flipped_ctx.owner = BGSAGE_OWNER_OPPONENT;
        else if (cube->owner == BGSAGE_OWNER_OPPONENT) flipped_ctx.owner = BGSAGE_OWNER_MOVER;
        std::swap(flipped_ctx.away1, flipped_ctx.away2);

        Board flipped = flip(to_board(board_in));
        int flipped_arr[BGSAGE_BOARD_SIZE];
        from_board(flipped, flipped_arr);

        bgsage_eval opp{};
        int rc = bgsage_post_move(engine, flipped_arr, &flipped_ctx, &opp, err, errlen);
        if (rc != BGSAGE_OK) return rc;

        out->probs = {1.0 - opp.probs.win, opp.probs.lose_gammon,
                      opp.probs.lose_backgammon, opp.probs.win_gammon,
                      opp.probs.win_backgammon};
        out->cubeless = -opp.cubeless;
        out->cubeful = -opp.cubeful;
        return BGSAGE_OK;
    } catch (const std::exception& ex) {
        set_error(err, errlen, ex.what());
        return BGSAGE_E_EVAL_FAILED;
    }
}

int bgsage_cube_action(
    bgsage_engine* engine, const int board_in[BGSAGE_BOARD_SIZE],
    const bgsage_cube_ctx* cube, bgsage_cube_result* out,
    char* err, size_t errlen) {
    if (engine == nullptr || board_in == nullptr || cube == nullptr || out == nullptr) {
        set_error(err, errlen, "invalid argument");
        return BGSAGE_E_INVALID_ARG;
    }
    try {
        const Board board = to_board(board_in);
        const auto call = make_cube_info(cube);
        const CubeInfo& ci = call.ci;

        if (engine->kind == BGSAGE_KIND_ROLLOUT) {
            // Port of RolloutStrategy.cube_decision's ND/DT/DP collapse.
            engine->rollout->reset_cancel();
            auto progress = [engine](int done, int total) {
                engine->report_progress(done, total);
            };
            auto cfr = engine->rollout->config().target_se > 0.0
                ? engine->rollout->cubeful_cube_decision_batched(board, ci, progress)
                : engine->rollout->cubeful_cube_decision(board, ci, progress);

            const bool sp_can_double = can_double(ci);
            float equity_nd, equity_dt, equity_dp;
            bool should_double, should_take;
            if (!ci.is_money()) {
                const int a1 = ci.match.away1, a2 = ci.match.away2, cv = ci.cube_value;
                const bool craw = ci.match.is_crawford;
                float nd_m = (float)cfr.nd_equity, dt_m = (float)cfr.dt_equity;
                float dp_m = dp_mwc(a1, a2, cv, craw);
                if (!sp_can_double) {
                    should_double = false;
                    should_take = true;
                } else {
                    const bool auto_double = (!craw && a1 > 1 && a2 == 1);
                    if (auto_double) {
                        should_double = true;
                        should_take = (dt_m <= dp_m);
                    } else {
                        should_double = (std::min(dt_m, dp_m) > nd_m);
                        should_take = (dt_m <= dp_m);
                    }
                }
                equity_nd = mwc2eq(nd_m, a1, a2, cv, craw);
                equity_dt = mwc2eq(dt_m, a1, a2, cv, craw);
                equity_dp = mwc2eq(dp_m, a1, a2, cv, craw);
            } else {
                equity_dp = 1.0f;
                equity_nd = (float)cfr.nd_equity;
                float actual_dt = (float)cfr.dt_equity;
                equity_dt = (ci.beaver && actual_dt < 0.0f) ? 2.0f * actual_dt : actual_dt;
                if (!sp_can_double) {
                    should_double = false;
                    should_take = true;
                } else {
                    should_double = (std::min(equity_dt, equity_dp) > equity_nd);
                    should_take = (equity_dt <= equity_dp);
                }
            }
            out->probs = to_probs(cfr.cubeless.mean_probs);
            out->cubeless = cfr.cubeless.equity;
            out->equity_nd = equity_nd;
            out->equity_dt = equity_dt;
            out->equity_dp = equity_dp;
            out->should_double = should_double ? 1 : 0;
            out->should_take = should_take ? 1 : 0;
            out->is_beaver =
                (ci.is_money() && ci.beaver && sp_can_double && cfr.dt_equity < 0.0) ? 1 : 0;
            return BGSAGE_OK;
        }

        // Ply engines. 1-ply mirrors evaluate_cube_decision_unified; N-ply
        // mirrors cube_decision_nply_unified.
        Board flipped = flip(board);
        auto post = engine->base_eval->evaluate_probs(flipped, is_race(flipped));
        auto pre = inverted(post);
        bool race = is_race(board);
        auto [pp, op] = pip_counts(board);
        float x = cube_efficiency(pre, race, pp, op);

        CubeDecision cd;
        if (engine->n_plies == 1) {
            cd = cube_decision_1ply(pre, ci, x);
        } else {
            // cube_decision_nply_unified passes a PubEval prefilter for
            // move selection inside the cubeful recursion; without it the
            // interior picks (and therefore the equities and the discrete
            // action near thresholds) diverge from the analyzer's.
            static PubEval pubeval_filter;
            cd = cube_decision_nply(board, ci, *engine->base_eval, engine->n_plies,
                                    engine->filter, analytics_threads(engine),
                                    &pubeval_filter);
            // Pre-roll probs at N-ply for the report, matching the unified
            // binding's temporary MultiPlyStrategy usage.
            auto nply_post = engine->multipy->evaluate_probs(flipped, flipped);
            pre = inverted(nply_post);
            engine->multipy->clear_cache();
            clear_cubeful_eval_cache();
        }

        out->probs = to_probs(pre);
        out->cubeless = NeuralNetwork::compute_equity(pre);
        out->equity_nd = cd.equity_nd;
        out->equity_dt = cd.equity_dt;
        out->equity_dp = cd.equity_dp;
        out->should_double = cd.should_double ? 1 : 0;
        out->should_take = cd.should_take ? 1 : 0;
        out->is_beaver = cd.is_beaver ? 1 : 0;
        return BGSAGE_OK;
    } catch (const RolloutCancelled&) {
        set_error(err, errlen, "cancelled");
        return BGSAGE_E_CANCELLED;
    } catch (const std::exception& ex) {
        set_error(err, errlen, ex.what());
        return BGSAGE_E_EVAL_FAILED;
    }
}
