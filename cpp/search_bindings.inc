// pybind11 bindings of the native lookahead (included by module.cpp inside its anonymous
// namespace, after engine_py, so the CState / action tuple / rng helpers are in scope):
//
//   HeuristicEval(temperature) / MlpEval(W, b, mean, std) / BlendEval(net, alpha, heuristic)
//       -> evaluator handles with .evaluate(states, players) (and .logits for MlpEval)
//   future_values(states, me, depth, levels, rolls, evaluator, robber=None, rng=None,
//                 deadline=None, node_budget=None, trace=False)
//       -> (values, nodes) or (values, nodes, steps, leaves, leaf_values) with trace=True
//   reduced_search(state, me, depth, levels, evaluator, ...) -> (value, nodes)
//   choose_discard / best_robber_move / win_path / robber_weights (differential tests)
namespace search_py {

using namespace engine_py;

struct EvalHandle {
    std::shared_ptr<const Evaluator> ev;
    std::string kind;
    std::shared_ptr<const MlpEval> mlp;  // set when kind == "mlp"
};

using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::vector<float> f32_vector(py::handle arr, const char* what, py::ssize_t expect) {
    F32 a = F32::ensure(arr);
    if (!a) throw py::type_error(std::string(what) + " must be a float array");
    if (a.ndim() != 1) throw py::value_error(std::string(what) + " must be one-dimensional");
    if (expect >= 0 && a.shape(0) != expect)
        throw py::value_error(std::string(what) + " has " + std::to_string(a.shape(0)) + " entries, expected " +
                              std::to_string(expect));
    return std::vector<float>(a.data(), a.data() + a.shape(0));
}

std::unique_ptr<EvalHandle> make_heuristic(double temperature) {
    std::unique_ptr<EvalHandle> h(new EvalHandle());
    h->ev = std::make_shared<HeuristicEval>(temperature);
    h->kind = "heuristic";
    return h;
}

std::unique_ptr<EvalHandle> make_mlp(py::sequence W, py::sequence b, py::handle mean, py::handle std_dev) {
    const py::ssize_t layers = (py::ssize_t)py::len(W);
    if (layers < 1 || (py::ssize_t)py::len(b) != layers)
        throw py::value_error("W and b must be non-empty lists of equal length");
    std::vector<int> sizes;
    std::vector<std::vector<float>> Wt, bias;
    for (py::ssize_t l = 0; l < layers; ++l) {
        py::object w_obj = W[l];
        F32 w = F32::ensure(w_obj);
        if (!w || w.ndim() != 2) throw py::value_error("every W must be a 2-D float array (in, out)");
        const int in = (int)w.shape(0), out = (int)w.shape(1);
        if (l == 0) sizes.push_back(in);
        else if (sizes.back() != in) throw py::value_error("layer " + std::to_string(l) + ": input size mismatch");
        sizes.push_back(out);
        std::vector<float> t((size_t)in * (size_t)out);
        const float* src = w.data();
        for (int i = 0; i < in; ++i)
            for (int j = 0; j < out; ++j) t[(size_t)j * (size_t)in + (size_t)i] = src[(size_t)i * (size_t)out + (size_t)j];
        Wt.push_back(std::move(t));
        py::object b_obj = b[l];
        bias.push_back(f32_vector(b_obj, "b", out));
    }
    if (sizes.front() != NUM_FEATURES) throw py::value_error("the net must take NUM_FEATURES inputs");
    if (sizes.back() != 1) throw py::value_error("the net must have a single output");
    std::vector<float> mu = f32_vector(mean, "mean", NUM_FEATURES);
    std::vector<float> sd = f32_vector(std_dev, "std", NUM_FEATURES);
    std::unique_ptr<EvalHandle> h(new EvalHandle());
    auto net = std::make_shared<MlpEval>(std::move(Wt), std::move(bias), std::move(mu), std::move(sd), sizes);
    h->mlp = net;
    h->ev = net;
    h->kind = "mlp";
    return h;
}

std::unique_ptr<EvalHandle> make_blend(const EvalHandle& net, double alpha, const EvalHandle& heuristic) {
    std::unique_ptr<EvalHandle> h(new EvalHandle());
    h->ev = std::make_shared<BlendEval>(net.ev, alpha, heuristic.ev);
    h->kind = "blend";
    return h;
}

// The states argument (a sequence of GameState / CState) as C++ structs plus, per state, the
// GameState it came from (for materialising leaves).
void read_states(py::handle states, std::vector<GameStateC>& out, std::vector<py::object>& origins) {
    Ref fast(PySequence_Fast(states.ptr(), "states must be a sequence"));
    if (!fast.p) throw py::error_already_set();
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
    PyObject** items = PySequence_Fast_ITEMS(fast.p);
    out.resize((size_t)n);
    origins.clear();
    for (Py_ssize_t i = 0; i < n; ++i) {
        py::handle o(items[i]);
        if (py::isinstance<CState>(o)) {
            const CState& c = py::cast<const CState&>(o);
            out[(size_t)i] = c.s;
            origins.push_back(c.origin);
        } else {
            state_from_python(items[i], out[(size_t)i]);
            origins.push_back(py::reinterpret_borrow<py::object>(o));
        }
    }
}

std::vector<int> read_players(py::handle players, const std::vector<GameStateC>& states) {
    Ref fast(PySequence_Fast(players.ptr(), "players must be a sequence"));
    if (!fast.p) throw py::error_already_set();
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
    if ((size_t)n != states.size()) throw py::value_error("states and players must have the same length");
    std::vector<int> out((size_t)n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const long p = index_of(PySequence_Fast_ITEMS(fast.p)[i]);
        check_player(states[(size_t)i], p);
        out[(size_t)i] = (int)p;
    }
    return out;
}

py::array_t<double> eval_evaluate(const EvalHandle& h, py::handle states, py::handle players) {
    std::vector<GameStateC> st;
    std::vector<py::object> origins;
    read_states(states, st, origins);
    const std::vector<int> pl = read_players(players, st);
    py::array_t<double> out({(py::ssize_t)st.size()});
    std::vector<const GameStateC*> ptrs(st.size());
    for (size_t i = 0; i < st.size(); ++i) ptrs[i] = &st[i];
    if (!st.empty()) h.ev->evaluate_batch(ptrs.data(), pl.data(), (int)st.size(), out.mutable_data());
    return out;
}

py::array_t<float> eval_logits(const EvalHandle& h, py::handle states, py::handle players) {
    if (!h.mlp) throw py::type_error("logits() is only available on an MlpEval handle");
    std::vector<GameStateC> st;
    std::vector<py::object> origins;
    read_states(states, st, origins);
    const std::vector<int> pl = read_players(players, st);
    py::array_t<float> out({(py::ssize_t)st.size()});
    for (size_t i = 0; i < st.size(); ++i) out.mutable_data()[i] = h.mlp->logit(st[i], pl[i]);
    return out;
}

int dict_int(py::dict d, const char* key, int dflt) {
    if (!d.contains(key)) return dflt;
    return d[key].cast<int>();
}

std::vector<LevelCfg> parse_levels(py::handle levels) {
    std::vector<LevelCfg> out;
    for (py::handle item : py::iter(levels)) {
        py::dict d = py::cast<py::dict>(item);
        LevelCfg c;
        c.opponent_actions = dict_int(d, "opponent_actions", c.opponent_actions);
        c.beam = dict_int(d, "beam", c.beam);
        c.expand = dict_int(d, "expand", c.expand);
        c.max_actions_per_turn = dict_int(d, "max_actions_per_turn", c.max_actions_per_turn);
        c.roll_samples = dict_int(d, "roll_samples", c.roll_samples);
        c.opp_roll_samples = dict_int(d, "opp_roll_samples", c.opp_roll_samples);
        c.finished_lookahead = dict_int(d, "finished_lookahead", c.finished_lookahead);
        c.discard_candidates = dict_int(d, "discard_candidates", c.discard_candidates);
        if (d.contains("max_nodes")) c.max_nodes = d["max_nodes"].cast<long>();
        out.push_back(c);
    }
    if (out.empty()) throw py::value_error("levels must contain at least one config");
    return out;
}

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::unique_ptr<RobberWeights> parse_robber(py::handle robber) {
    if (robber.is_none()) return nullptr;
    py::dict d = py::cast<py::dict>(robber);
    std::unique_ptr<RobberWeights> rw(new RobberWeights());
    const std::string mode = d["mode"].cast<std::string>();
    if (mode == "none") return nullptr;
    rw->mode = mode == "politics" ? 1 : (mode == "habit" ? 2 : 0);
    if (rw->mode == 0) throw py::value_error("robber mode must be none / politics / habit");
    const int n = d["n"].cast<int>();
    if (n < 1 || n > MAX_PLAYERS) throw py::value_error("robber weights: bad player count");
    rw->n = n;
    rw->has_habit = d.contains("has_habit") && d["has_habit"].cast<bool>();
    py::object c_obj = d["const"], hl_obj = d["habit_leader"], co_obj = d["coal"];
    F64 c = F64::ensure(c_obj);
    if (!c || c.ndim() != 3 || c.shape(0) != n || c.shape(1) != n || c.shape(2) != 4)
        throw py::value_error("robber weights: 'const' must have shape (n, n, 4)");
    F64 hl = F64::ensure(hl_obj);
    F64 co = F64::ensure(co_obj);
    if (!hl || hl.ndim() != 2 || hl.shape(0) != n || hl.shape(1) != n || !co || co.ndim() != 2 || co.shape(0) != n ||
        co.shape(1) != n)
        throw py::value_error("robber weights: 'habit_leader' / 'coal' must have shape (n, n)");
    for (int a = 0; a < n; ++a)
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < 4; ++k) rw->c[a][j][k] = c.data()[((size_t)a * n + j) * 4 + k];
            rw->habit_leader[a][j] = hl.data()[(size_t)a * n + j];
            rw->coal[a][j] = co.data()[(size_t)a * n + j];
        }
    return rw;
}

// rng argument: None (fresh seed), an int seed, or a random.Random shared by every simulation.
struct RngHolder {
    std::unique_ptr<PyRngDraw> py;
    uint64_t seed = 0;
    explicit RngHolder(py::handle rng) {
        if (rng.is_none() || PyLong_Check(rng.ptr())) seed = seed_of(rng);
        else py.reset(new PyRngDraw(rng.ptr()));
    }
};

std::vector<std::vector<int>> parse_rolls(py::handle rolls) {
    std::vector<std::vector<int>> out;
    for (py::handle seq : py::iter(rolls)) {
        std::vector<int> s;
        for (py::handle v : py::iter(seq)) {
            const int t = py::cast<int>(v);
            if (t < 2 || t > 12) throw py::value_error("roll totals must be in 2..12");
            s.push_back(t);
        }
        if (s.empty()) throw py::value_error("a roll sequence must not be empty");
        out.push_back(std::move(s));
    }
    if (out.empty()) throw py::value_error("rolls must contain at least one sequence");
    return out;
}

void setup_ctx(SearchCtx& ctx, const std::vector<LevelCfg>& levels, const EvalHandle& ev, const RobberWeights* rw,
               RngHolder& rng, py::handle deadline, py::handle node_budget) {
    ctx.levels = &levels;
    ctx.ev = ev.ev.get();
    ctx.rw = rw;
    ctx.shared_rng = rng.py.get();
    ctx.seed = rng.seed;
    ctx.deadline = deadline.is_none() ? -1.0 : py::cast<double>(deadline);
    ctx.node_budget = node_budget.is_none() ? -1L : py::cast<long>(node_budget);
}

py::object future_values_py(py::handle states, long me, int depth, py::handle levels, py::handle rolls,
                            const EvalHandle& ev, py::handle robber, py::handle rng, py::handle deadline,
                            py::handle node_budget, bool trace) {
    std::vector<GameStateC> st;
    std::vector<py::object> origins;
    read_states(states, st, origins);
    for (const GameStateC& s : st) check_player(s, me);
    const std::vector<LevelCfg> lv = parse_levels(levels);
    const std::vector<std::vector<int>> rl = parse_rolls(rolls);
    std::unique_ptr<RobberWeights> rw = parse_robber(robber);
    RngHolder holder(rng);
    SearchCtx ctx;
    setup_ctx(ctx, lv, ev, rw.get(), holder, deadline, node_budget);
    ctx.trace = trace;
    std::vector<double> vals;
    std::vector<GameStateC> leaves;
    std::vector<double> leaf_vals;
    if (!st.empty()) future_values(ctx, st, (int)me, depth, 0, rl, vals, trace ? &leaves : nullptr, trace ? &leaf_vals : nullptr);
    py::array_t<double> out({(py::ssize_t)vals.size()});
    for (size_t i = 0; i < vals.size(); ++i) out.mutable_data()[i] = vals[i];
    if (!trace) return py::make_tuple(out, ctx.nodes);
    const int n_samples = (int)rl.size();
    py::list steps((py::ssize_t)leaves.size());
    for (size_t i = 0; i < leaves.size(); ++i) steps[i] = py::list();
    for (const TraceStep& t : ctx.steps) {
        py::object act = py::reinterpret_steal<py::object>(action_to_python(t.action));
        py::object item = steps[(size_t)t.state_idx * (size_t)n_samples + (size_t)t.sample_idx];
        py::reinterpret_borrow<py::list>(item).append(py::make_tuple(t.player, act, t.draw));
    }
    py::list leaf_handles;
    for (size_t i = 0; i < leaves.size(); ++i) {
        std::unique_ptr<CState> c(new CState());
        c->s = leaves[i];
        c->origin = origins[i / (size_t)n_samples];
        leaf_handles.append(py::cast(std::move(c)));
    }
    py::array_t<double> lv_arr({(py::ssize_t)leaf_vals.size()});
    for (size_t i = 0; i < leaf_vals.size(); ++i) lv_arr.mutable_data()[i] = leaf_vals[i];
    return py::make_tuple(out, ctx.nodes, steps, leaf_handles, lv_arr);
}

py::object reduced_search_py(py::handle state, long me, int depth, py::handle levels, const EvalHandle& ev,
                             py::handle robber, py::handle rng, py::handle deadline, py::handle node_budget) {
    GameStateC buf;
    const GameStateC& s = *resolve(state.ptr(), buf);
    check_player(s, me);
    const std::vector<LevelCfg> lv = parse_levels(levels);
    std::unique_ptr<RobberWeights> rw = parse_robber(robber);
    RngHolder holder(rng);
    SearchCtx ctx;
    setup_ctx(ctx, lv, ev, rw.get(), holder, deadline, node_budget);
    Xoshiro xo(holder.seed);
    DrawSource& draw = holder.py ? static_cast<DrawSource&>(*holder.py) : static_cast<DrawSource&>(xo);
    const double v = reduced_search(ctx, s, (int)me, depth, 0, draw);
    return py::make_tuple(v, ctx.nodes);
}

py::object choose_discard_py(py::handle state, long player, bool legal) {
    GameStateC buf;
    const GameStateC& s = *resolve(state.ptr(), buf);
    check_player(s, player);
    std::unique_ptr<ActionList> acts;
    if (legal) {
        acts.reset(new ActionList());
        legal_actions(s, *acts);
    }
    const ActionC a = choose_discard_action(s, (int)player, acts.get());
    return py::reinterpret_steal<py::object>(action_to_python(a));
}

py::tuple best_robber_move_py(py::handle state, long player, py::handle robber) {
    GameStateC buf;
    const GameStateC& s = *resolve(state.ptr(), buf);
    check_player(s, player);
    std::unique_ptr<RobberWeights> rw = parse_robber(robber);
    double tw[MAX_PLAYERS];
    const double* twp = nullptr;
    if (rw && rw->mode != 0) {
        robber_weights(s, (int)player, *rw, tw);
        twp = tw;
    }
    const RobberMove m = best_robber_move(s, (int)player, twp);
    return py::make_tuple(m.hex, m.victim, m.score);
}

py::list five(const double* v) {
    py::list out;
    for (int r = 0; r < NUM_RESOURCES; ++r) out.append(v[r]);
    return out;
}

py::dict win_path_py(py::handle state, long player) {
    GameStateC buf;
    const GameStateC& s = *resolve(state.ptr(), buf);
    check_player(s, player);
    WinPathC wp;
    win_path(s, (int)player, wp);
    py::dict d;
    d["vp"] = wp.vp;
    d["need_vp"] = wp.need_vp;
    d["cost"] = five(wp.cost);
    d["hand"] = five(wp.hand);
    d["hand_known"] = wp.hand_known;
    d["missing"] = five(wp.missing);
    d["need_share"] = five(wp.need_share);
    d["eff_need"] = five(wp.eff_need);
    d["prod"] = five(wp.prod);
    d["supply"] = five(wp.supply);
    d["turns"] = wp.turns;
    d["min_turns"] = wp.min_turns;
    d["can_win_now"] = wp.can_win_now;
    d["danger"] = wp.danger;
    d["blocked_now"] = wp.blocked_now;
    d["n_steps"] = wp.n_steps + (wp.no_path ? 1 : 0);
    return d;
}

py::list robber_weights_py(py::handle state, long actor, py::handle robber) {
    GameStateC buf;
    const GameStateC& s = *resolve(state.ptr(), buf);
    check_player(s, actor);
    std::unique_ptr<RobberWeights> rw = parse_robber(robber);
    py::list out;
    if (!rw) return out;
    double w[MAX_PLAYERS];
    robber_weights(s, (int)actor, *rw, w);
    for (int j = 0; j < s.num_players; ++j) out.append(w[j]);
    return out;
}

py::list needed_vector_py(py::handle state, long player) {
    GameStateC buf;
    const GameStateC& s = *resolve(state.ptr(), buf);
    check_player(s, player);
    int need[NUM_RESOURCES];
    needed_vector(s, (int)player, need);
    py::list out;
    for (int r = 0; r < NUM_RESOURCES; ++r) out.append(need[r]);
    return out;
}

void register_search(py::module_& m) {
    py::class_<EvalHandle>(m, "Evaluator",
                           "A native leaf evaluator: HeuristicEval / MlpEval / BlendEval.  evaluate(states, players) "
                           "returns float64 win probabilities like the Python evaluators; MlpEval.logits gives the "
                           "float32 logits.")
        .def("evaluate", &eval_evaluate, py::arg("states"), py::arg("players"))
        .def("logits", &eval_logits, py::arg("states"), py::arg("players"))
        .def_property_readonly("kind", [](const EvalHandle& h) { return h.kind; });
    m.def("HeuristicEval", &make_heuristic, py::arg("temperature") = 16.0,
          "heuristic.HeuristicEvaluator(temperature) as a native evaluator handle.");
    m.def("MlpEval", &make_mlp, py::arg("W"), py::arg("b"), py::arg("mean"), py::arg("std"),
          "model.ValueNet as a native evaluator: W = [W0 (in, out), ...], b = [b0, ...] (float32), mean / std "
          "(float32, NUM_FEATURES).  Agrees with ValueNet.predict to float32 precision.");
    m.def("BlendEval", &make_blend, py::arg("net"), py::arg("alpha"), py::arg("heuristic"),
          "selfplay.BlendedEvaluator(net, alpha, heuristic) as a native evaluator.");
    m.def("future_values", &future_values_py, py::arg("states"), py::arg("me"), py::arg("depth"), py::arg("levels"),
          py::arg("rolls"), py::arg("evaluator"), py::arg("robber") = py::none(), py::arg("rng") = py::none(),
          py::arg("deadline") = py::none(), py::arg("node_budget") = py::none(), py::arg("trace") = false,
          "Searcher._future_values(states, me, depth) natively: -> (values float64 (N,), nodes).  `levels` are "
          "the SearchConfig dicts per lookahead level (search.reduced_config), `rolls` the sampled dice "
          "sequences, `robber` the accel.robber_weights_bundle dict, `rng` None / int seed / random.Random.  "
          "With trace=True -> (values, nodes, steps, leaves, leaf_values): per (state, sample) the applied "
          "(player, action, draw_index) list and the leaf as a CState.");
    m.def("reduced_search", &reduced_search_py, py::arg("state"), py::arg("me"), py::arg("depth"), py::arg("levels"),
          py::arg("evaluator"), py::arg("robber") = py::none(), py::arg("rng") = py::none(),
          py::arg("deadline") = py::none(), py::arg("node_budget") = py::none(),
          "The native reduced our-turn search (no trades): -> (value, nodes).");
    m.def("choose_discard", &choose_discard_py, py::arg("state"), py::arg("player"), py::arg("legal") = false,
          "discard.choose_discard(state, player[, legal_actions=engine.legal_actions(state)]).");
    m.def("best_robber_move", &best_robber_move_py, py::arg("state"), py::arg("player"), py::arg("robber") = py::none(),
          "robber.best_robber_move(state, player, target_weights=<from the robber bundle>) -> (hex, victim, score).");
    m.def("win_path", &win_path_py, py::arg("state"), py::arg("player"), "danger.win_path numeric fields as a dict.");
    m.def("robber_weights", &robber_weights_py, py::arg("state"), py::arg("actor"), py::arg("robber"),
          "politics.robber_target_weights / OpponentModel.robber_habit_weights from the robber bundle.");
    m.def("needed_vector", &needed_vector_py, py::arg("state"), py::arg("player"), "discard.needed_vector.");
}

}  // namespace search_py
