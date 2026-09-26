/* replay_core.js: decoder and view models of the Catanbot Replay Archive (docs/designs/replay_archive.md, section 7).
   UMD: window.ReplayCore in the page (inlined by scripts/replay_export.py), require() in node
   (scripts/check_replay_bundle.mjs). No dependencies, no game rules: every hand delta, VP, road length and title
   change is stored in the bundle; this file only accumulates them. */
(function (root) {
  "use strict";

  const COLORS = ["RED", "BLUE", "ORANGE", "WHITE"];
  const COLOR_NAMES = ["Red", "Blue", "Orange", "White"];
  const RES = ["wood", "brick", "sheep", "wheat", "ore"];
  const DEV = ["Knight", "Victory Point", "Road Building", "Year of Plenty", "Monopoly"];
  const KINDS = ["BUILD_SETTLEMENT", "BUILD_ROAD", "BUILD_CITY", "ROLL", "END_TURN", "BUY_DEVELOPMENT_CARD",
    "MOVE_ROBBER", "MARITIME_TRADE", "DISCARD_RESOURCE", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
    "PLAY_ROAD_BUILDING", "DISCARD"];
  const K = { SETTLE: 0, ROAD: 1, CITY: 2, ROLL: 3, END: 4, BUY: 5, ROBBER: 6, TRADE: 7, DISCARD1: 8, KNIGHT: 9,
    MONO: 10, YOP: 11, RB: 12, DISCARD: 13 };
  const OPP = { c: "our bot", v: "ValueFunction", a: "AlphaBeta", s: "SameTurnAlphaBeta",
    f: "ValueFunction stand-in", b: "AlphaBeta stand-in" };
  const TEST_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "R1", "R2",
    "H1", "H2", "M1", "M2", "M3"];
  const DEV_DECK = [14, 5, 2, 2, 2];
  const VPS_TO_WIN = 10;
  const HIDDEN = { NONE: 0, STEAL: 1, DISCARD: 2, DEV: 3 };
  const SORT_KEYS = ["game", "seed", "rolls", "turns", "ourvp", "oppvp", "margin", "behind", "dev", "meanp", "exact",
    "hidden"];
  const FMT_CODES = { "1v3": "1v3", "1v3m": "1v3-mixed", "2v2": "2v2", "2v2m": "2v2-mixed", "1v1": "1v1", "3v1": "3v1" };
  const VERSION_CODES = { c33: "3.3.0", c321: "3.2.1" };
  const BOARD_RES = { W: 0, B: 1, S: 2, G: 3, O: 4, D: -1 };
  const PORT_RES = { W: 0, B: 1, S: 2, G: 3, O: 4, "3": -1 };
  const NUM = { "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, A: 10, B: 11, C: 12, "-": 0 };
  const SEAT_W = 20; // per seat and position: hand 0-4, held 5-9, played 10-14, vp 15, pvp 16, lr 17, LR 18, LA 19

  function err(kind, props) {
    const e = new Error(kind === "http" ? `HTTP ${props.status} for ${props.url}` : kind === "notdata"
      ? `not game data: ${props.url}` : kind === "nogzip" ? "this browser cannot unpack gzip data" : kind);
    e.kind = kind;
    Object.assign(e, props || {});
    return e;
  }

  function shardPath(t, g, size) {
    return `shards/${t}_${String(Math.floor(g / (size || 100))).padStart(2, "0")}.json.gz`;
  }

  // ---- bytes → JSON (7.2) -------------------------------------------------------------------------------------
  function bytesKind(u8) {
    if (u8.length >= 2 && u8[0] === 0x1f && u8[1] === 0x8b) return "gzip";
    for (let i = 0; i < u8.length; i++) {
      const c = u8[i];
      if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d || c === 0xef || c === 0xbb || c === 0xbf) continue;
      return c === 0x7b ? "json" : "notdata";
    }
    return "notdata";
  }

  async function browserGunzip(u8) {
    if (typeof DecompressionStream === "undefined") throw err("nogzip", {});
    const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  // gunzip: (u8) => Uint8Array | Promise<Uint8Array>. Returns {data, path: "gzip" | "plain"}.
  async function parseJsonBytesEx(u8, gunzip, url) {
    const kind = bytesKind(u8);
    let bytes = u8;
    if (kind === "gzip") {
      if (!gunzip) throw err("nogzip", { url });
      bytes = await gunzip(u8);
      if (bytesKind(bytes) !== "json") throw err("notdata", { url });
    } else if (kind !== "json") {
      throw err("notdata", { url });
    }
    const text = new TextDecoder("utf-8").decode(bytes);
    return { data: JSON.parse(text), path: kind === "gzip" ? "gzip" : "plain" };
  }

  async function parseJsonBytes(u8, gunzip) {
    return (await parseJsonBytesEx(u8, gunzip)).data;
  }

  // opts.fetch and opts.gunzip can be injected (node validator); defaults are the browser's.
  async function fetchJsonEx(url, opts) {
    const o = opts || {};
    const f = o.fetch || (typeof fetch !== "undefined" ? fetch : null);
    const res = await f(url);
    if (!res.ok) throw err("http", { status: res.status, url });
    const u8 = new Uint8Array(await res.arrayBuffer());
    return parseJsonBytesEx(u8, o.gunzip || browserGunzip, url);
  }

  async function fetchJson(url, opts) {
    return (await fetchJsonEx(url, opts)).data;
  }

  // ---- board (3.2) ----------------------------------------------------------------------------------------------
  function parseBoard(code, geo) {
    const [land, ports] = code.split("|");
    if (!land || land.length !== 2 * geo.tiles.length || !ports || ports.length !== geo.ports.length) {
      throw new Error(`bad board code ${code}`);
    }
    const tiles = geo.tiles.map((t, i) => {
      const res = BOARD_RES[land[2 * i]], num = NUM[land[2 * i + 1]];
      if (res === undefined || num === undefined) throw new Error(`bad board code ${code}`);
      return { i, c: t[0].slice(), x: t[1], y: t[2], res, num };
    });
    const portsOut = geo.ports.map((p, i) => {
      const res = PORT_RES[ports[i]];
      if (res === undefined) throw new Error(`bad board code ${code}`);
      return { i, x: p[0], y: p[1], res, nodes: p[2].slice() };
    });
    const robber0 = tiles.findIndex(t => t.res === -1);
    return { tiles, ports: portsOut, robber0 };
  }

  function tileName(board, ti) {
    const t = board.tiles[ti];
    if (!t) return "an unknown tile";
    return t.res < 0 ? "the desert" : `${RES[t.res]} ${t.num}`;
  }

  // ---- decoder (3.4, 3.5) ---------------------------------------------------------------------------------------
  function expandH(hs, n) {
    if (!hs) return null;
    const h = [];
    for (let s = 0; s < n; s++) h.push([0, 0, 0, 0, 0]);
    for (let q = 0; q < hs.length; q += 2) h[Math.floor(hs[q] / 5)][hs[q] % 5] += hs[q + 1];
    return h;
  }

  function snapOf(D, k) {
    const n = D.n, W = n * SEAT_W, o = k * W, P = [];
    for (let s = 0; s < n; s++) {
      const b = o + s * SEAT_W, S = D.S;
      P.push({ hand: Array.from(S.subarray(b, b + 5)), held: Array.from(S.subarray(b + 5, b + 10)),
        played: Array.from(S.subarray(b + 10, b + 15)), vp: S[b + 15], pvp: S[b + 16], lr: S[b + 17],
        LR: S[b + 18] === 1, LA: S[b + 19] === 1 });
    }
    const g = k * 4;
    return { P, robber: D.G[g], bank: Array.from(D.B.subarray(k * 5, k * 5 + 5)), deck: D.G[g + 1],
      turns: D.G[g + 2], rolls: D.G[g + 3] };
  }

  function decodeGame(g, geo) {
    const n = g.n, ours = g.o.slice(), N = g.s.length;
    const board = parseBoard(g.b, geo);
    const me = g.bv ? ours[0] : -1;
    const W = n * SEAT_W;
    const S = new Int16Array((N + 1) * W), G = new Int16Array((N + 1) * 4), B = new Int16Array((N + 1) * 5);
    const cur = new Int16Array(W);
    let robber = board.robber0, deck = 25, turns = 0, rolls = 0, setupRoads = 0, lastEnd = -1;
    const write = k => {
      S.set(cur, k * W);
      G[k * 4] = robber; G[k * 4 + 1] = deck; G[k * 4 + 2] = turns; G[k * 4 + 3] = rolls;
      for (let r = 0; r < 5; r++) {
        let sum = 0;
        for (let s = 0; s < n; s++) sum += cur[s * SEAT_W + r];
        B[k * 5 + r] = 19 - sum;
      }
    };
    write(0);
    const steps = [], hiddenIdx = [];
    for (let i = 0; i < N; i++) {
      const t = g.s[i];
      const op = t[0], kind = op >> 2, a = op & 3;
      if (kind < 0 || kind >= KINDS.length || a >= n) throw new Error(`${g.id}: bad op ${op} at step ${i}`);
      const v = t.length > 1 ? t[1] : null;
      const hs = t.length > 2 && t[2] ? t[2] : null;
      const xs = t.length > 3 ? t[3] : null;
      const st = { a, kind, v, dice: null, rob: null, add: null, free: false, hidden: 0, h: expandH(hs, n),
        x: xs ? xs.slice() : null, r: 0 };
      if (hs) for (let q = 0; q < hs.length; q += 2) cur[Math.floor(hs[q] / 5) * SEAT_W + hs[q] % 5] += hs[q + 1];
      const devPlay = (d) => { cur[a * SEAT_W + 5 + d]--; cur[a * SEAT_W + 10 + d]++; };
      switch (kind) {
        case K.SETTLE: st.add = [["s", a, v]]; st.free = !hs || !hs.some((x, q) => q % 2 === 1 && x < 0); break;
        case K.ROAD:
          st.add = [["r", a, v]]; st.free = !hs;
          if (rolls === 0) { setupRoads++; if (setupRoads !== n && setupRoads !== 2 * n) turns++; }
          break;
        case K.CITY: st.add = [["c", a, v]]; break;
        case K.ROLL: rolls++; st.dice = [Math.floor(v / 10), v % 10]; break;
        case K.END: turns++; lastEnd = i; break;
        case K.BUY: deck--; cur[a * SEAT_W + 5 + v]++; break;
        case K.ROBBER: {
          robber = Math.floor(v / 100); st.rob = robber;
          st.victim = Math.floor(v / 10) % 10 - 1; st.stolen = v % 10 - 1;
          break;
        }
        case K.KNIGHT: devPlay(0); break;
        case K.MONO: devPlay(4); break;
        case K.YOP: devPlay(3); break;
        case K.RB: devPlay(2); break;
        default: break;
      }
      if (xs) {
        for (let q = 0; q < xs.length; q += 2) {
          const c = xs[q], val = xs[q + 1];
          if (c < 12) cur[(c & 3) * SEAT_W + 15 + (c >> 2)] = val;
          else for (let s = 0; s < n; s++) cur[s * SEAT_W + (c === 12 ? 18 : 19)] = s === val ? 1 : 0;
        }
      }
      st.r = rolls;
      if (me >= 0 && a !== me) {
        if (kind === K.ROBBER && st.victim >= 0 && st.stolen >= 0 && st.victim !== me) st.hidden = HIDDEN.STEAL;
        else if (kind === K.DISCARD1 || kind === K.DISCARD) st.hidden = HIDDEN.DISCARD;
        else if (kind === K.BUY) st.hidden = HIDDEN.DEV;
      }
      if (st.hidden) hiddenIdx.push(i);
      steps.push(st);
      write(i + 1);
    }
    const D = { id: g.id, t: g.id.split("-")[0], g: g.g, n, ours, me, counted: !!g.bv, board, N, steps, hiddenIdx,
      lastEnd, S, G, B, pub: false };
    D.snap = k => snapOf(D, k);
    return D;
  }

  function cloneDecoded(D) {
    const C = Object.assign({}, D, {
      ours: D.ours.slice(), board: JSON.parse(JSON.stringify(D.board)), hiddenIdx: D.hiddenIdx.slice(),
      S: new Int16Array(D.S), G: new Int16Array(D.G), B: new Int16Array(D.B),
      steps: D.steps.map(s => JSON.parse(JSON.stringify(s)))
    });
    C.snap = k => snapOf(C, k);
    return C;
  }

  // ---- public projection (7.1 PV) -------------------------------------------------------------------------------
  // Copies only what our bot saw at seat `me`: sizes, card counts, played cards, public VP, road lengths, titles,
  // pieces, robber, dice, bank (frozen inside discard runs, as the tracker's _observe_entry does), deck; for `me`
  // also hand, held cards and actual VP. Never keeps a reference to D.
  const PV_SEAT = 11; // size, devCount, played 0-4, pvp, lr, LR, LA
  function publicView(D, me) {
    const n = D.n, N = D.N;
    const P = new Int16Array((N + 1) * n * PV_SEAT), M = new Int16Array((N + 1) * 11);
    const G = new Int16Array((N + 1) * 4), B = new Int16Array((N + 1) * 5);
    let runStart = -1;
    for (let k = 0; k <= N; k++) {
      const sn = D.snap(k);
      for (let s = 0; s < n; s++) {
        const p = sn.P[s], b = (k * n + s) * PV_SEAT;
        P[b] = p.hand.reduce((x, y) => x + y, 0);
        P[b + 1] = p.held.reduce((x, y) => x + y, 0);
        for (let d = 0; d < 5; d++) P[b + 2 + d] = p.played[d];
        P[b + 7] = p.pvp; P[b + 8] = p.lr; P[b + 9] = p.LR ? 1 : 0; P[b + 10] = p.LA ? 1 : 0;
      }
      const mp = sn.P[me];
      for (let r = 0; r < 5; r++) { M[k * 11 + r] = mp.hand[r]; M[k * 11 + 5 + r] = mp.held[r]; }
      M[k * 11 + 10] = mp.vp;
      G[k * 4] = sn.robber; G[k * 4 + 1] = sn.deck; G[k * 4 + 2] = sn.turns; G[k * 4 + 3] = sn.rolls;
      const st = k > 0 ? D.steps[k - 1] : null;
      const isDiscard = st && (st.kind === K.DISCARD1 || st.kind === K.DISCARD);
      if (isDiscard) { if (runStart < 0) runStart = k - 1; } else runStart = -1;
      const src = runStart >= 0 ? D.snap(runStart).bank : sn.bank;
      for (let r = 0; r < 5; r++) B[k * 5 + r] = src[r];
    }
    const steps = D.steps.map((s, i) => {
      const pre = D.snap(i).P, post = D.snap(i + 1).P;
      const sizeD = [];
      for (let q = 0; q < n; q++) {
        sizeD.push(post[q].hand.reduce((x, y) => x + y, 0) - pre[q].hand.reduce((x, y) => x + y, 0));
      }
      const o = { a: s.a, kind: s.kind, v: null, dice: s.dice ? s.dice.slice() : null, rob: s.rob,
        add: s.add ? s.add.map(x => x.slice()) : null, free: s.free, hidden: s.hidden, h: null,
        hs: sizeD.some(x => x) ? sizeD : null, x: null, r: s.r };
      if (s.kind === K.ROBBER) { o.victim = s.victim; o.stolen = s.hidden ? null : s.stolen; }
      if (s.hidden === HIDDEN.STEAL) o.v = Math.floor(s.v / 10) * 10;
      else if (s.hidden === HIDDEN.DISCARD) o.v = null;
      else if (s.hidden === HIDDEN.DEV) o.v = null;
      else o.v = Array.isArray(s.v) ? s.v.slice() : s.v;
      if (!s.hidden && s.h) o.h = s.h.map(r => r.slice());
      if (s.x) {
        const x = [];
        for (let q = 0; q < s.x.length; q += 2) {
          const c = s.x[q];
          if (c >= 12 || (c >> 2) >= 1 || (c & 3) === me) x.push(c, s.x[q + 1]);
        }
        o.x = x.length ? x : null;
      }
      return o;
    });
    const PV = { id: D.id, t: D.t, g: D.g, n, ours: D.ours.slice(), me, counted: D.counted,
      board: JSON.parse(JSON.stringify(D.board)), N, steps, hiddenIdx: D.hiddenIdx.slice(), lastEnd: D.lastEnd,
      pub: true };
    PV.snap = function (k) {
      const out = [];
      for (let s = 0; s < n; s++) {
        const b = (k * n + s) * PV_SEAT;
        const row = { size: P[b], devCount: P[b + 1], played: Array.from(P.subarray(b + 2, b + 7)), pvp: P[b + 7],
          lr: P[b + 8], LR: P[b + 9] === 1, LA: P[b + 10] === 1 };
        if (s === me) {
          row.hand = Array.from(M.subarray(k * 11, k * 11 + 5));
          row.held = Array.from(M.subarray(k * 11 + 5, k * 11 + 10));
          row.vp = M[k * 11 + 10];
        }
        out.push(row);
      }
      return { P: out, robber: G[k * 4], bank: Array.from(B.subarray(k * 5, k * 5 + 5)), deck: G[k * 4 + 1],
        turns: G[k * 4 + 2], rolls: G[k * 4 + 3] };
    };
    return PV;
  }

  // ---- bot view (5.6, 7.1 BV) -----------------------------------------------------------------------------------
  function botView(g, D) {
    if (!g.bv) return null;
    const bv = g.bv, N = D.N, n = D.n, me = D.ours[0];
    const opps = [];
    for (let s = 0; s < n; s++) if (s !== me) opps.push(s);
    const qIdx = {}, dvIdx = {};
    for (const j of opps) { qIdx[j] = new Int32Array(N + 1).fill(-1); dvIdx[j] = new Int32Array(N + 1).fill(-1); }
    const nhAt = new Int32Array(N + 1).fill(1), dpIdx = new Int32Array(N + 1).fill(-1);
    const curQ = {}, curDv = {};
    for (const j of opps) { curQ[j] = -1; curDv[j] = -1; }
    let qi = 0, di = 0, ni = 0, pi = 0, nh = 1, dp = -1, lastK = 0;
    const q = bv.q || [], dv = bv.dv || [], nhs = bv.nh || [], dps = bv.dp || [];
    for (let k = 0; k <= N; k++) {
      while (qi < q.length && q[qi][0] === k) { curQ[q[qi][1]] = qi; qi++; }
      while (di < dv.length && dv[di][0] === k) { curDv[dv[di][1]] = di; di++; }
      while (ni < nhs.length && nhs[ni] === k) { nh = nhs[ni + 1]; ni += 2; }
      while (pi < dps.length && dps[pi][0] === k) { dp = pi; pi++; }
      for (const j of opps) { qIdx[j][k] = curQ[j]; dvIdx[j][k] = curDv[j]; }
      nhAt[k] = nh; dpIdx[k] = dp; lastK = k;
    }
    if (qi !== q.length || di !== dv.length || ni !== nhs.length || pi !== dps.length) {
      throw new Error(`${g.id}: bot-view rows out of order or beyond position ${lastK}`);
    }
    const unlog = x => Math.pow(10, -x / 1000);
    function opp(k, j) {
      const qi_ = qIdx[j][k], row = qi_ >= 0 ? q[qi_] : null;
      const di_ = dvIdx[j][k], law = di_ >= 0 ? dv[di_].slice(2).map(x => x / 1000) : [1];
      let o;
      if (!row || row[2] === 1) {
        const own = row ? row.slice(3, 8) : [0, 0, 0, 0, 0];
        o = { seat: j, exact: true, own, E: own.slice(), u: 0, top: 1, calp: 1 };
      } else {
        o = { seat: j, exact: false, own: null, E: row.slice(7, 12).map(x => x / 100), u: row[6], top: unlog(row[4]),
          calp: unlog(row[5]) };
      }
      o.vpLaw = law;
      return o;
    }
    function pub(k) {
      const dpr = dpIdx[k] >= 0 ? dps[dpIdx[k]].slice(1) : DEV_DECK.slice();
      return { nh: nhAt[k], pool: dpr, opps: opps.map(j => opp(k, j)) };
    }
    function p(k, j) {
      const qi_ = qIdx[j][k], row = qi_ >= 0 ? q[qi_] : null;
      return !row || row[2] === 1 ? 1 : unlog(row[3]);
    }
    const summary = bv.sum || {};
    const chk = bv.chk || {};
    return { me, opps: opps.slice(), N, pub, p, summary, chk, publicPart: () => ({ pub, summary, chk }) };
  }

  // ---- log text (7.3) -------------------------------------------------------------------------------------------
  function resList(counts) {
    return counts.map((c, r) => c ? `${c} ${RES[r]}` : null).filter(Boolean).join(", ");
  }
  function deltaText(row) {
    return row.map((d, r) => d ? `${d > 0 ? "+" : "−"}${Math.abs(d)} ${RES[r]}` : null).filter(Boolean).join(" ");
  }
  function plural(n, one, many) { return `${n} ${n === 1 ? one : (many || one + "s")}`; }

  // src: Decoded (views "full" and "truth") or PV (view "bot"). Returns the sentence after the actor's name.
  function stepText(src, i, opts) {
    const view = (opts && opts.view) || "full";
    if (view === "bot" && !src.pub) throw new Error("the bot view reads the public view only");
    const s = src.steps[i];
    const name = q => COLOR_NAMES[q];
    const tagTruth = view === "truth" && s.hidden;
    switch (s.kind) {
      case K.SETTLE: {
        let out = "builds a settlement";
        const got = s.h && s.h[s.a] && s.h[s.a].some(x => x > 0) && s.free ? deltaText(s.h[s.a]) : "";
        if (got) out += ` · starting cards ${got}`;
        return out;
      }
      case K.ROAD: return "builds a road" + (s.free && s.r > 0 ? " (free)" : "");
      case K.CITY: return "upgrades a settlement to a city";
      case K.ROLL: {
        const sum = s.dice[0] + s.dice[1];
        if (sum === 7) return "rolls 7 · the robber moves";
        const h = s.h;
        const parts = [];
        if (h) for (let q = 0; q < src.n; q++) { const t = deltaText(h[q]); if (t) parts.push(`${name(q)} ${t}`); }
        return parts.length ? `rolls ${sum} · ${parts.join("; ")}` : `rolls ${sum} · nobody produces`;
      }
      case K.END: return "ends the turn";
      case K.BUY:
        if (view === "bot" && s.hidden) return "buys a development card";
        if (tagTruth) return `buys a development card · type hidden from our bot (${DEV[s.v]})`;
        return `buys a development card (${DEV[s.v]})`;
      case K.ROBBER: {
        const tile = Math.floor(s.v / 100);
        let out = `moves the robber to ${tileName(src.board, tile)}`;
        if (s.victim < 0) return out;
        if (view === "bot" && s.hidden) return `${out} and steals a card from ${name(s.victim)}`;
        if (s.stolen == null || s.stolen < 0) return `${out}; ${name(s.victim)} has no cards to steal`;
        if (tagTruth) return `${out} and steals a card from ${name(s.victim)} · hidden from our bot (it was ${RES[s.stolen]})`;
        return `${out} and steals ${RES[s.stolen]} from ${name(s.victim)}`;
      }
      case K.TRADE: {
        const give = Math.floor(s.v / 100), ratio = Math.floor(s.v / 10) % 10, get = s.v % 10;
        return `trades ${ratio} ${RES[give]} for 1 ${RES[get]} (${ratio}:1)`;
      }
      case K.DISCARD1:
        if (view === "bot" && s.hidden) return "discards a card";
        if (tagTruth) return `discards a card · hidden from our bot (${RES[s.v]})`;
        return `discards ${RES[s.v]}`;
      case K.KNIGHT: return "plays a Knight";
      case K.MONO: {
        const got = s.h ? s.h[s.a][s.v] : 0;
        return `plays Monopoly on ${RES[s.v]} and collects ${got}`;
      }
      case K.YOP: {
        const cards = Array.isArray(s.v) ? s.v : [s.v];
        return `plays Year of Plenty: ${cards.map(r => RES[r]).join(" and ")}`;
      }
      case K.RB: return "plays Road Building";
      case K.DISCARD: {
        if (view === "bot" && s.hidden) {
          const k = s.hs ? -s.hs[s.a] : 0;
          return `discards ${plural(k, "card")}`;
        }
        const tot = s.v.reduce((x, y) => x + y, 0);
        if (tagTruth) return `discards ${plural(tot, "card")} · hidden from our bot (${resList(s.v)})`;
        return `discards ${plural(tot, "card")}: ${resList(s.v)}`;
      }
      default: return KINDS[s.kind] || "acts";
    }
  }

  // ---- view model (8.6) -----------------------------------------------------------------------------------------
  function piecesAt(src, k) {
    const bld = new Map(), roads = new Map();
    for (let i = 0; i < k; i++) {
      const add = src.steps[i].add;
      if (!add) continue;
      for (const [type, seat, where] of add) {
        if (type === "r") roads.set(where, seat); else bld.set(where, [type, seat]);
      }
    }
    const last = k > 0 ? src.steps[k - 1] : null;
    return {
      bld: [...bld.entries()].sort((a, b) => a[0] - b[0]).map(([node, [type, seat]]) => [node, type, seat]),
      roads: [...roads.entries()].sort((a, b) => a[0] - b[0]),
      fresh: last && last.add ? last.add.map(x => [x[0], x[2]]) : []
    };
  }

  function diceAt(src, k) {
    for (let i = k - 1; i >= 0; i--) {
      const s = src.steps[i];
      if (s.kind === K.ROLL) return s.dice.slice();
      if (s.kind === K.END) return null;
    }
    return null;
  }

  function unsureOf(u) { return RES.filter((_, r) => u & (1 << r)); }

  // mode: "full" (full-information game, src = D), "truth" (Real hands, src = D), "both" (Real + bot's guess,
  // src = D, bv = BV), "bot" (Bot's view only, src = PV, bv = {pub} and never BV.p).
  function viewModel(src, bv, k, mode) {
    const n = src.n, N = src.N;
    k = Math.max(0, Math.min(N, k | 0));
    if ((mode === "bot") !== !!src.pub) throw new Error(`mode ${mode} needs ${mode === "bot" ? "the public view" : "the decoded game"}`);
    const view = mode === "bot" ? "bot" : mode === "full" ? "full" : "truth";
    const sn = src.snap(k);
    const st = k > 0 ? src.steps[k - 1] : null;
    const me = mode === "bot" ? src.me : -1;
    const vm = { id: src.id, k, N, mode, n, ours: src.ours.slice(), actor: st ? st.a : null, rolls: sn.rolls,
      turns: sn.turns, dice: diceAt(src, k), text: st ? stepText(src, k - 1, { view }) : null,
      kind: st ? KINDS[st.kind] : null, hidden: st ? st.hidden : 0, robber: sn.robber, bank: sn.bank,
      deck: sn.deck, pieces: piecesAt(src, k), players: [], belief: null };
    for (let s = 0; s < n; s++) {
      const P = sn.P[s];
      const known = mode !== "bot" || s === me;
      const size = known ? P.hand.reduce((x, y) => x + y, 0) : P.size;
      const devCount = known ? P.held.reduce((x, y) => x + y, 0) : P.devCount;
      const pl = { seat: s, ours: src.ours.includes(s), size, devCount, pvp: P.pvp, lr: P.lr, LR: P.LR, LA: P.LA,
        played: P.played.slice() };
      if (known) { pl.vp = P.vp; pl.hand = P.hand.slice(); pl.held = P.held.slice(); }
      if (st) {
        if (mode !== "bot") {
          if (st.h && st.h[s].some(x => x)) pl.delta = st.h[s].slice();
        } else if (s === me && st.h && st.h[s].some(x => x)) {
          pl.delta = st.h[s].slice();
        } else if (st.hs && st.hs[s]) {
          pl.sizeDelta = st.hs[s];
        }
      }
      vm.players.push(pl);
    }
    if (bv && (mode === "both" || mode === "bot")) {
      const pb = bv.pub(k);
      const pool = pb.pool, poolTot = pool.reduce((x, y) => x + y, 0);
      const opps = pb.opps.map(o => {
        const pl = vm.players[o.seat];
        const law = o.vpLaw;
        const b = { seat: o.seat, size: pl.size, exact: o.exact, own: o.own ? o.own.slice() : null,
          E: o.E.slice(), u: o.u, unsure: unsureOf(o.u), top: o.top, calp: o.calp, vpLaw: law.slice(),
          vpMean: law.reduce((x, w, v) => x + w * v, 0), devCount: pl.devCount };
        if (mode === "both") {
          b.p = bv.p(k, o.seat);
          b.real = pl.hand.slice();
          b.differs = b.E.map((e, r) => Math.abs(e - b.real[r]) >= 0.5);
          b.realVpCards = pl.held[1];
        }
        return b;
      });
      vm.belief = { allExact: opps.every(o => o.exact), nh: pb.nh, pool: pool.slice(), poolVp: pool[1], poolTot,
        opps };
    }
    return vm;
  }

  // ---- derived index fields (4.1), checked by the validator and #selftest ----------------------------------------
  function indexFields(D) {
    const fin = D.snap(D.N);
    const vps = fin.P.map(p => p.vp), pvps = fin.P.map(p => p.pvp);
    const last = D.N > 0 ? D.steps[D.N - 1].a : 0;
    let win = vps[last] >= VPS_TO_WIN ? last : vps.indexOf(Math.max(...vps));
    const opp = [];
    for (let s = 0; s < D.n; s++) if (!D.ours.includes(s)) opp.push(s);
    const upto = D.lastEnd >= 0 ? D.lastEnd + 1 : D.N;
    let def = 0, defp = 0;
    for (let k = 0; k <= upto; k++) {
      const sn = D.snap(k);
      const bo = Math.max(...opp.map(s => sn.P[s].vp)), bu = Math.max(...D.ours.map(s => sn.P[s].vp));
      const bop = Math.max(...opp.map(s => sn.P[s].pvp)), bup = Math.max(...D.ours.map(s => sn.P[s].pvp));
      def = Math.max(def, bo - bu); defp = Math.max(defp, bop - bup);
    }
    const dev = D.steps.filter(s => s.kind === K.BUY && D.ours.includes(s.a)).length;
    return { win, vps, pvps, turns: fin.turns, rolls: fin.rolls, acts: D.N, hvp: pvps[win] < VPS_TO_WIN ? 1 : 0,
      hv: vps[win] - pvps[win], dev, lr: fin.P.findIndex(p => p.LR), la: fin.P.findIndex(p => p.LA), def, defp };
  }

  // ---- index (4.1) ------------------------------------------------------------------------------------------------
  function parseIndex(doc) {
    const cols = doc.cols, at = {};
    cols.forEach((c, i) => { at[c] = i; });
    const tests = doc.tests || {};
    const order = Object.keys(tests).sort((a, b) => TEST_ORDER.indexOf(a) - TEST_ORDER.indexOf(b));
    const rows = doc.rows.map((r, idx) => {
      const lineup = r[at.lineup];
      const ours = [];
      for (let s = 0; s < lineup.length; s++) if (lineup[s] === "c") ours.push(s);
      const vps = r[at.vps], pvps = r[at.pvps];
      const win = r[at.win];
      const oppSeats = [];
      for (let s = 0; s < lineup.length; s++) if (lineup[s] !== "c") oppSeats.push(s);
      const ourVp = Math.max(...ours.map(s => vps[s])), oppVp = Math.max(...oppSeats.map(s => vps[s]));
      const t = r[at.t], g = r[at.g], bk = r[at.bk] || null;
      const meta = tests[t] || {};
      return { idx, t, g, id: `${t}-${g}`, seed: r[at.seed], d: r[at.d], lineup, n: lineup.length, ours, win,
        won: lineup[win] === "c", vps, pvps, turns: r[at.turns], rolls: r[at.rolls], acts: r[at.acts],
        hvp: r[at.hvp], hv: r[at.hv], dev: r[at.dev], lr: r[at.lr], la: r[at.la], def: r[at.def], defp: r[at.defp],
        bk, ourVp, oppVp, margin: ourVp - oppVp, counted: !!(meta.info && meta.info.mode === "counted"),
        fmt: meta.fmt || "", ver: meta.catanatron || "", testRank: TEST_ORDER.indexOf(t),
        hiddenEvents: bk ? (bk[8] || 0) + (bk[9] || 0) + (bk[10] || 0) : -1 };
    });
    const byId = new Map(rows.map(r => [r.id, r]));
    const byDeal = new Map();
    for (const r of rows) { if (!byDeal.has(r.d)) byDeal.set(r.d, []); byDeal.get(r.d).push(r); }
    return { v: doc.v, built: doc.built, shardSize: doc.shardSize || 100, geo: doc.geo, tests, order, rows, byId,
      byDeal, dealExceptions: doc.dealExceptions || [] };
  }

  // ---- filters, deep links (4.3, 8.2) -----------------------------------------------------------------------------
  function defaultFilters() {
    return { tests: [], fmt: [], info: null, opp: [], result: null, seat: null, hv: null, wp: null, behind: null,
      behindPub: false, rmin: null, rmax: null, ver: null, q: "", sort: "game", desc: false };
  }

  function testIdOf(s, index) {
    const u = String(s).toUpperCase();
    return index && index.tests && index.tests[u] ? u : null;
  }

  function parseFilterTokens(parts, index) {
    const f = defaultFilters(), bad = [];
    let any = false;
    for (const tok of parts) {
      if (!tok) continue;
      const low = tok.toLowerCase();
      let m;
      if (low === "all") { f.tests = []; any = true; }
      else if (tok.split(".").every(x => testIdOf(x, index))) { f.tests = tok.split(".").map(x => testIdOf(x, index)); any = true; }
      else if (low === "won" || low === "lost") { f.result = low; any = true; }
      else if (low === "pub" || low === "full") { f.info = low; any = true; }
      else if (low === "hvo" || low === "hvx") { f.hv = low; any = true; }
      else if ((m = /^wp([0-9]+)$/.exec(low))) { f.wp = +m[1]; any = true; }
      else if ((m = /^b(p?)([0-9]+)$/.exec(low))) { f.behind = +m[2]; f.behindPub = m[1] === "p"; any = true; }
      else if ((m = /^r([0-9]*)-([0-9]*)$/.exec(low))) {
        f.rmin = m[1] === "" ? null : +m[1]; f.rmax = m[2] === "" ? null : +m[2]; any = true;
      }
      else if ((m = /^s([1-4])$/.exec(low))) { f.seat = +m[1] - 1; any = true; }
      else if ((m = /^o-([a-z]+)(-d)?$/.exec(low)) && SORT_KEYS.includes(m[1])) { f.sort = m[1]; f.desc = !!m[2]; any = true; }
      else if ((m = /^fm-([0-9a-z.]+)$/.exec(low)) && m[1].split(".").every(x => FMT_CODES[x])) {
        f.fmt = m[1].split(".").map(x => FMT_CODES[x]); any = true;
      }
      else if ((m = /^vs-([vasfb.]+)$/.exec(low))) { f.opp = m[1].split(".").filter(Boolean); any = true; }
      else if (VERSION_CODES[low]) { f.ver = VERSION_CODES[low]; any = true; }
      else if ((m = /^q-([a-z0-9.]+)$/i.exec(tok))) { f.q = m[1].replace(/\./g, "-"); any = true; }
      else bad.push(tok);
    }
    return { filters: f, bad, any };
  }

  function parseHash(hash, index) {
    const raw = String(hash || "").replace(/^#/, "");
    let token = raw;
    try { token = decodeURIComponent(raw); } catch (e) { token = raw; }
    if (!token) return { view: "picker", filters: null, message: null };
    if (token.toLowerCase() === "selftest") return { view: "selftest", filters: null, message: null };
    const gm = /^([A-Za-z]+[0-9]+)-([0-9]+)(?:\.([0-9]+))?$/.exec(token);
    if (gm && testIdOf(gm[1], index)) {
      const t = testIdOf(gm[1], index), g = +gm[2];
      const meta = index.tests[t];
      const games = meta.games;
      const id = `${t}-${g}`;
      if (g >= games || (index.byId && !index.byId.has(id))) {
        const f = defaultFilters(); f.tests = [t];
        return { view: "picker", filters: f, message: `There is no game ${id} (${t} has games 0–${games - 1})` };
      }
      let k = gm[3] === undefined ? null : +gm[3], message = null;
      const row = index.byId ? index.byId.get(id) : null;
      if (k !== null && row && k > row.acts) {
        message = `Game ${id} has ${row.acts} actions; showing the end`;
        k = row.acts;
      }
      return { view: "game", id, t, g, k, filters: null, message };
    }
    const { filters, bad, any } = parseFilterTokens(token.split("~"), index);
    let message = null;
    if (bad.length) message = `Unknown link part '${bad[0]}'; ${any ? "ignored it" : "showing all games"}`;
    return { view: "picker", filters: any ? filters : defaultFilters(), message, explicit: true };
  }

  function filterToken(f) {
    const d = defaultFilters(), out = [];
    out.push(f.tests && f.tests.length ? f.tests.join(".") : "all");
    if (f.result) out.push(f.result);
    if (f.info) out.push(f.info);
    if (f.hv) out.push(f.hv);
    if (f.wp != null) out.push(`wp${f.wp}`);
    if (f.behind != null) out.push(`b${f.behindPub ? "p" : ""}${f.behind}`);
    if (f.rmin != null || f.rmax != null) out.push(`r${f.rmin == null ? "" : f.rmin}-${f.rmax == null ? "" : f.rmax}`);
    if (f.seat != null) out.push(`s${f.seat + 1}`);
    if (f.fmt && f.fmt.length) {
      const inv = {};
      for (const [c, v] of Object.entries(FMT_CODES)) inv[v] = c;
      out.push(`fm-${f.fmt.map(x => inv[x]).join(".")}`);
    }
    if (f.opp && f.opp.length) out.push(`vs-${f.opp.join(".")}`);
    if (f.ver) out.push(Object.keys(VERSION_CODES).find(c => VERSION_CODES[c] === f.ver));
    if (f.q) {
      const q = String(f.q).replace(/-/g, ".").replace(/[^A-Za-z0-9.]/g, "");
      if (q) out.push(`q-${q}`);
    }
    if (f.sort && (f.sort !== d.sort || f.desc)) out.push(`o-${f.sort}${f.desc ? "-d" : ""}`);
    return out.join("~");
  }

  // Outcome-selecting filters (4.5): the win rate is not meaningful when one of them is on.
  function selectsByOutcome(f) { return !!(f.result || f.hv || f.wp != null); }

  function countFilters(f) {
    let c = 0;
    if (f.tests.length) c++;
    if (f.fmt.length) c++;
    if (f.info) c++;
    if (f.opp.length) c++;
    if (f.result) c++;
    if (f.seat != null) c++;
    if (f.hv) c++;
    if (f.wp != null) c++;
    if (f.behind != null) c++;
    if (f.rmin != null || f.rmax != null) c++;
    if (f.ver) c++;
    if (f.q) c++;
    return c;
  }

  // Returns {rows, seedNote}. seedNote: the deal shared by a seed search.
  function applyFilters(index, f) {
    const tests = f.tests.length ? new Set(f.tests) : null;
    const fmt = f.fmt.length ? new Set(f.fmt) : null;
    const q = String(f.q || "").trim();
    let qid = null, qnum = null, seedDeal = null;
    if (q) {
      const m = /^([A-Za-z]+[0-9]+)-([0-9]+)$/.exec(q);
      if (m) qid = `${m[1].toUpperCase()}-${+m[2]}`;
      else if (/^[0-9]+$/.test(q)) qnum = +q;
    }
    const seedRows = qnum != null ? index.rows.filter(r => r.seed === qnum) : [];
    if (seedRows.length) seedDeal = seedRows[0].d;
    const out = index.rows.filter(r => {
      if (tests && !tests.has(r.t)) return false;
      if (fmt && !fmt.has(r.fmt)) return false;
      if (f.info === "pub" && !r.counted) return false;
      if (f.info === "full" && r.counted) return false;
      if (f.opp.length && !f.opp.some(o => r.lineup.includes(o))) return false;
      if (f.result === "won" && !r.won) return false;
      if (f.result === "lost" && r.won) return false;
      if (f.seat != null && !r.ours.includes(f.seat)) return false;
      if (f.hv === "hvo" && !(r.hvp && r.won)) return false;
      if (f.hv === "hvx" && !(r.hvp && !r.won)) return false;
      if (f.wp != null && !(r.pvps[r.win] <= f.wp)) return false;
      if (f.behind != null && (f.behindPub ? r.defp : r.def) < f.behind) return false;
      if (f.rmin != null && r.rolls < f.rmin) return false;
      if (f.rmax != null && r.rolls > f.rmax) return false;
      if (f.ver && r.ver !== f.ver) return false;
      if (q) {
        if (qid) { if (r.id !== qid) return false; }
        else if (qnum != null) { if (!(r.g === qnum || r.seed === qnum || (seedDeal != null && r.d === seedDeal))) return false; }
        else if (!r.id.toLowerCase().includes(q.toLowerCase())) return false;
      }
      return true;
    });
    const key = sortKeyFn(f.sort);
    out.sort((a, b) => {
      const ka = key(a), kb = key(b);
      let c = ka < kb ? -1 : ka > kb ? 1 : 0;
      if (f.desc) c = -c;
      if (c === 0) c = a.testRank - b.testRank || a.g - b.g;
      return c;
    });
    let seedNote = null;
    if (seedDeal != null) {
      const same = index.byDeal.get(seedDeal) || [];
      seedNote = { seed: qnum, deal: seedDeal, count: same.length, ids: same.map(r => r.id) };
    }
    return { rows: out, seedNote };
  }

  function sortKeyFn(key) {
    switch (key) {
      case "seed": return r => r.seed;
      case "rolls": return r => r.rolls;
      case "turns": return r => r.turns;
      case "ourvp": return r => r.ourVp;
      case "oppvp": return r => r.oppVp;
      case "margin": return r => r.margin;
      case "behind": return r => r.def;
      case "dev": return r => r.dev;
      case "meanp": return r => (r.bk ? r.bk[1] : 2);
      case "exact": return r => (r.bk ? r.bk[0] : 2);
      case "hidden": return r => r.hiddenEvents;
      default: return r => r.testRank * 100000 + r.g;
    }
  }

  function wilson(wins, n, z) {
    if (!n) return [0, 0];
    z = z || 1.959963984540054;
    const p = wins / n, z2 = z * z, d = 1 + z2 / n;
    const c = (p + z2 / (2 * n)) / d;
    const m = (z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / d;
    return [Math.max(0, c - m), Math.min(1, c + m)];
  }

  // Which log file holds game g of test t (4.2), for "Check this yourself".
  function logFileOf(meta, g) {
    const f = (meta.files || []).find(x => g >= x.from && g < x.to);
    return f ? { path: f.path, line: g - f.from + 1, sha256: f.sha256, manifest: f.manifest } : null;
  }

  const RC = {
    VERSION: 2, COLORS, COLOR_NAMES, RES, DEV, KINDS, K, OPP, TEST_ORDER, DEV_DECK, VPS_TO_WIN, HIDDEN, SORT_KEYS,
    FMT_CODES, VERSION_CODES,
    shardPath, bytesKind, parseJsonBytes, parseJsonBytesEx, fetchJson, fetchJsonEx, parseIndex, parseHash,
    filterToken, parseFilterTokens, defaultFilters, applyFilters, countFilters, selectsByOutcome, parseBoard,
    tileName, decodeGame, cloneDecoded, publicView, botView, stepText, viewModel, piecesAt, diceAt, indexFields,
    wilson, logFileOf, unsureOf
  };
  if (typeof module === "object" && module.exports) module.exports = RC; else root.ReplayCore = RC;
})(typeof self !== "undefined" ? self : this);
