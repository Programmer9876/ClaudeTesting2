#!/usr/bin/env node
// check_replay_bundle.mjs: node validator of a Replay Archive bundle (docs/designs/replay_archive.md, section 9.1).
//
//   node scripts/check_replay_bundle.mjs --bundle $S/replay_archive [--repo /home/user/ClaudeTesting2] \
//        [--truth $S/replay_archive_work/truth] [--core scripts/replay_core.js] \
//        [--template scripts/replay_archive_template.html] [--expect-all] [--bundle-only [--work DIR]]
//
// It loads the SAME replay_core.js the page inlines, decodes every game of the bundle and compares the final
// state with the ORIGINAL action logs (read here with zlib, independent of the exporter), the index rows with the
// decoded games and the results files, the per-position truth of the sampled games (positions, belief), the
// bot-view invariants, a swap test (the bot view never reads hidden fields), the page's colour contrast, and the
// parsers. Node >= 18, no npm packages. Exit 0 when everything passes, 1 otherwise.
//
// --bundle-only: validate a partial bundle (for example two games written by one exporter worker): no index.html
// needed, no completeness or results-file checks; without index.json.gz the geometry and index rows are read
// from <work>/parts (--work).
// --expect-all: also require all 18 tests (9,600 rows).
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import crypto from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const opt = { bundle: null, repo: path.dirname(HERE), truth: null, core: path.join(HERE, "replay_core.js"),
  template: path.join(HERE, "replay_archive_template.html"), work: null, bundleOnly: false, expectAll: false,
  quiet: false };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  const val = () => { if (i + 1 >= argv.length) usage(`${a} needs a value`); return argv[++i]; };
  if (a === "--bundle") opt.bundle = val();
  else if (a === "--repo") opt.repo = val();
  else if (a === "--truth") opt.truth = val();
  else if (a === "--core") opt.core = val();
  else if (a === "--template") opt.template = val();
  else if (a === "--work") opt.work = val();
  else if (a === "--bundle-only") opt.bundleOnly = true;
  else if (a === "--expect-all") opt.expectAll = true;
  else if (a === "--quiet") opt.quiet = true;
  else if (a === "-h" || a === "--help") usage(null);
  else usage(`unknown argument ${a}`);
}
function usage(msg) {
  if (msg) console.error(msg);
  console.error("usage: node check_replay_bundle.mjs --bundle DIR [--repo DIR] [--truth DIR] [--core FILE] " +
    "[--template FILE] [--expect-all] [--bundle-only [--work DIR]]");
  process.exit(msg ? 2 : 0);
}
if (!opt.bundle) usage("--bundle is required");
opt.bundle = path.resolve(opt.bundle);
if (!opt.truth) {
  const guess = path.join(path.dirname(opt.bundle), "replay_archive_work", "truth");
  if (fs.existsSync(guess)) opt.truth = guess;
}
if (!opt.work) {
  const guess = path.join(path.dirname(opt.bundle), "replay_archive_work");
  if (fs.existsSync(guess)) opt.work = guess;
}

const require = createRequire(import.meta.url);
const RC = require(path.resolve(opt.core));
const { K, RES, DEV } = RC;

const problems = [];
let problemCount = 0;
function bad(msg) { problemCount++; if (problems.length < 20) problems.push(msg); }
function need(cond, msg) { if (!cond) bad(msg); return cond; }
const lines = [];
function say(msg) { lines.push(msg); if (!opt.quiet) console.log(msg); }
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const gunzip = u8 => new Uint8Array(zlib.gunzipSync(u8));
async function readJson(file) {
  return RC.parseJsonBytes(new Uint8Array(fs.readFileSync(file)), gunzip);
}
const RES_UP = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];
const DEV_UP = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"];
const MB = 1024 * 1024;

// ---------------------------------------------------------------------------------------------------------------
// 1. bundle limits
function listFiles(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...listFiles(p)); else out.push(p);
  }
  return out;
}
const files = listFiles(opt.bundle);
const total = files.reduce((s, f) => s + fs.statSync(f).size, 0);
need(files.length <= 255, `bundle has ${files.length} files (limit 255)`);
need(total <= 64 * MB, `bundle is ${(total / MB).toFixed(2)} MB (limit 64 MB)`);
for (const f of files) {
  const sz = fs.statSync(f).size;
  need(sz <= 15 * MB, `${path.relative(opt.bundle, f)} is ${(sz / MB).toFixed(2)} MB (limit 15 MB)`);
  need(!f.endsWith(".js") && !f.endsWith(".mjs"), `${path.relative(opt.bundle, f)}: no .js file may be published`);
}
say(`bundle: ${files.length} files, ${total.toLocaleString("en-US")} bytes, largest ` +
  `${Math.max(...files.map(f => fs.statSync(f).size)).toLocaleString("en-US")} bytes`);

const pagePath = path.join(opt.bundle, "index.html");
const coreText = fs.readFileSync(opt.core, "utf8");
if (fs.existsSync(pagePath)) {
  const html = fs.readFileSync(pagePath, "utf8");
  need(/<title>[^<]+<\/title>/.test(html.slice(0, 8192)), "index.html: no <title> in the first 8 KB");
  need(!/<\s*(html|head|body)[\s>]/i.test(html), "index.html must not contain <html>, <head> or <body>");
  need(!/<script[^>]*\bsrc\s*=/i.test(html), "index.html must not load scripts with <script src>");
  // 2. inlined core
  const B = "/* BEGIN replay_core.js */", E = "/* END replay_core.js */";
  const a = html.indexOf(B), b = html.indexOf(E);
  if (need(a >= 0 && b > a, "index.html lacks the replay_core.js markers")) {
    need(html.slice(a + B.length, b) === coreText, "index.html: the inlined core differs from --core byte for byte");
  }
  if (fs.existsSync(opt.template)) {
    const tpl = fs.readFileSync(opt.template, "utf8");
    const ta = tpl.indexOf(B), tb = tpl.indexOf(E);
    need(ta >= 0 && tb > ta && tpl.slice(0, ta + B.length) === html.slice(0, a + B.length) &&
      tpl.slice(tb) === html.slice(b), "index.html is not the template with the core inlined");
  }
  say("page: index.html has a title, no html/head/body tags, no script src; the inlined core equals " +
    path.relative(process.cwd(), opt.core));
} else if (!opt.bundleOnly) {
  bad("index.html is missing (run scripts/replay_export.py --merge-only to write it)");
}

// ---------------------------------------------------------------------------------------------------------------
// index (or, --bundle-only without an index, the exporter's parts)
let indexDoc = null;
const indexPath = path.join(opt.bundle, "index.json.gz");
if (fs.existsSync(indexPath)) {
  indexDoc = await readJson(indexPath);
} else if (opt.bundleOnly && opt.work && fs.existsSync(path.join(opt.work, "parts"))) {
  const parts = fs.readdirSync(path.join(opt.work, "parts"));
  let geo = null;
  const rows = [], tests = {};
  for (const p of parts.filter(x => x.endsWith(".checks.json")).sort()) {
    const c = JSON.parse(fs.readFileSync(path.join(opt.work, "parts", p), "utf8"));
    geo = geo || c.geo;
    tests[c.test] = tests[c.test] || { info: c.info, games: 0, files: [], partial: true };
  }
  for (const p of parts.filter(x => x.endsWith(".rows.json")).sort()) {
    rows.push(...JSON.parse(fs.readFileSync(path.join(opt.work, "parts", p), "utf8")).rows);
  }
  for (const r of rows) tests[r[0]].games = Math.max(tests[r[0]].games, r[1] + 1);
  indexDoc = { v: 2, shardSize: 100, geo, tests, rows,
    cols: ["t", "g", "seed", "d", "lineup", "win", "vps", "pvps", "turns", "rolls", "acts", "hvp", "hv", "dev", "lr",
      "la", "def", "defp", "bk"] };
  say(`index: none in the bundle; ${rows.length} rows read from ${path.join(opt.work, "parts")}`);
} else {
  bad("index.json.gz is missing" + (opt.bundleOnly ? " (and no --work parts to fall back on)" : ""));
}
if (!indexDoc) finish();
const index = RC.parseIndex(indexDoc);
const geo = index.geo;
const edgeIdx = new Map(geo.edges.map((e, i) => [`${e[0]}-${e[1]}`, i]));
const tileIdx = new Map(geo.tiles.map((t, i) => [t[0].join(","), i]));

// 3. index and provenance
const testsHere = index.order;
say(`index: ${index.rows.length.toLocaleString("en-US")} rows, tests ${testsHere.join(", ")}`);
if (opt.expectAll) {
  need(eq(testsHere, RC.TEST_ORDER), `expected all 18 tests, found ${testsHere.join(",")}`);
  need(index.rows.length === 9600, `expected 9,600 rows, found ${index.rows.length}`);
}
const rowsByTest = {};
for (const r of index.rows) (rowsByTest[r.t] = rowsByTest[r.t] || []).push(r);
function suiteOf(t) {
  for (const s of ["proof", "bench_1v1", "bench_multi"]) if (fs.existsSync(path.join(opt.repo, s, t, "logs"))) return s;
  return null;
}
const manifests = {};
function manifest(suite) {
  if (!manifests[suite]) {
    manifests[suite] = new Map();
    const p = path.join(opt.repo, suite, "MANIFEST.sha256");
    if (fs.existsSync(p)) {
      for (const line of fs.readFileSync(p, "utf8").split("\n")) {
        const m = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line.trim());
        if (m) manifests[suite].set(m[2], m[1]);
      }
    }
  }
  return manifests[suite];
}
for (const t of testsHere) {
  const meta = index.tests[t], rows = rowsByTest[t] || [];
  const gs = rows.map(r => r.g).sort((a, b) => a - b);
  if (!opt.bundleOnly) {
    need(gs.length === meta.games && gs.every((g, i) => g === i), `${t}: index games are not 0..${meta.games - 1} exactly once`);
    const suite = suiteOf(t);
    if (need(suite, `${t}: no log directory in ${opt.repo}`)) {
      // results files, read here directly
      const rdir = path.join(opt.repo, suite, t, "results");
      let games = 0, wins = 0;
      const winner = new Map();
      for (const f of fs.readdirSync(rdir).filter(x => x.endsWith(".json")).sort()) {
        const d = JSON.parse(fs.readFileSync(path.join(rdir, f), "utf8"));
        games += d.games; wins += d.wins;
        for (const r of d.results || []) winner.set(r.game, r.winner_seat);
      }
      need(games === rows.length, `${t}: results files hold ${games} games, the index ${rows.length}`);
      need(wins === rows.filter(r => r.won).length, `${t}: results files count ${wins} wins, the index ${rows.filter(r => r.won).length}`);
      for (const r of rows) need(winner.get(r.g) === r.win, `${t}-${r.g}: winner seat ${r.win}, results ${winner.get(r.g)}`);
      if (meta.published) need(meta.published.games === games && meta.published.wins === wins, `${t}: published totals differ from the results files`);
      const man = manifest(suite);
      for (const f of meta.files || []) {
        const rel = path.relative(path.join(opt.repo, suite), path.join(opt.repo, f.path));
        need(man.get(rel) === f.sha256, `${f.path}: sha256 differs from ${suite}/MANIFEST.sha256`);
        const h = crypto.createHash("sha256").update(fs.readFileSync(path.join(opt.repo, f.path))).digest("hex");
        need(h === f.sha256, `${f.path}: file sha256 ${h.slice(0, 12)} differs from the index`);
      }
    }
    for (let k = 0; k < (meta.shards || 0); k++) {
      const sp = path.join(opt.bundle, "shards", `${t}_${String(k).padStart(2, "0")}.json.gz`);
      need(fs.existsSync(sp), `${t}: shard ${path.basename(sp)} is missing`);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// original logs (read independently of the exporter)
function readLogRecords(t, wanted) {
  const suite = suiteOf(t);
  const out = new Map();
  if (!suite) return out;
  const dir = path.join(opt.repo, suite, t, "logs");
  for (const f of fs.readdirSync(dir).filter(x => x.endsWith(".jsonl.gz")).sort()) {
    const m = /_g(\d+)-(\d+)\.jsonl\.gz$/.exec(f);
    if (!m) continue;
    const a = +m[1], b = +m[2];
    let any = false;
    for (let g = a; g < b; g++) if (wanted.has(g)) { any = true; break; }
    if (!any) continue;
    const text = zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString("utf8");
    let g = a;
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      if (wanted.has(g)) {
        const rec = JSON.parse(line);
        need(rec.game === g, `${f}: line ${g - a + 1} holds game ${rec.game}, expected ${g}`);
        out.set(g, rec);
      }
      g++;
    }
  }
  return out;
}

function piecesBySeat(D, k) {
  const pc = RC.piecesAt(D, k), out = [];
  for (let s = 0; s < D.n; s++) out.push({ settlements: [], cities: [], roads: [] });
  for (const [node, type, seat] of pc.bld) out[seat][type === "s" ? "settlements" : "cities"].push(node);
  for (const [edge, seat] of pc.roads) out[seat].roads.push(edge);
  for (const o of out) { o.settlements.sort((a, b) => a - b); o.cities.sort((a, b) => a - b); o.roads.sort((a, b) => a - b); }
  return out;
}

// 4. compare the decoded end with the log's own final state
function checkFinal(D, rec) {
  const id = D.id, fin = rec.final, st = fin.state, sn = D.snap(D.N);
  const pcs = piecesBySeat(D, D.N);
  need(D.N === rec.actions.length && D.N === fin.num_actions, `${id}: ${D.N} steps, the log has ${rec.actions.length}`);
  st.players.forEach((p, s) => {
    const P = sn.P[s];
    need(eq(P.hand, RES_UP.map(r => p.resources[r])), `${id} seat ${s}: hand ${P.hand} vs log ${JSON.stringify(p.resources)}`);
    need(eq(P.held, DEV_UP.map(d => p.dev_cards[d])), `${id} seat ${s}: dev held ${P.held}`);
    need(eq(P.played, DEV_UP.map(d => p.dev_played[d] || 0)), `${id} seat ${s}: dev played ${P.played}`);
    need(P.vp === p.vp && P.pvp === p.public_vp, `${id} seat ${s}: VP ${P.vp}/${P.pvp} vs ${p.vp}/${p.public_vp}`);
    need(P.lr === p.longest_road_length, `${id} seat ${s}: road length ${P.lr} vs ${p.longest_road_length}`);
    need(P.LR === !!p.has_longest_road && P.LA === !!p.has_largest_army, `${id} seat ${s}: titles`);
    need(eq(pcs[s].settlements, [...p.settlements].sort((a, b) => a - b)), `${id} seat ${s}: settlements`);
    need(eq(pcs[s].cities, [...p.cities].sort((a, b) => a - b)), `${id} seat ${s}: cities`);
    const roads = p.roads.map(e => edgeIdx.get(`${Math.min(e[0], e[1])}-${Math.max(e[0], e[1])}`)).sort((a, b) => a - b);
    need(eq(pcs[s].roads, roads), `${id} seat ${s}: roads`);
    need(fin.vps[s] === P.vp, `${id} seat ${s}: final vps ${fin.vps[s]} vs ${P.vp}`);
  });
  need(sn.robber === tileIdx.get(st.robber.join(",")), `${id}: robber ${sn.robber} vs ${st.robber}`);
  need(eq(sn.bank, RES_UP.map(r => st.bank[r])), `${id}: bank ${sn.bank}`);
  need(sn.deck === st.dev_deck_left, `${id}: deck ${sn.deck} vs ${st.dev_deck_left}`);
  need(sn.turns === fin.turns && sn.turns === st.turn, `${id}: turns ${sn.turns} vs ${fin.turns}`);
  need(RC.indexFields(D).win === fin.winner_seat, `${id}: winner`);
}

function checkIndexRow(D, row) {
  const f = RC.indexFields(D);
  for (const k of ["win", "turns", "rolls", "acts", "hvp", "hv", "dev", "lr", "la", "def", "defp"]) {
    need(f[k] === row[k], `${D.id}: index ${k} ${row[k]}, decoded ${f[k]}`);
  }
  need(eq(f.vps, row.vps) && eq(f.pvps, row.pvps), `${D.id}: index VPs`);
}

// 5. per-position truth (sampled games)
function checkTruthPositions(D, BV, tr) {
  const id = D.id;
  if (!need(tr.pos.length === D.N + 1, `${id}: truth has ${tr.pos.length} positions, expected ${D.N + 1}`)) return;
  for (let k = 0; k <= D.N; k++) {
    const T = tr.pos[k], sn = D.snap(k), pcs = piecesBySeat(D, k);
    T.P.forEach((p, s) => {
      const P = sn.P[s];
      const [vp, pvp, hand, held, played, sett, cities, roads, lrLen, LR, LA] = p;
      need(P.vp === vp && P.pvp === pvp && eq(P.hand, hand) && eq(P.held, held) && eq(P.played, played) &&
        P.lr === lrLen && P.LR === !!LR && P.LA === !!LA, `${id} position ${k} seat ${s}: state differs from the truth`);
      need(eq(pcs[s].settlements, [...sett].sort((a, b) => a - b)) && eq(pcs[s].cities, [...cities].sort((a, b) => a - b)) &&
        eq(pcs[s].roads, roads), `${id} position ${k} seat ${s}: pieces differ from the truth`);
    });
    need(sn.robber === tileIdx.get(T.robber.join(",")), `${id} position ${k}: robber`);
    need(eq(sn.bank, T.bank), `${id} position ${k}: bank ${sn.bank} vs ${T.bank}`);
    need(sn.deck === T.deck, `${id} position ${k}: deck`);
    need(sn.turns === T.turn, `${id} position ${k}: catanatron turn ${sn.turns} vs ${T.turn}`);
    if (BV && tr.bv) {
      const B = tr.bv[k], pb = BV.pub(k);
      need(pb.nh === B.nh, `${id} position ${k}: nh ${pb.nh} vs ${B.nh}`);
      need(eq(pb.pool, B.pool), `${id} position ${k}: pool ${pb.pool} vs ${B.pool}`);
      for (const o of pb.opps) {
        const b = B.opp[String(o.seat)];
        const rel = (x, y) => Math.abs(x - y) <= 0.003 * Math.max(Math.abs(x), Math.abs(y));
        need(o.exact === b.exact && eq(o.own, b.own) && o.u === b.u, `${id} position ${k} seat ${o.seat}: exact/own/u`);
        need(o.E.every((e, r) => Math.abs(e - b.E[r]) <= 0.005 + 1e-9), `${id} position ${k} seat ${o.seat}: E ${o.E} vs ${b.E}`);
        need(rel(BV.p(k, o.seat), b.p), `${id} position ${k} seat ${o.seat}: p ${BV.p(k, o.seat)} vs ${b.p}`);
        need(rel(o.top, b.top) && rel(o.calp, b.calp), `${id} position ${k} seat ${o.seat}: top/calp`);
        need(o.vpLaw.length === b.vpLaw.length && o.vpLaw.every((x, v) => Math.abs(x - b.vpLaw[v]) <= 0.001 + 1e-9),
          `${id} position ${k} seat ${o.seat}: vpLaw ${o.vpLaw} vs ${b.vpLaw}`);
      }
    }
  }
}

// 6. bot-view invariants (every counted game, every position)
function checkBotInvariants(D, BV, g) {
  const id = D.id, me = BV.me, bv = g.bv;
  for (const r of bv.q) {
    if (r[2] === 0) {
      const [, , , lp, lt, lc] = r;
      need([lp, lt, lc].every(x => Number.isInteger(x) && x >= 0) && lt <= lp, `${id}: q row ${JSON.stringify(r)}`);
    }
  }
  for (const r of bv.dv) need(r.slice(2).reduce((a, b) => a + b, 0) === 1000, `${id}: dv row ${JSON.stringify(r)} does not sum to 1000`);
  let pairs = 0, pSum = 0, calpSum = 0, pMin = 1, allExact = 0, nu = 0, puSum = 0, calpuSum = 0;
  for (let k = 0; k <= D.N; k++) {
    const sn = D.snap(k), pb = BV.pub(k);
    const played = [0, 0, 0, 0, 0];
    sn.P.forEach(p => p.played.forEach((x, d) => { played[d] += x; }));
    const pool = RC.DEV_DECK.map((x, d) => x - played[d] - sn.P[me].held[d]);
    need(eq(pb.pool, pool), `${id} position ${k}: pool ${pb.pool} vs ${pool} from the public state`);
    const oppHeld = pb.opps.reduce((a, o) => a + sn.P[o.seat].held.reduce((x, y) => x + y, 0), 0);
    need(pool.reduce((a, b) => a + b, 0) === sn.deck + oppHeld, `${id} position ${k}: sum of pool != deck + opponents' cards`);
    let every = true;
    for (const o of pb.opps) {
      const hand = sn.P[o.seat].hand, size = hand.reduce((a, b) => a + b, 0);
      need(Math.abs(o.E.reduce((a, b) => a + b, 0) - size) <= 0.03, `${id} position ${k} seat ${o.seat}: sum E vs size ${size}`);
      if (o.exact) need(eq(o.own, hand) && o.u === 0, `${id} position ${k} seat ${o.seat}: exact belief ${o.own} != real ${hand}`);
      const devCount = sn.P[o.seat].held.reduce((a, b) => a + b, 0);
      need(o.vpLaw.length === devCount + 1, `${id} position ${k} seat ${o.seat}: vpLaw over 0..${o.vpLaw.length - 1}, ${devCount} cards`);
      if (k > 0) {
        const p = BV.p(k, o.seat);
        pairs++; pSum += p; calpSum += o.calp; pMin = Math.min(pMin, p);
        if (!o.exact) { nu++; puSum += p; calpuSum += o.calp; }
      }
      every = every && o.exact;
    }
    if (k > 0 && every) allExact++;
  }
  const s = bv.sum, close = (x, y) => (x == null && y == null) || (x != null && y != null && Math.abs(x - y) <= 1e-3);
  need(close(allExact / Math.max(1, D.N), s.allExact), `${id}: allExact ${allExact / D.N} vs ${s.allExact}`);
  need(close(pSum / pairs, s.meanP), `${id}: meanP ${pSum / pairs} vs ${s.meanP}`);
  need(close(calpSum / pairs, s.calP), `${id}: calP ${calpSum / pairs} vs ${s.calP}`);
  need(Math.abs(pMin - s.minP) <= 0.003 * s.minP + 1e-9, `${id}: minP ${pMin} vs ${s.minP}`);
  need(close(nu ? puSum / nu : null, s.meanPu) && close(nu ? calpuSum / nu : null, s.calPu) && nu === s.nu,
    `${id}: uncertain-only means`);
  const c = bv.chk;
  need(c.stats === 1 && c.twin === 1 && c.same === 1 && (c.head === 1 || c.head == null) && c.audit[1] === 0,
    `${id}: checks ${JSON.stringify(c)}`);
}

// 7. swap test: mutate only hidden fields; the bot view must not change
const RES_WORDS = /\b(wood|brick|sheep|wheat|ore|Knight|Victory Point|Road Building|Year of Plenty|Monopoly)\b/;
// Everything the bot mode can draw from: the view model and log text at every position, and the public view's own
// data (its steps and every snapshot), so a hidden field kept inside PV but not yet rendered is caught too.
function botRender(PV, pubPart) {
  const out = [];
  for (let k = 0; k <= PV.N; k++) out.push(JSON.stringify(RC.viewModel(PV, pubPart, k, "bot")));
  for (let i = 0; i < PV.N; i++) out.push(RC.stepText(PV, i, { view: "bot" }));
  for (let k = 0; k <= PV.N; k++) out.push(JSON.stringify(PV.snap(k)));
  out.push(JSON.stringify(PV.steps));
  return out;
}
function swapTest(D, BV) {
  const me = BV.me, opps = BV.opps, n = D.n, W = n * 20;
  const pubPart = BV.publicPart();
  const base = botRender(RC.publicView(D, me), pubPart);
  // M1: at each position, one card from opponent A to B and one of another type back
  const M1 = RC.cloneDecoded(D);
  let moved = 0;
  for (let k = 0; k <= D.N; k++) {
    for (let x = 0; x < opps.length && true; x++) {
      const A = opps[x], Bs = opps[(x + 1) % opps.length];
      if (A === Bs) continue;
      const ha = k * W + A * 20, hb = k * W + Bs * 20;
      let r1 = -1, r2 = -1;
      for (let r = 0; r < 5 && r1 < 0; r++) if (M1.S[ha + r] > 0) r1 = r;
      for (let r = 0; r < 5 && r2 < 0; r++) if (r !== r1 && M1.S[hb + r] > 0) r2 = r;
      if (r1 >= 0 && r2 >= 0) { M1.S[ha + r1]--; M1.S[hb + r1]++; M1.S[hb + r2]--; M1.S[ha + r2]++; moved++; break; }
    }
  }
  // M2: relabel opponents' held development types (counts kept), with actual VP
  const M2 = RC.cloneDecoded(D);
  let relabelled = 0;
  for (let k = 0; k <= D.N; k++) {
    for (const j of opps) {
      const b = k * W + j * 20, held = Array.from(M2.S.subarray(b + 5, b + 10));
      const tot = held.reduce((a, c) => a + c, 0);
      if (!tot) continue;
      const nh = [0, 0, 0, 0, 0];
      nh[held[1] ? 0 : 1] = tot;             // all cards become Knights, or all VP cards if there were none
      for (let d = 0; d < 5; d++) M2.S[b + 5 + d] = nh[d];
      M2.S[b + 15] += nh[1] - held[1];
      relabelled++;
    }
  }
  // M3: stolen cards of hidden steals, hidden discards, opponents' drawn cards, bank inside discard runs
  const M3 = RC.cloneDecoded(D);
  let changed = 0;
  M3.steps.forEach((s, i) => {
    if (s.hidden === RC.HIDDEN.STEAL) {
      const r = s.stolen, r2 = (r + 1) % 5;
      s.stolen = r2; s.v = s.v - (r + 1) + (r2 + 1);
      if (s.h) { s.h[s.a][r]--; s.h[s.a][r2]++; s.h[s.victim][r]++; s.h[s.victim][r2]--; }
      changed++;
    } else if (s.hidden === RC.HIDDEN.DISCARD) {
      if (s.kind === K.DISCARD1) {
        const r = s.v, r2 = (r + 2) % 5;
        s.v = r2; if (s.h) { s.h[s.a][r]++; s.h[s.a][r2]--; }
      } else if (Array.isArray(s.v)) { s.v = s.v.slice(1).concat(s.v.slice(0, 1)); }
      for (let r = 0; r < 5; r++) M3.B[(i + 1) * 5 + r] += (r % 2 ? 1 : -1);
      changed++;
    } else if (s.hidden === RC.HIDDEN.DEV) {
      s.v = (s.v + 1) % 5; changed++;
    }
  });
  const res = {};
  for (const [name, M] of [["M1", M1], ["M2", M2], ["M3", M3]]) {
    const got = botRender(RC.publicView(M, me), pubPart);
    let diff = -1;
    for (let i = 0; i < got.length; i++) if (got[i] !== base[i]) { diff = i; break; }
    const where = diff <= D.N ? `position ${diff}` : diff <= 2 * D.N ? `log line ${diff - D.N - 1}`
      : diff <= 3 * D.N + 1 ? `public snapshot ${diff - 2 * D.N - 1}` : "the public view's steps";
    need(diff < 0, `${D.id}: swap ${name} changes the bot view at ${where}`);
    res[name] = diff < 0;
  }
  // sanity: the mutations are real (the truth view does change)
  need(moved > 0 && changed > 0, `${D.id}: swap test found nothing to mutate (moved ${moved}, changed ${changed})`);
  const tv = JSON.stringify(RC.viewModel(M1, null, D.N, "truth")) !== JSON.stringify(RC.viewModel(D, null, D.N, "truth")) || moved === 0;
  return { moved, relabelled, changed, ok: res.M1 && res.M2 && res.M3, truthChanges: tv };
}

// 8. text check (every counted game)
function textCheck(D, BV) {
  const PV = RC.publicView(D, BV.me);
  const pubPart = BV.publicPart();
  for (const i of PV.hiddenIdx) {
    const s = PV.steps[i];
    let text = RC.stepText(PV, i, { view: "bot" });
    if (s.kind === K.ROBBER) text = text.replace(`to ${RC.tileName(PV.board, s.rob)}`, "to <tile>");
    need(!RES_WORDS.test(text), `${D.id} step ${i}: bot-mode text names a hidden card: "${text}"`);
    const vm = RC.viewModel(PV, pubPart, i + 1, "bot");
    let vt = vm.text;
    if (s.kind === K.ROBBER) vt = vt.replace(`to ${RC.tileName(PV.board, s.rob)}`, "to <tile>");
    need(!RES_WORDS.test(vt), `${D.id} position ${i + 1}: bot-mode last-action text names a hidden card`);
    vm.players.forEach(p => need(p.seat === BV.me || !p.delta, `${D.id} position ${i + 1}: bot-mode resource chips for seat ${p.seat}`));
  }
}

// ---------------------------------------------------------------------------------------------------------------
// decode every game
const truthFiles = {};
function truthFor(t, g) {
  if (!opt.truth) return null;
  const name = `${t}_${String(Math.floor(g / index.shardSize)).padStart(2, "0")}.steps.json.gz`;
  if (!(name in truthFiles)) {
    const p = path.join(opt.truth, name);
    truthFiles[name] = fs.existsSync(p) ? JSON.parse(zlib.gunzipSync(fs.readFileSync(p)).toString("utf8")) : null;
  }
  const d = truthFiles[name];
  return d ? d[`${t}-${g}`] || null : null;
}

const perTest = {};
for (const t of testsHere) {
  const rows = rowsByTest[t] || [];
  const wanted = new Set(rows.map(r => r.g));
  const logs = readLogRecords(t, wanted);
  const st = perTest[t] = { games: 0, ok: 0, sampled: 0, positions: 0, swaps: 0, counted: 0, paths: new Set() };
  const shards = [...new Set(rows.map(r => RC.shardPath(t, r.g, index.shardSize)))].sort();
  for (const sp of shards) {
    const file = path.join(opt.bundle, sp);
    if (!need(fs.existsSync(file), `${sp} is missing`)) continue;
    const { data: shard, path: how } = await RC.parseJsonBytesEx(new Uint8Array(fs.readFileSync(file)), gunzip, sp);
    st.paths.add(how);
    need(shard.v === 2 && shard.t === t, `${sp}: header ${shard.v} ${shard.t}`);
    shard.games.forEach((g, i) => need(g.g === shard.from + i, `${sp}: games[${i}] is game ${g.g}, expected ${shard.from + i}`));
    for (const g of shard.games) {
      const row = index.byId.get(g.id);
      if (!row) { if (!opt.bundleOnly) bad(`${g.id} is in ${sp} but not in the index`); continue; }
      const before = problemCount;
      st.games++;
      let D;
      try { D = RC.decodeGame(g, geo); } catch (e) { bad(`${g.id}: decode failed: ${e.message}`); continue; }
      const rec = logs.get(g.g);
      if (need(rec, `${g.id}: not found in the original logs`)) checkFinal(D, rec);
      checkIndexRow(D, row);
      need(RC.shardPath(t, g.g, index.shardSize) === sp, `${g.id}: shardPath`);
      const BV = RC.botView(g, D);
      need(!!BV === row.counted, `${g.id}: bot view present ${!!BV}, test counted ${row.counted}`);
      if (BV) {
        st.counted++;
        try { checkBotInvariants(D, BV, g); textCheck(D, BV); } catch (e) { bad(`${g.id}: bot-view check crashed: ${e.message}`); }
        const bk = row.bk, s = g.bv.sum;
        const sig = (x, y) => (x == null && y == null) || (x != null && y != null && Math.abs(x - y) <= 5e-4 * Math.max(1, Math.abs(y)) + 1e-9);
        need(bk && sig(bk[0], s.allExact) && sig(bk[1], s.meanP) && sig(bk[3], s.calP) && sig(bk[4], s.meanPu) &&
          sig(bk[5], s.calPu) && bk[6] === s.maxH && eq(bk.slice(8), s.hidden), `${g.id}: index bk differs from bv.sum`);
        need(Math.abs(bk[2] - s.minP) <= 5e-4 * s.minP + 1e-12, `${g.id}: index minP`);
      }
      const tr = truthFor(t, g.g);
      if (tr) {
        st.sampled++;
        st.positions += tr.pos.length;
        checkTruthPositions(D, BV, tr);
        if (BV) {
          try { const sw = swapTest(D, BV); if (sw.ok) st.swaps++; } catch (e) { bad(`${g.id}: swap test crashed: ${e.message}`); }
        }
      }
      if (problemCount === before) st.ok++;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 9. contrast of the page tokens
function hexToRgb(h) {
  let x = h.replace("#", "");
  if (x.length === 3) x = x.split("").map(c => c + c).join("");
  return [0, 2, 4].map(i => parseInt(x.slice(i, i + 2), 16) / 255);
}
function lum(h) {
  const c = hexToRgb(h).map(v => (v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
function ratio(a, b) { const la = lum(a), lb = lum(b); return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05); }
function tokens(block) {
  const out = {};
  for (const m of block.matchAll(/(--[a-zA-Z0-9-]+)\s*:\s*(#[0-9a-fA-F]{3,6})\b/g)) out[m[1]] = m[2];
  return out;
}
const pageSrc = fs.existsSync(pagePath) ? fs.readFileSync(pagePath, "utf8") : fs.existsSync(opt.template) ? fs.readFileSync(opt.template, "utf8") : null;
let contrastLine = "contrast: no page to check";
if (pageSrc) {
  const style = (/<style>([\s\S]*?)<\/style>/.exec(pageSrc) || [])[1] || "";
  const rootM = /(^|\n):root\s*\{([\s\S]*?)\n\}/.exec(style);
  const mediaM = /@media \(prefers-color-scheme: dark\)\s*\{\s*:root:not\(\[data-theme="light"\]\)\s*\{([\s\S]*?)\}\s*\}/.exec(style);
  const darkM = /:root\[data-theme="dark"\]\s*\{([\s\S]*?)\}/.exec(style);
  if (need(rootM && mediaM && darkM, "page: the three token blocks (:root, media query, [data-theme=dark]) were not found")) {
    const light = tokens(rootM[2]), media = Object.assign({}, light, tokens(mediaM[1])), dark = Object.assign({}, light, tokens(darkM[1]));
    need(eq(tokens(mediaM[1]), tokens(darkM[1])), "page: the two dark token blocks differ");
    const text = ["--known", "--differs", "--hidden", "--delta-up", "--delta-down", "--muted", "--ink"];
    const strokes = ["--meter-fill"];
    let worst = Infinity, worstName = "";
    for (const [theme, T] of [["light", light], ["dark", dark], ["dark (system)", media]]) {
      for (const bg of ["--panel", "--ground"]) {
        for (const tk of text.concat(strokes)) {
          if (!need(T[tk] && T[bg], `page: token ${tk} or ${bg} missing in the ${theme} theme`)) continue;
          const r = ratio(T[tk], T[bg]), min = text.includes(tk) ? 4.5 : 3;
          need(r >= min, `page: ${tk} ${T[tk]} on ${bg} ${T[bg]} is ${r.toFixed(2)}:1 in the ${theme} theme (needs ${min}:1)`);
          if (r / min < worst) { worst = r / min; worstName = `${tk} on ${bg} (${theme}) ${r.toFixed(2)}:1`; }
        }
      }
      // WHITE marks keep the --piece-edge casing: the casing stands out from the WHITE fill, and the cased mark
      // (casing or fill) stands out from --panel and --ground (the dark theme needs no casing, the fill does it)
      need(ratio(T["--piece-edge"], T["--p-WHITE"]) >= 3, `page: WHITE casing ${T["--piece-edge"]} vs ${T["--p-WHITE"]} in the ${theme} theme`);
      for (const bg of ["--panel", "--ground"]) {
        const r = Math.max(ratio(T["--piece-edge"], T[bg]), ratio(T["--p-WHITE"], T[bg]));
        need(r >= 3, `page: a cased WHITE mark on ${bg} is ${r.toFixed(2)}:1 in the ${theme} theme (needs 3:1)`);
      }
    }
    contrastLine = `contrast: 7 text tokens >= 4.5:1, --meter-fill and the cased WHITE mark >= 3:1 on --panel and --ground in both themes; tightest ${worstName}`;
  }
}
say(contrastLine);

// ---------------------------------------------------------------------------------------------------------------
// 10. parsers
{
  const idx = index;
  const tIds = idx.order;
  const t0 = tIds.includes("T8") ? "T8" : tIds[0];
  const games = idx.tests[t0].games;
  const r5 = idx.byId.get(`${t0}-5`);
  const cases = [];
  const ph = h => RC.parseHash(h, idx);
  let x = ph(`#${t0.toLowerCase()}-5`);
  cases.push([x.view === "game" && x.id === `${t0}-5` && x.k === null, `#${t0.toLowerCase()}-5 opens ${t0}-5`]);
  x = ph(`#${t0}-${games + 599}`);
  cases.push([x.view === "picker" && eq(x.filters.tests, [t0]) && x.message === `There is no game ${t0}-${games + 599} (${t0} has games 0–${games - 1})`, `#${t0}-${games + 599}`]);
  x = ph(`#${t0}-5.9999`);
  cases.push([x.view === "game" && r5 && x.k === r5.acts && x.message === `Game ${t0}-5 has ${r5.acts} actions; showing the end`, `#${t0}-5.9999`]);
  x = ph(`#${t0}-5.57`);
  cases.push([x.view === "game" && x.k === 57 && !x.message, `#${t0}-5.57`]);
  x = ph(`#${t0}`);
  cases.push([x.view === "picker" && eq(x.filters.tests, [t0]) && x.explicit, `#${t0}`]);
  const two = tIds.slice(0, 2);
  x = ph(`#${two.join(".")}~won~b3`);
  cases.push([x.view === "picker" && eq(x.filters.tests, two) && x.filters.result === "won" && x.filters.behind === 3 && !x.filters.behindPub, `#${two.join(".")}~won~b3`]);
  x = ph("#x9");
  cases.push([x.view === "picker" && x.message === "Unknown link part 'x9'; showing all games" && x.filters.tests.length === 0, "#x9"]);
  x = ph("#selftest");
  cases.push([x.view === "selftest", "#selftest"]);
  x = ph("");
  cases.push([x.view === "picker" && x.filters === null, "empty hash"]);
  // filterToken round trips
  const samples = [RC.defaultFilters(), Object.assign(RC.defaultFilters(), { tests: two, result: "lost", info: "pub", hv: "hvo" }),
    Object.assign(RC.defaultFilters(), { wp: 8, behind: 2, behindPub: true, rmin: 40, rmax: null, seat: 3, sort: "meanp", desc: true }),
    Object.assign(RC.defaultFilters(), { fmt: ["1v3", "2v2-mixed"], opp: ["a", "s"], ver: "3.3.0", q: `${t0}-5`, sort: "rolls" })];
  for (const f of samples) {
    const tok = RC.filterToken(f);
    const back = ph("#" + tok);
    cases.push([/^[A-Za-z0-9._~-]+$/.test(tok) && eq(back.filters, f), `filterToken round trip ${tok}`]);
  }
  cases.push([RC.shardPath("T7", 537) === "shards/T7_05.json.gz", "shardPath(T7, 537)"]);
  const w = RC.wilson(234, 400);
  cases.push([Math.abs(w[0] - 0.536) < 5e-4 && Math.abs(w[1] - 0.632) < 5e-4, `wilson(234, 400) = ${w.map(v => v.toFixed(4))}`]);
  // fetchJson through an injected fetch
  const obj = { hello: [1, 2, 3] };
  const plain = new TextEncoder().encode(JSON.stringify(obj));
  const gz = new Uint8Array(zlib.gzipSync(plain));
  const html = new TextEncoder().encode("<!doctype html><title>Not found</title>");
  const fakeFetch = async url => ({
    ok: url !== "missing", status: url === "missing" ? 404 : 200,
    arrayBuffer: async () => (url === "gz" ? gz : url === "plain" ? plain : html).buffer.slice(0)
  });
  const fo = { fetch: fakeFetch, gunzip };
  const a1 = await RC.fetchJsonEx("gz", fo), a2 = await RC.fetchJsonEx("plain", fo);
  cases.push([eq(a1.data, obj) && a1.path === "gzip", "fetchJson: gzip bytes"]);
  cases.push([eq(a2.data, obj) && a2.path === "plain", "fetchJson: plain JSON bytes"]);
  let k1 = null, k2 = null, k3 = null;
  try { await RC.fetchJson("html", fo); } catch (e) { k1 = e.kind; }
  try { await RC.fetchJson("missing", fo); } catch (e) { k2 = e.kind + ":" + e.status; }
  try { await RC.fetchJson("gz", { fetch: fakeFetch, gunzip: async () => { throw Object.assign(new Error("x"), { kind: "nogzip" }); } }); } catch (e) { k3 = e.kind; }
  cases.push([k1 === "notdata", "fetchJson: an HTML page with status 200 gives notdata"]);
  cases.push([k2 === "http:404", "fetchJson: a 404 gives http"]);
  cases.push([k3 === "nogzip", "fetchJson: no gunzip gives nogzip"]);
  const failed = cases.filter(c => !c[0]);
  for (const c of failed) bad(`parser check failed: ${c[1]}`);
  say(`parsers: ${cases.length - failed.length}/${cases.length} checks pass (parseHash, filterToken, shardPath, wilson, fetchJson)`);
}

// ---------------------------------------------------------------------------------------------------------------
for (const t of testsHere) {
  const s = perTest[t];
  const counted = s.counted ? `, ${s.counted} bot views checked` : "";
  const sampled = s.sampled ? `, ${s.sampled} sampled games × ${Math.round(s.positions / s.sampled)} positions (mean) OK` : "";
  const swap = s.counted ? `, swap ${s.swaps}/${s.sampled} OK` : "";
  say(`${t}: ${s.ok}/${s.games} games OK${counted}${sampled}${swap} [shards read as ${[...s.paths].join("+")}]`);
}
finish();

function finish() {
  if (problemCount) {
    console.log(`FAILED: ${problemCount} problem(s); the first ${problems.length}:`);
    for (const p of problems) console.log("  " + p);
    process.exit(1);
  }
  console.log("OK: every check passed");
  process.exit(0);
}
