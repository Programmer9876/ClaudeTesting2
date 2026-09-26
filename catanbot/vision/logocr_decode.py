"""Decoding of the log OCR: glyph posteriors -> words -> the canonical text of an entry.

Per text run (:func:`decode_run`)
    1. every candidate segment's log posterior over :data:`~catanbot.vision.logocr_glyphs.CHARS`
       (plus *garbage*) from the glyph classifier, laid out as a banded matrix over pairs of cut
       candidates;
    2. a *word lattice*: every lexicon word (:data:`LEXICON`: the log vocabulary, numbers, card
       words with a trailing comma) is aligned to the cuts by a character-level dynamic
       programme run for the whole vocabulary at once (numpy over words x cuts), giving its best
       score for each end cut; an open-vocabulary path (best character per segment, penalised per
       character) covers unexpected words - player names drawn in the text colour included;
    3. word boundaries cost what the gap says: a wide gap is a space, a cut through ink is not
       (a soft logistic in x-height units), so touching words at small sizes still split;
    4. a Viterbi search over the lattice with a word bigram model (:class:`WordLM`, estimated from
       the phrase table's wordings with slots for names, card icons, dice and numbers, smoothed so
       unseen word pairs stay possible) picks the words; the previous token (a name, an icon) is
       the context.

Per entry (:func:`assemble`)
    names become their colour words, icon runs become ``N res`` / ``a card`` / ``N cards`` /
    ``Development Card`` / ``d1 d2``, a number glued before a card icon multiplies it, a text-coloured
    word in a player slot that is not "you" is the grey (white) player; the line must parse with
    :func:`catanbot.colonist_log.parse_log_line` into a known kind without problems, else the
    runs' alternatives are tried (:func:`decode_entry`).
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .. import colonist_log as L
from . import logocr_glyphs as G

__all__ = ["LEXICON", "WordLM", "RunReading", "decode_run", "assemble", "decode_entry", "icon_text"]

NEG = -1e4

# ---------------------------------------------------------------------------
# lexicon and language model
# ---------------------------------------------------------------------------
_BASE_WORDS = """got rolled stole from wants to give for traded with bank gave and took used Monopoly Year of Plenty
Knight Soldier Road Building bought Development Card Cards built a an Settlement City placed received starting
resources counter-offered counteroffered offered moved Robber robber discarded card cards wood brick sheep wheat ore
lumber grain wool You you desert the Victory Point Points total in all accepted rejected declined cancelled trade
turn ended their his her is selecting choosing placing discarding won game Largest Army Longest player nobody
No gets receives offers proposed played monopolized everyone Dev dev""".split()
_CARD_WORDS = ("wood", "brick", "sheep", "wheat", "ore", "lumber", "grain", "wool", "card", "cards")
_NUMBERS = [str(n) for n in range(0, 31)]
#: Words Colonist writes with a glued colon ("got:", "gave bank:", "wants to give:", "for:", "from:" ...).
_COLON_WORDS = L.COLON_WORDS
#: The recognisable words (a comma glued to a card word, a colon glued to a verb, are part of the word).
LEXICON: Tuple[str, ...] = tuple(dict.fromkeys(
    _BASE_WORDS + [w + "," for w in _CARD_WORDS] + _NUMBERS + [n + "," for n in _NUMBERS[:13]] + [":"]
    + [w + ":" for w in _COLON_WORDS]))
_NUM_RE = re.compile(r"^\d+,?$")


def drop_verb_colon(word: str) -> str:
    """A word of :data:`_COLON_WORDS` without its glued colon ("got:" -> "got"): Colonist writes the
    colon on some lines only, so the canonical text leaves it out (as the synthetic panel's does)."""
    if len(word) > 1 and word.endswith(":") and word[:-1].lower() in _COLON_WORDS:
        return word[:-1]
    return word


def _lm_class(tok: str) -> str:
    """LM symbol of a token: numbers collapse to D (1..6, a die face or a small count) / NUM, with or
    without a comma; a word's glued colon is left out ("got:" is "got")."""
    if len(tok) > 1 and tok.endswith(":"):
        tok = tok[:-1]
    if _NUM_RE.match(tok):
        core = tok.rstrip(",")
        c = "D" if core in ("1", "2", "3", "4", "5", "6") else "NUM"
        return c + "," if tok.endswith(",") else c
    return tok


def _lm_corpus(rng: random.Random, n: int = 6000) -> List[List[str]]:
    """Token sequences of log lines in the phrase table's wordings: player slots NAME / You / you,
    card slots as icons (CARDS) or text card lists, dice (DICE) or two numbers."""
    res = ["wood", "brick", "sheep", "wheat", "ore", "lumber", "grain", "wool"]

    def num() -> str:
        return "D" if rng.random() < 0.9 else "NUM"

    def cards_text() -> List[str]:
        k = rng.random()
        if k < 0.45:
            m = rng.randint(1, 3)
            out: List[str] = []
            for i in range(m):
                out += [num(), rng.choice(res) + ("," if i < m - 1 else "")]
            return out
        if k < 0.75:
            return [rng.choice(res) for _ in range(rng.randint(1, 4))]
        if k < 0.87:
            return ["a", "card"]
        return [num(), "cards"]

    def cards() -> List[str]:
        r = rng.random()
        if r < 0.55:
            return ["CARDS"]
        if r < 0.6:
            return [num(), "CARDS"]
        return cards_text()

    def p() -> str:
        return "NAME" if rng.random() < 0.8 else "You"

    def other() -> str:
        return "NAME" if rng.random() < 0.8 else "you"

    item = lambda: rng.choice(["Road", "Settlement", "City"])  # noqa: E731
    temps = [
        lambda: [p(), "got"] + cards(),
        lambda: [p(), "rolled"] + (["DICE"] if rng.random() < 0.6 else ["D", "D"]),
        lambda: [p(), "received", "starting", "resources"] + cards(),
        lambda: [p(), "placed", "a", item()],
        lambda: [p(), "built", "a", item()],
        lambda: [p(), "bought"] + (["DEV"] if rng.random() < 0.6 else ["Development", "Card"]),
        lambda: [p(), "used", rng.choice(["Knight", "Monopoly"])],
        lambda: [p(), "used", "Road", "Building"],
        lambda: [p(), "used", "Year", "of", "Plenty"],
        lambda: [p(), "used", "Monopoly", "and", "stole"] + ([num(), "CARDS"] if rng.random() < 0.6 else
                                                              [num(), rng.choice(res)]),
        lambda: [p(), "used", "Year", "of", "Plenty", "and", "took"] + cards(),
        lambda: [p(), "stole"] + ([num(), "CARDS"] if rng.random() < 0.6 else [num(), rng.choice(res)]),
        lambda: [p(), "stole"] + (["CARDS"] if rng.random() < 0.5 else (["a", "card"] if rng.random() < 0.5 else
                                                                         ["1", rng.choice(res)])) + ["from", other()],
        lambda: [p(), "moved", "Robber", "to"] + ([num(), rng.choice(res)] if rng.random() < 0.85 else ["desert"]),
        lambda: [p(), "discarded"] + cards(),
        lambda: [p(), "gave", "bank"] + cards() + ["and", "took"] + cards(),
        lambda: [p(), "gave"] + cards() + ["and", "got"] + cards() + ["from", "bank"],
        lambda: [p(), "traded"] + cards() + ["for"] + cards() + ["with", "bank"],
        lambda: [p(), "traded"] + cards() + ["for"] + cards() + ["with", other()],
        lambda: [p(), "counter-offered"] + cards() + ["for"] + cards(),
        lambda: [p(), "counter-offered", "to", "NAME", ":"] + cards() + ["for"] + cards(),
        lambda: [p(), "wants", "to", "give"] + cards() + ["for"] + cards(),
        lambda: [p(), "took", "from", "bank"] + cards(),
        lambda: [p(), rng.choice(["accepted", "rejected", "declined"]), "the", "trade"],
        lambda: [p(), "ended", "their", "turn"],
    ]
    wts = [14, 10, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 3, 1, 1, 6, 1, 0.3, 0.3]
    out = []
    for _ in range(n):
        t = rng.choices(temps, weights=wts)[0]()
        out.append(t)
    return out


class WordLM:
    """Word bigram model over LM symbols (:func:`_lm_class`) with add-k smoothing: ``logp(prev, w)``.
    ``<s>`` starts a line, ``UNK`` is any word outside the lexicon (a name in the text colour, an
    unexpected word)."""

    def __init__(self, seed: int = 0, k: float = 0.05) -> None:
        rng = random.Random(seed)
        counts: Dict[str, Dict[str, float]] = {}
        vocab = set(_lm_class(w) for w in LEXICON) | {"NAME", "CARDS", "DICE", "DEV", "UNK", "</s>", "NUM", "NUM,",
                                                        "D", "D,"}
        for seq in _lm_corpus(rng):
            prev = "<s>"
            for tok in seq + ["</s>"]:
                t = _lm_class(tok)
                counts.setdefault(prev, {})
                counts[prev][t] = counts[prev].get(t, 0.0) + 1.0
                prev = t
        self.vocab = sorted(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.table: Dict[str, np.ndarray] = {}
        for prev in list(counts) + ["UNK"]:
            row = np.full(V, k, dtype=np.float64)
            for t, c in counts.get(prev, {}).items():
                if t in self.index:
                    row[self.index[t]] += c
            # UNK (names in the text colour, stray words) is always possible
            row[self.index["UNK"]] += 0.02 * row.sum()
            self.table[prev] = np.log(row / row.sum())
        self.uniform = float(np.log(1.0 / V))
        self._memo: Dict[Tuple[str, str], float] = {}

    def logp(self, prev: str, w: str) -> float:
        k = (prev, w)
        v = self._memo.get(k)
        if v is not None:
            return v
        row = self.table.get(_lm_class(prev))
        if row is None:
            v = self.uniform
        else:
            i = self.index.get(_lm_class(w))
            if i is None:
                i = self.index["UNK"]
            v = float(row[i])
        self._memo[k] = v
        return v


@lru_cache(maxsize=1)
def _lm() -> WordLM:
    return WordLM()


@lru_cache(maxsize=1)
def _lexicon_arrays() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Words as a padded char-index matrix (W, Lmax) and their lengths."""
    words = [w for w in LEXICON if all(c in G.CHAR_INDEX for c in w)]
    Lmax = max(len(w) for w in words)
    M = np.zeros((len(words), Lmax), dtype=np.int64)
    lens = np.array([len(w) for w in words], dtype=np.int64)
    for i, w in enumerate(words):
        for k, c in enumerate(w):
            M[i, k] = G.CHAR_INDEX[c]
    return M, lens, words


# ---------------------------------------------------------------------------
# run decoding
# ---------------------------------------------------------------------------
@dataclass
class RunReading:
    """Words read from one text run: ``words`` with per-word ``scores`` (log-prob margins vs the
    open-vocabulary reading: 0 best), ``unknown`` flags (open-vocabulary words), the x extent of
    each word (panel-inner pixels) and the run's overall confidence."""

    words: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    unknown: List[bool] = field(default_factory=list)
    spans: List[Tuple[int, int]] = field(default_factory=list)
    conf: float = 0.0
    total: float = 0.0
    alts: List[Tuple[float, List[str]]] = field(default_factory=list)


def _space_logp(gap_px: float, xh: float) -> Tuple[float, float]:
    """``(log P(space), log P(no space))`` at a cut with an empty gap of ``gap_px`` pixels."""
    if gap_px >= 50:
        return 0.0, NEG
    g = gap_px / max(1.0, xh)
    z = (g - 0.3) / 0.06
    p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p), math.log(1.0 - p)


#: Lexicon words whose score upper bound (log-prob) is below this are not tried on a run.
WORD_UB_MIN = -40.0

#: Marker word of a player name drawn in the text colour (the grey / white player).
NAME_WORD = "\x00NAME"
_NAME_SLOT_AFTER = ("<s>", "from", "with")


def decode_run(lp: np.ndarray, pairs: Sequence[Tuple[int, int]], bounds: Sequence[int], gaps: Sequence[float],
               xh: float, prev: str = "<s>", run_x0: int = 0, lm: Optional[WordLM] = None,
               unk_penalty: float = 2.0, topk: int = 8, nxt: Optional[str] = None,
               name_cost: float = 5.5) -> RunReading:
    """Read one run from its segments' log posteriors ``lp`` (``len(pairs) x NUM_CLASSES``).

    ``prev`` / ``nxt`` are the LM symbols of the tokens around the run (``nxt`` None: unknown).
    In a player slot (the entry's start, after "from" / "with" / "to") a text-coloured player name
    may cover any words (:data:`NAME_WORD`, a fixed cost ``name_cost``)."""
    lm = lm or _lm()
    nb = len(bounds)
    rr = RunReading()
    if nb < 2 or lp.shape[0] == 0:
        return rr
    P = np.asarray(pairs, dtype=np.int64)
    D = int((P[:, 1] - P[:, 0]).max())
    C = G.NUM_CLASSES
    LPB = np.full((nb, D + 1, C), NEG, dtype=np.float32)
    LPB[P[:, 0], P[:, 1] - P[:, 0]] = lp
    spl = [_space_logp(g, xh) for g in gaps]
    sp = np.array([a for a, _ in spl])
    ns = np.array([b for _, b in spl])
    sp[0] = sp[-1] = 0.0
    ns[0] = ns[-1] = NEG
    # word boundary candidates: a real gap, or a low-ink cut (touching words at small sizes)
    wb = sp > math.log(0.02)
    wb[0] = wb[-1] = True
    # --- lexicon words: banded DP over all words at once ---------------------------------------
    M, lens, words = _lexicon_arrays()
    # words that cannot score (a letter no segment of the run shows at all) are left out: an
    # upper bound of a word's score is the sum of its letters' best segment scores
    cmax = lp.max(axis=0)
    ub = np.where(np.arange(M.shape[1])[None, :] < lens[:, None], cmax[M], 0.0).sum(axis=1)
    keep = np.flatnonzero(ub > WORD_UB_MIN)
    if keep.size < M.shape[0]:
        M, lens, words = M[keep], lens[keep], [words[i] for i in keep.tolist()]
    if M.shape[0] == 0:
        M, lens, words = _lexicon_arrays()
    W, Lmax = M.shape
    start = np.where(wb, sp, NEG).astype(np.float32)
    start[0] = 0.0
    dp = np.broadcast_to(start, (W, nb)).copy()
    st = np.broadcast_to(np.arange(nb), (W, nb)).copy()
    best_end = np.full((W, nb), NEG, dtype=np.float32)
    best_st = np.zeros((W, nb), dtype=np.int64)
    nsf = ns.astype(np.float32)
    # a colon glued to a word may stand a space's width off it (its side bearing): no gap cost
    colon_join = np.maximum(ns, sp).astype(np.float32)
    colon = G.CHAR_INDEX[":"]
    LPT = np.ascontiguousarray(LPB.transpose(1, 2, 0))          # (shift, class, start)
    active = np.arange(W)
    for k in range(Lmax):
        active = active[lens[active] > k]
        if active.size == 0:
            break
        ck = M[active, k]
        if k == 0:
            cur = dp[active]
        else:
            cur = dp[active] + np.where((ck == colon)[:, None], colon_join[None, :], nsf[None, :])
        cst = st[active]
        new = np.full((active.size, nb), NEG, dtype=np.float32)
        nst = np.zeros((active.size, nb), dtype=np.int64)
        for sft in range(1, min(D, nb - 1) + 1):
            cand = cur[:, :nb - sft] + LPT[sft, :, :nb - sft][ck]
            tgt = new[:, sft:]
            better = cand > tgt
            np.copyto(tgt, cand, where=better)
            np.copyto(nst[:, sft:], cst[:, :nb - sft], where=better)
        dp[active] = new
        st[active] = nst
        done = active[lens[active] == k + 1]
        if done.size:
            best_end[done] = dp[done]
            best_st[done] = st[done]
    # --- open-vocabulary path: best character per segment, spaces from the gaps -----------------
    free = LPB[:, :, :G.GARBAGE].max(axis=2)
    free_c = LPB[:, :, :G.GARBAGE].argmax(axis=2)
    V = np.full(nb, NEG)
    back = np.full(nb, -1, dtype=np.int64)
    brk = np.zeros(nb, dtype=bool)
    V[0] = 0.0
    for j in range(1, nb):
        bestv, arg, isb = NEG, -1, False
        for sft in range(1, min(D, j) + 1):
            i = j - sft
            if V[i] <= NEG / 2 or free[i, sft] <= NEG / 2:
                continue
            if i == 0:
                t, b = 0.0, True
            elif wb[i] and sp[i] >= ns[i]:
                t, b = sp[i], True
            else:
                t, b = ns[i], False
            v = V[i] + t + float(free[i, sft]) - unk_penalty
            if v > bestv:
                bestv, arg, isb = v, i, b
        V[j], back[j], brk[j] = bestv, arg, isb
    # the open path's words become the unknown-word edges
    edges: Dict[int, List[Tuple[int, str, float, bool]]] = {}
    if V[nb - 1] > NEG / 2:
        cuts = []
        j = nb - 1
        chars = []
        wend = j
        while j > 0:
            i = int(back[j])
            chars.append(G.CHARS[int(free_c[i, j - i])])
            if brk[j] or i == 0:
                # a word from i to wend
                cuts.append((i, wend, "".join(reversed(chars))))
                chars = []
                wend = i
            j = i
        for i, j, wd in cuts:
            if wb[i] and wb[j]:
                sc = 0.0
                x = j
                while x != i:
                    y = int(back[x])
                    sc += float(free[y, x - y]) + (ns[y] if y != i else 0.0)
                    x = y
                edges.setdefault(j, []).append((i, wd, sc - unk_penalty * len(wd), True))
    # lexicon edges: the top words ending at each boundary
    ends = np.flatnonzero(wb)
    for j in ends.tolist():
        col = best_end[:, j]
        if not (col > NEG / 2).any():
            continue
        top = np.argsort(-col)[:topk]
        bestv = float(col[top[0]])
        for wi in top.tolist():
            v = float(col[wi])
            if v < bestv - 14.0 or v <= NEG / 2:
                break
            i = int(best_st[wi, j])
            edges.setdefault(j, []).append((i, words[wi], v - float(start[i]), False))
    # --- Viterbi over the boundaries with the bigram model ------------------------------------
    # player-name edges (a name in the text colour): from a player slot to any real word gap
    real = [int(j) for j in np.flatnonzero(wb)]
    real_gap = np.asarray([g < 50 and g / max(1.0, xh) >= 0.3 for g in gaps])
    states: Dict[int, Dict[str, Tuple[float, Any]]] = {0: {prev: (0.0, None)}}
    for j in sorted(set(edges) | set(real)):
        if j == 0:
            continue
        cur: Dict[str, Tuple[float, Any]] = {}
        for (i, word, sc, unk) in edges.get(j, []):
            si = states.get(i)
            if not si:
                continue
            key = "UNK" if unk else word
            startc = sp[i] if i != 0 else 0.0
            for pw, (ps, bp) in si.items():
                v = ps + sc + startc + lm.logp(pw, key)
                old = cur.get(key)
                if old is None or v > old[0]:
                    cur[key] = (v, (i, pw, word, sc, unk))
        # name edges ending at j (each real word gap inside the name costs extra: a name is one or
        # two words, never the verb after it)
        name_end = j == nb - 1 or bool(real_gap[j])
        for i, si in (states.items() if name_end else ()):
            if i >= j or not (i == 0 or real_gap[i]):
                continue
            for pw, (ps, bp) in si.items():
                if _lm_class(pw) not in _NAME_SLOT_AFTER:
                    continue
                startc = sp[i] if i != 0 else 0.0
                inner = int(np.count_nonzero(real_gap[i + 1:j]))
                v = ps - name_cost - 3.5 * inner + startc + lm.logp(pw, "NAME")
                old = cur.get("NAME")
                if old is None or v > old[0]:
                    cur["NAME"] = (v, (i, pw, NAME_WORD, -name_cost, True))
        if cur:
            if len(cur) > 10:
                cur = dict(sorted(cur.items(), key=lambda kv: -kv[1][0])[:10])
            states[j] = cur
    end = nb - 1
    if end not in states:
        return rr
    tail = (lambda k: lm.logp(k, nxt)) if nxt else (lambda k: 0.0)
    finals = sorted(((v + tail(k), k) for k, (v, _) in states[end].items()), reverse=True)
    total, last = finals[0]
    seq = []
    j, k = end, last
    while True:
        v, bp = states[j][k]
        if bp is None:
            break
        i, pw, word, sc, unk = bp
        seq.append((word, sc, unk, i, j))
        j, k = i, pw
    seq.reverse()
    rr.total = float(total)
    for word, sc, unk, i, j in seq:
        rr.words.append(word)
        rr.scores.append(sc / max(1, len(word)))
        rr.unknown.append(unk)
        rr.spans.append((run_x0 + int(bounds[i]), run_x0 + int(bounds[j])))
    for v, k in finals[1:3]:
        rr.alts.append((float(v - total), [k]))
    # a reading ending in the same word with / without its glued colon ("got:" / "got") is no rival
    rivals = [v for v, k in finals[1:] if _lm_class(k) != _lm_class(last)]
    margin = (finals[0][0] - rivals[0]) if rivals else 10.0
    q = min((s for s, u in zip(rr.scores, rr.unknown) if not u), default=-10.0)
    rr.conf = float(max(0.0, min(1.0, 1.0 + q / 1.5)) * min(1.0, 0.5 + margin / 4.0))
    return rr


# ---------------------------------------------------------------------------
# icons -> text
# ---------------------------------------------------------------------------
_RES = ("wood", "brick", "sheep", "wheat", "ore")
#: Building icons -> their canonical words.
BUILDING_WORDS: Dict[str, str] = {"road": "Road", "settlement": "Settlement", "city": "City"}


def icon_text(kinds: Sequence[str], mult: Optional[int] = None) -> str:
    """Canonical text of a group of adjacent icons (``wood``.. ``ore`` / ``card`` / ``dev`` /
    ``die3``): card runs -> ``N res`` joined by ``, `` (a leading multiplier multiplies the first
    run), card backs -> ``a card`` / ``N cards``, the dev card -> ``Development Card``, dice ->
    their faces."""
    out: List[str] = []
    runs: List[List[Any]] = []
    for k in kinds:
        if k.startswith("die"):
            out.append(k[3:])
            continue
        if k == "dev":
            out.append("Development Card")
            continue
        if k in BUILDING_WORDS:
            out.append(BUILDING_WORDS[k])
            continue
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    if runs:
        if mult is not None:
            runs[0][1] *= mult
        words = []
        for k, n in runs:
            if k == "card":
                words.append("a card" if n == 1 else f"{n} cards")
            else:
                words.append(f"{n} {k}")
        out.insert(0, ", ".join(words)) if not out else out.append(", ".join(words))
    return " ".join(out)
