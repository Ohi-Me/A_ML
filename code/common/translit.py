"""Learn native-script -> english word maps from training pairs only.

Many S2/S3 Indic names are word-by-word transliterations of the S1 name ('ऑल सिस्टम्स प्राइवेट लिमिटेड' <->
'All Systems Private Limited'). When both names have the same number of words we align them by position and
count. Native-script state names in addresses are learned the same way against the S1 state.
Only pairs of the given S1 ids are used, so the validation fold never leaks in."""
import json
import unicodedata
from collections import Counter, defaultdict

from common.normalize import IN_STATES, US_STATES, _RE_NONLATIN, _STRIP, norm_addr

CODE2NAME = {}
for name, code in list(IN_STATES.items()) + list(US_STATES.items()):
    CODE2NAME.setdefault(code, name)


def _clean(t):
    return unicodedata.normalize("NFKC", t).strip(_STRIP)


def learn(rows, min_count=2, min_share=0.5, state_min=5, state_share=0.8):
    """rows: iterable of (s1_name, s1_addr, rid_name, rid_addr)."""
    wc = defaultdict(Counter)
    sc = defaultdict(Counter)
    for n1, a1, n2, a2 in rows:
        if n2 and _RE_NONLATIN.search(n2):
            t1 = [x for x in (_clean(t) for t in n1.split()) if x]
            t2 = [x for x in (_clean(t) for t in n2.split()) if x]
            if len(t1) == len(t2):
                for x, y in zip(t2, t1):
                    if _RE_NONLATIN.search(x):
                        wc[x][y.lower()] += 1
        if a2 and _RE_NONLATIN.search(a2):
            st = norm_addr(a1)["state"]
            if st:
                for p in a2.split(","):
                    p = unicodedata.normalize("NFKC", p).strip()
                    if p and _RE_NONLATIN.search(p):
                        sc[p][st] += 1
    words = {}
    for k, c in wc.items():
        w, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= min_share:
            words[k] = w
    states = {}
    for k, c in sc.items():
        st, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= state_min and n / tot >= state_share and st in CODE2NAME:
            states[k] = CODE2NAME[st]
    return words, states


def save(path, words, states):
    json.dump({"words": words, "states": states}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["words"], d["states"]
