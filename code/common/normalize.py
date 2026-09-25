"""Text normalization for business names and addresses (v1).

Everything here is rule based and country generic where possible. Word lists hold general language knowledge
(legal forms, street words, honorifics, US / Indian / French words) and are never tied to a country code, so a new
country still goes through the same path. The raw text is always kept next to the normalized fields.
"""
import re
import unicodedata

from anyascii import anyascii

# ------------------------------------------------------------------------------------------------ word lists
HONORIFIC = {"mr", "mrs", "ms", "dr", "smt", "shri", "sri", "shree", "sree", "ms", "m/s", "messrs", "the", "prof", "er",
             "kumari", "km", "late", "mx",
             "formerly", "fka", "dba", "aka", "doing", "business", "as"}
# legal / company-form words, mapped to one canonical token
LEGAL_MAP = {
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt ltd", "(p)": "pvt", "limited": "ltd", "ltd": "ltd", "limted": "ltd",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc", "corporation": "corp", "corp": "corp", "company": "co",
    "co": "co", "cos": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "pa": "pa", "plc": "plc",
    "ltda": "ltd", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc", "ei": "ei",
    "scop": "scop", "scp": "scp", "selarl": "selarl", "gie": "gie", "sca": "sca",
}
LEGAL_TOKENS = set(t for v in LEGAL_MAP.values() for t in v.split()) | {"public", "opc"}
# multi word legal phrases first (on the spaced, lower, ascii text)
LEGAL_PHRASES = [
    (r"\bl ?l ?c\b", "llc"), (r"\bl ?l ?p\b", "llp"), (r"\bp ?l ?l ?c\b", "pllc"), (r"\bp ?c\b", "pc"),
    (r"\bs ?a ?s ?u\b", "sasu"), (r"\bs ?a ?s\b", "sas"), (r"\bs ?a ?r ?l\b", "sarl"), (r"\be ?u ?r ?l\b", "eurl"),
    (r"\bs ?c ?i\b", "sci"), (r"\bpvt ?ltd\b", "pvt ltd"), (r"\bpvtltd\b", "pvt ltd"),
    (r"\bprivate limited\b", "pvt ltd"), (r"\bpublic limited\b", "public ltd"), (r"\bpublic ltd\b", "public ltd"),
    (r"\blimited liability partnership\b", "llp"), (r"\blimited liability company\b", "llc"),
    (r"\bsociete par actions simplifiee( unipersonnelle)?\b", "sas"), (r"\bsociete a responsabilite limitee\b", "sarl"),
    (r"\bsociete civile immobiliere\b", "sci"), (r"\bentreprise individuelle\b", "ei"),
    (r"\betablissements?\b", "ets"),
]
# generic business words that the noise often adds, drops or swaps (kept in the name, dropped in "key" form)
GENERIC = {"services", "service", "center", "centre", "group", "holdings", "holding", "partners", "enterprises",
           "enterprise", "associates", "solutions", "company", "co", "and", "of", "india", "france", "usa", "us",
           "international", "global", "industries", "ventures", "trading", "traders", "corporation", "systems",
           "ets", "et", "de", "du", "des", "la", "le", "les", "l", "d", "fils", "freres", "cie", "groupe"}

NUM_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
             "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12", "one": "1", "two": "2",
             "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
             "ground": "0", "premier": "1", "deuxieme": "2", "troisieme": "3"}

ADDR_MAP = {
    # english street words
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave",
    "avn": "ave", "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bvd": "blvd", "highway": "hwy",
    "hwy": "hwy", "place": "pl", "pl": "pl", "terrace": "ter", "ter": "ter", "parkway": "pkwy", "pkwy": "pkwy",
    "square": "sq", "sq": "sq", "trail": "trl", "trl": "trl", "route": "rte", "rte": "rte", "rt": "rte",
    "point": "pt", "mount": "mt", "mountain": "mtn", "heights": "hts", "junction": "jct", "expressway": "expy",
    "freeway": "fwy", "turnpike": "tpke", "crossing": "xing", "cove": "cv", "creek": "crk", "ridge": "rdg",
    "valley": "vly", "view": "vw", "village": "vlg", "center": "ctr", "centre": "ctr", "ctr": "ctr", "plaza": "plz",
    "path": "path", "pike": "pike", "loop": "loop", "run": "run", "way": "way", "alley": "aly", "hollow": "holw",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
    "southwest": "sw", "apartment": "apt", "apartments": "apt", "apt": "apt", "apts": "apt", "suite": "ste",
    "ste": "ste", "floor": "fl", "flr": "fl", "fl": "fl", "building": "bldg", "bldg": "bldg", "bld": "bldg",
    "room": "rm", "department": "dept", "county": "cnty", "township": "twp", "city": "city", "saint": "st",
    "fort": "ft", "unit": "unit", "po": "po", "box": "box", "pmb": "pmb",
    # indian address words
    "nagar": "nagar", "ngr": "nagar", "colony": "colony", "col": "colony", "sector": "sector", "sec": "sector",
    "near": "near", "nr": "near", "opposite": "opp", "opp": "opp", "behind": "behind", "bhd": "behind",
    "cross": "cross", "main": "main", "layout": "layout", "marg": "marg", "chowk": "chowk", "bazar": "bazaar",
    "bazaar": "bazaar", "mohalla": "mohalla", "mohall": "mohalla", "gali": "gali", "tehsil": "tehsil",
    "taluka": "taluka", "tq": "taluka", "dist": "district", "district": "district", "distt": "district",
    "village": "vlg", "vill": "vlg", "vpo": "vlg", "post": "post", "industrial": "indl", "indl": "indl",
    "estate": "estate", "complex": "complex", "society": "soc", "soc": "soc", "chs": "chs", "phase": "phase",
    "block": "block", "blk": "block", "extension": "extn", "extn": "extn", "ext": "extn",
    # french address words
    "rue": "rue", "r": "rue", "avenue": "ave", "boulevard": "blvd", "impasse": "imp", "imp": "imp", "allee": "allee",
    "all": "allee", "chemin": "chemin", "che": "chemin", "chem": "chemin", "quai": "quai", "cours": "crs",
    "crs": "crs", "faubourg": "fbg", "fbg": "fbg", "residence": "res", "res": "res", "lieu dit": "ld",
    "square": "sq", "passage": "pass", "pass": "pass", "sentier": "sent", "voie": "voie", "bis": "bis", "ter": "ter",
}
# prefixes before a house number that carry nothing
ADDR_NOISE = {"no", "number", "num", "nos", "h", "hno", "house", "door", "plot", "flat", "shop", "khasra", "kh",
              "s", "sy", "survey", "ward", "unit", "pmb", "po", "box", "ste", "apt", "fl", "rm", "bldg", "n",
              "null", "na", "n/a", "none", "nil", "tbd", "and", "of", "the", "de", "du", "des", "la", "le", "les",
              "d", "l"}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr", "guam": "gu",
}
IN_STATES = {
    "andhra pradesh": "in_ap", "arunachal pradesh": "in_ar", "assam": "in_as", "bihar": "in_br",
    "chhattisgarh": "in_cg", "chattisgarh": "in_cg", "goa": "in_ga", "gujarat": "in_gj", "haryana": "in_hr",
    "himachal pradesh": "in_hp", "jharkhand": "in_jh", "karnataka": "in_ka", "kerala": "in_kl", "keralam": "in_kl",
    "madhya pradesh": "in_mp", "maharashtra": "in_mh", "manipur": "in_mn", "meghalaya": "in_ml", "mizoram": "in_mz",
    "nagaland": "in_nl", "odisha": "in_od", "orissa": "in_od", "punjab": "in_pb", "rajasthan": "in_rj",
    "sikkim": "in_sk", "tamil nadu": "in_tn", "telangana": "in_tg", "tripura": "in_tr", "uttar pradesh": "in_up",
    "uttarakhand": "in_uk", "uttaranchal": "in_uk", "west bengal": "in_wb", "delhi": "in_dl", "new delhi": "in_dl",
    "jammu and kashmir": "in_jk", "jammu & kashmir": "in_jk", "ladakh": "in_la", "puducherry": "in_py",
    "pondicherry": "in_py", "chandigarh": "in_ch", "dadra and nagar haveli": "in_dn", "daman and diu": "in_dd",
    "lakshadweep": "in_ld", "andaman and nicobar islands": "in_an",
}
# two-letter Indian state codes as they appear in the data ("MH", "UP", "DL", ...)
IN_CODES = {"ap": "in_ap", "ar": "in_ar", "as": "in_as", "br": "in_br", "cg": "in_cg", "ct": "in_cg", "ga": "in_ga",
            "gj": "in_gj", "hr": "in_hr", "hp": "in_hp", "jh": "in_jh", "ka": "in_ka", "kl": "in_kl", "mp": "in_mp",
            "mh": "in_mh", "mn": "in_mn", "ml": "in_ml", "mz": "in_mz", "nl": "in_nl", "od": "in_od", "or": "in_od",
            "pb": "in_pb", "rj": "in_rj", "sk": "in_sk", "tn": "in_tn", "tg": "in_tg", "ts": "in_tg", "tr": "in_tr",
            "up": "in_up", "uk": "in_uk", "ut": "in_uk", "wb": "in_wb", "dl": "in_dl", "jk": "in_jk", "py": "in_py",
            "ch": "in_ch"}

# French regions and departments (general administrative geography, like US state codes). A department is
# mapped to its region, because S1 writes the region and S2/S3 often write the department. Matched only when it
# is a whole comma part of the address, so street names such as "rue de la loire" are never touched.
FR_REGIONS = {
    "auvergne rhone alpes": "fr_ara", "bourgogne franche comte": "fr_bfc", "bretagne": "fr_bre",
    "centre val de loire": "fr_cvl", "corse": "fr_cor", "grand est": "fr_ges", "hauts de france": "fr_hdf",
    "ile de france": "fr_idf", "normandie": "fr_nor", "nouvelle aquitaine": "fr_naq", "occitanie": "fr_occ",
    "pays de la loire": "fr_pdl", "provence alpes cote d azur": "fr_pac", "guadeloupe": "fr_gp",
    "martinique": "fr_mq", "guyane": "fr_gf", "la reunion": "fr_re", "reunion": "fr_re", "mayotte": "fr_yt",
}
_FR_DEPT_LIST = {
    "fr_ara": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute loire", "puy de dome", "rhone",
               "savoie", "haute savoie"],
    "fr_bfc": ["cote d or", "doubs", "jura", "nievre", "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "fr_bre": ["cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "fr_cvl": ["cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "fr_cor": ["corse du sud", "haute corse"],
    "fr_ges": ["ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse", "moselle", "bas rhin",
               "haut rhin", "vosges"],
    "fr_hdf": ["aisne", "nord", "oise", "pas de calais", "somme"],
    "fr_idf": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis", "val de marne",
               "val d oise"],
    "fr_nor": ["calvados", "eure", "manche", "orne", "seine maritime"],
    "fr_naq": ["charente", "charente maritime", "correze", "creuse", "dordogne", "gironde", "landes", "lot et garonne",
               "pyrenees atlantiques", "deux sevres", "vienne", "haute vienne"],
    "fr_occ": ["ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot", "lozere",
               "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "fr_pdl": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "fr_pac": ["alpes de haute provence", "hautes alpes", "alpes maritimes", "bouches du rhone", "var", "vaucluse"],
}
FR_ADMIN = dict(FR_REGIONS)
for _r, _ds in _FR_DEPT_LIST.items():
    for _d in _ds:
        FR_ADMIN.setdefault(_d, _r)
_RE_COMP_CLEAN = re.compile(r"[\-'\.]+")

_RE_ID = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|\bid\s*:\s*\d+|#\s*\d+|\b\d{6,}\b|\s-\s*\d{5,}")
_RE_DOMAIN = re.compile(r"^\s*(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:c0m|com|co\.in|in|net|org|co|biz|info|fr|us|io)\s*$")
_RE_PUNCT_N = re.compile(r"[^a-z0-9 ]+")
_RE_WS = re.compile(r"\s+")
_RE_ORD = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b", "9": "g", "7": "t"})
_RE_ALNUM_MIX = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]+$")
_STATE_PHRASES = sorted(list(US_STATES.items()) + list(IN_STATES.items()), key=lambda kv: -len(kv[0]))
_RE_STATE = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in _STATE_PHRASES) + r")\b")
_STATE_LOOKUP = dict(_STATE_PHRASES)
_US_CODES = set(US_STATES.values())


# learned from training pairs (code/common/translit.py): native-script word -> english word,
# native-script address part -> state code. Empty until set_translit() is called.
_TR_WORD = {}
_TR_STATE = {}
ADMIN_V2 = False      # normalization v2 switches this on (French regions / departments as state)
_RE_NONLATIN = re.compile(r"[^\x00-\u024F\u1E00-\u1EFF\s]")
_STRIP = "()[]{},.;:!?\"'|-_*#<>/"


def set_translit(words, states):
    global _TR_WORD, _TR_STATE
    _TR_WORD, _TR_STATE = dict(words), dict(states)


def apply_translit(raw):
    """Replace native-script words that the training pairs taught us (e.g. Devanagari 'limited') by english."""
    if not raw or not _TR_WORD or not _RE_NONLATIN.search(raw):
        return raw
    out = []
    for t in raw.split():
        k = unicodedata.normalize("NFKC", t).strip(_STRIP)
        out.append(_TR_WORD.get(k, t) if _RE_NONLATIN.search(t) else t)
    return " ".join(out)


def to_ascii(s):
    """NFKC, transliterate every script to ascii (Devanagari, Tamil, accents ...), lower case."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return anyascii(s).lower()


def _fix_leet(tok):
    """y0uth -> youth, 6lobal -> global, aut0 -> auto. Leaves 5th, 2nd, c-97 style codes alone."""
    if _RE_ALNUM_MIX.match(tok) and not _RE_ORD.match(tok):
        n_dig = sum(c.isdigit() for c in tok)
        if n_dig <= 2 and len(tok) >= 3:
            return tok.translate(_LEET)
    return tok


def norm_name(raw):
    """Returns dict with: n (normalized full), core (without legal/honorific tokens, sorted variant elsewhere),
    legal (canonical legal tokens), key (core without generic words), domain flag, nonlatin flag."""
    nonlatin = bool(_RE_NONLATIN.search(raw or ""))
    s = to_ascii(apply_translit(raw))
    s = s.replace('"', " ")
    s = _RE_ID.sub(" ", s)
    dom = _RE_DOMAIN.match(s.strip())
    is_domain = bool(dom)
    if dom:
        s = dom.group(1).replace("-", " ")
    s = s.replace("&", " and ").replace("m/s", " ").replace("@", " at ")
    s = re.sub(r"(?<=[a-z])\.(?=[a-z]\b)", "", s)      # l.l.c -> llc, p.c. -> pc
    s = s.replace(".", "").replace("'", "")
    s = _RE_PUNCT_N.sub(" ", s)
    toks = [_fix_leet(t) for t in s.split()]
    s = " ".join(toks)
    for pat, rep in LEGAL_PHRASES:
        s = re.sub(pat, rep, s)
    toks = []
    for t in s.split():
        toks.extend(LEGAL_MAP.get(t, t).split())
    legal = [t for t in toks if t in LEGAL_TOKENS]
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in HONORIFIC]
    if not core:                                    # the whole name was legal words, keep it
        core = [t for t in toks if t not in HONORIFIC] or toks
    key = [t for t in core if t not in GENERIC] or core
    return {
        "n": " ".join(toks), "core": " ".join(core), "key": " ".join(key),
        "legal": " ".join(sorted(set(legal))), "is_domain": is_domain, "nonlatin": nonlatin,
    }


def formerly_name(raw):
    """For 'NewName formerly OldName' return OldName (normalized core) else ''."""
    s = to_ascii(raw)
    m = re.search(r"\bformerly\b(.*)$", s)
    return norm_name(m.group(1))["core"] if m else ""


def norm_addr(raw):
    """Returns dict: a (normalized address tokens, state removed), nums (sorted unique digit runs, no leading zeros),
    state (canonical code or ''), first_num (first number seen), comps (number of comma parts)."""
    if raw and _TR_STATE and _RE_NONLATIN.search(raw):
        raw = ", ".join(_TR_STATE.get(unicodedata.normalize("NFKC", p).strip(), p) for p in raw.split(","))
    s = to_ascii(raw)
    s = s.replace('"', " ")
    if s.strip() in ("", "null", "none", "nan", "n/a", "na"):
        return {"a": "", "nums": "", "state": "", "first_num": "", "comps": 0}
    comps = [c.strip() for c in s.split(",") if c.strip() and c.strip() not in ("null", "n/a", "na", "none", "nan")]
    s = ", ".join(comps)
    state = ""
    if ADMIN_V2:
        # v2: a full state name counts only as a whole comma part (a street named after a state is left alone)
        kept = []
        for p in s.split(","):
            key = " ".join(p.replace("-", " ").split())
            if key in _STATE_LOOKUP:
                state = _STATE_LOOKUP[key]
                continue
            kept.append(p)
        s = ",".join(kept)
    else:
        m = _RE_STATE.findall(s)
        if m:
            state = _STATE_LOOKUP[m[-1]]
            s = _RE_STATE.sub(" ", s)
    # 2-letter state codes as their own comma part; the last such part wins ("..., Al, Mumbai, MH, ..." -> MH)
    parts = [p.strip() for p in s.split(",")]
    if ADMIN_V2:
        kept = []
        for p in parts:
            key = " ".join(_RE_COMP_CLEAN.sub(" ", p).split())
            code = FR_ADMIN.get(key)
            if code:
                state = state or code
                continue
            kept.append(p)
        parts = kept
    if not state:
        for p in reversed(parts):
            pp = p.replace(".", "").strip()
            if len(pp) == 2 and (pp in _US_CODES or pp in IN_CODES):
                state = pp if pp in _US_CODES else IN_CODES[pp]
                break
    keep = []
    for p in parts:
        pp = p.replace(".", "").strip()
        if len(pp) == 2 and (pp == state or IN_CODES.get(pp) == state):
            continue
        keep.append(p)
    s = " , ".join(keep)
    s = s.replace("#", " ").replace("&", " and ")
    s = re.sub(r"(?<=[a-z])\.(?=[a-z]\b)", "", s)
    s = _RE_ORD.sub(r"\1", s)
    # keep codes like c-97, 1-84/4h together as one token, but split off plain punctuation
    s = re.sub(r"[^a-z0-9/\- ]+", " ", s)
    s = re.sub(r"(?<![a-z0-9])[/\-]+|[/\-]+(?![a-z0-9])", " ", s)
    out = []
    for t in s.split():
        t = NUM_WORDS.get(t, t)
        t = ADDR_MAP.get(t, t)
        out.append(t)
    nums = []
    for t in out:
        for d in re.findall(r"\d+", t):
            d = d.lstrip("0") or "0"
            nums.append(d)
    first_num = nums[0] if nums else ""
    return {"a": " ".join(out), "nums": " ".join(sorted(set(nums))), "state": state, "first_num": first_num,
            "comps": len(comps)}


def addr_tokens_for_match(a):
    """Address tokens used in similarity: drop noise words, split codes into digit runs and letters."""
    toks = []
    for t in a.split():
        if t in ADDR_NOISE:
            continue
        if any(c.isdigit() for c in t):
            toks.extend(d.lstrip("0") or "0" for d in re.findall(r"\d+", t))
        else:
            toks.append(t)
    return toks
