# V6 research: where the remaining F0.5 is, how 0.997 is possible, and what to build

> **Implemented as V6** — see `V6.md` for the final design, the run command (`bash run_all.sh --stage v6all`) and the decision rules. This file is the data study and literature behind it.

Everything here was measured on files in this repository: 3,000 real true pairs, 40 real multi-record entities,
the validation error dumps, the EDA, the France test samples, and the phase-2 ablation from the H100 run. Items
marked **(to measure)** need the H100 and come with a ready script. Nothing is extrapolated from synthetic data.

## 0. Summary

1. **Our leaderboard loss is mainly a US test-universe shift, not France and not model capacity.** Test US has
   half the S1 density of train US (0.66 M vs 1.32 M). India is at 92 %. Every system over-accepts on test US
   compared with validation, and the leaderboard falls in exactly that order: V2 +0.95 % → 0.98332,
   V3 +2.9 % → 0.98281, V4A +3.5 % → 0.981. India stays within ±0.2 % for all of them. The GNN, which reads
   graph and count features, inflates the most (+9.3 %).
2. **Best untested submission right now: "V3 without GNN"** (stage-1 blend + XLM-R stack). Validation 0.98951
   (V2 0.98696), and the most consistent US behaviour of any system (+0.6 %). Its test decisions are already
   cached; `run_all.sh --only v6_v3nognn` writes the file.
3. **Decisive next experiment (≈2 h, labels available).** Rebuild the training universe at test density
   (`simulate_universe.py`), then score the unchanged V2 chain on it (`score_chain.py`). If the US gap reproduces,
   we can measure and fix the test problem on labelled data for the first time. Retraining the V2 recipe in that
   universe (`--stage v6a`) is the direct fix.
4. **The data comes from a hierarchical generator:** base entity → per-source version → per-record noise.
   Same-source siblings share address details that the S1 lacks (§1.3). That's a strong collective signal we
   barely use, and it targets our biggest error class (missed pairs: 31.7 k FN vs 1.6 k FP on V3 validation).
5. **Realistic odds:** 0.986–0.988 after the density fix; 0.990–0.993 with collective sibling evidence and a
   stronger cross-encoder; 0.995+ only if the sibling and count-prior signals are as strong at scale as in the
   samples. **0.999 is not realistic.** It would need near-perfect decisions on records that are ambiguous by
   content (empty address + a name shared by several S1) and on twin distractors that copy an S1 exactly.

## 1. What the data is

### 1.1 Noise operators (3,000 real true pairs, after our normaliser)

| name side | share | address side | share |
|---|---|---|---|
| normalised key equal | 59.4 % | same first house number | 74.6 % |
| record drops key word(s) (e.g. "Midwest Academy" → "Midwest Center") | 8.7 % | record lost the house number | 10.1 % |
| single-token typo ("Risoce", "Haemhis", "Merdicine") | 6.8 % | **house number differs** (both present) | **14.8 %** |
| added words: "Orbinex formerly …", "d/b/a", duplicated word | 4.8 % | address empty | 4.9 % |
| non-Latin script (Devanagari, Telugu, Kannada …) | 7.2 % | address token set equal | 40.6 % |
| domain form / hashtag ("#Cóopersilver", "…sarl.com") | 6.2 % | record tokens ⊂ S1 tokens | 60.2 % |
| all-caps (S2 22.7 %, S3 2.3 %) | 12.5 % | state equal | 85.5 % |
| bracketed legal / word ("[Ltd]", "(LLC)") | 7.6 % | S2 address all-caps | 59.7 % |
| accent noise ("Gároa", "Límited", "Çhasse") | 6.5 % | S3 address with full state name | 39.0 % (S2 0.2 %) |
| honorific prefix, leet digits ("Upt0wn"), prefix junk ("--", ">>") | 3.8 / 1.6 / 1.3 % | | |

Each source has its own format: S2 is upper case with abbreviations and state codes; S3 is title case, spells out
the state, and reorders components. Treating source identity as information (source-specific comparators) is cheap
and correct.

### 1.2 Hard cases (validation error dumps)

* **Hard positives:** a DBA/new name with no overlap ("Sharp Fresh Physical PLLC" ↔ "Zetazetaquo"), a different
  house number on the same street ("5804 Forest Haven Trail" ↔ "5784-C Forest Haven Trl"), initials
  ("Garoa" ↔ "Gna"), and native-script names with a truncated address.
* **Hard negatives ("twins"):** distractor records that copy an S1's name and address almost exactly but belong to
  no S1 in the ground truth. There are few of them (V3 validation FP = 1.6 k), and they are the only thing capping
  precision.
* **Empty-address records:** a name shared by several S1 in different cities ("Oncology Custom Medicine LLC" in
  UT and OR; "Golden Nails" in MO and VA). In the old error analysis, **more than half of all missed pairs** involve
  an empty-address record: 14.6 k blocking misses, 9.7 k below threshold, 9.6 k lost to a same-name S1.

### 1.3 The generator is hierarchical (the key structural finding)

In the 40 real multi-record entities:

| pair type | identical normalised address | name token-set = 100 |
|---|---|---|
| record ↔ its S1 | 76 % | 71 % |
| **same-source siblings** (two S2 records of one S1) | **81 %** | 54 % |
| cross-source siblings (S2 ↔ S3 of one S1) | 60 % | 57 % |

When a record's house number differs from S1's, then in **8 of 12** source groups at least two siblings share
*the same* different number (all S2 records of "Construction Mayra" say "4/2" where the S1 says "5/2"; all S3
records of "Vishnupriya Brothers" add "Noida Iii Sector Xviii", which the S1 lacks).
So each source holds a version of the entity, and records are noisy copies of that version. **A hard record can be
linked through an easy sibling.** Our stage-2 sibling features only compare against *confident* records with one
cosine, and the GNN never compares record content with record content. **(to measure at scale: `data_probe.py` T2)**

### 1.4 Where the F0.5 is lost (validation)

V3 validation: FN 31,722 vs FP 1,629. **Recall is ~90 % of the loss.** Blocking misses ~1.2 % of true pairs
(≈ 80 % of them empty-address). A perfect scorer on today's candidates caps near 0.997 on validation.

## 2. What the leaderboard is telling us

| system | val F0.5 | US test/val predicted matches per S1 | US uncertain share val → test | India ratio | leaderboard |
|---|---|---|---|---|---|
| V2 | 0.98696 | +0.95 % | 4.9 → 6.1 % | +0.18 % | **0.98332** |
| V3 | 0.99078 | +2.92 % | 3.2 → 7.8 % | +0.06 % | 0.98281 |
| V4A | 0.99078 | +3.48 % | 3.2 → 8.9 % | +0.12 % | 0.981 |
| GNN only | 0.98841 | +9.31 % | 3.8 → 10.7 % | +4.3 % | – |
| **V3 without GNN** | **0.98951** | **+0.60 %** | 4.4 → 4.6 % | −0.25 % | not submitted |

| country | train S1 | SIM19 S1 | test S1 | test records per S1 |
|---|---|---|---|---|
| US | 1.32 M | 1.07 M | **0.66 M** | 5.76 |
| India | 0.88 M | 0.72 M | 0.81 M | 5.82 |
| France | – | – | **0.26 M** | 5.53 |

SIM19 fixed records-per-S1, but not how crowded each country's index is. Competition features (margins to the
runner-up, ranks) and counts (`b_n`, `a_n`, `key_n_s1`, `key_n_r`) take different values when half the competing
S1 are missing. Models learn P(match | features) in the dense universe and apply it in the sparse one. A
distractor's nearest S1 looks "unchallenged" and gets accepted. Content-based evidence (the XLM-R stack) is
immune; graph and count evidence (the GNN) is the most exposed. This also warns about V5, whose ambiguity and
frequency features are density-dependent. `choose_v5.py` now rejects any system whose US/India test/val ratio
exceeds V2's by more than 0.3 %.

## 3. How could the leaders reach 0.99717? (ranked)

1. **Density- and shift-robust scoring.** Their test/validation behaviour stays consistent (content-based scorers,
   or training at test density). This alone is probably worth +0.003–0.005 for us.
2. **Collective / cluster-level resolution** that exploits the per-source versions (§1.3) to recover hard positives
   and empty-address records. This is the only content-level route to recall ≈ 0.995.
3. **High-recall blocking** (≥ 99.7 %): exact-key passes, sibling-driven expansion, all same-key S1 for
   empty-address records.
4. **A generator artefact** such as file row order. We checked: IDs carry no signal (S1-id vs record-id
   correlation 0.0005; modular checks exactly at chance). Row order is measured by `data_probe.py` T1. If it were
   strong, **do not use it without the organisers' approval**: the top packages are reviewed for fair play.

## 4. Methods per pipeline step (with the literature)

| step | method | what it buys on this data | key references |
|---|---|---|---|
| normalisation | operator-inverting canonicalisation: legal-form families, street types, `N°/Nº/#`, bis/ter, accents, domain split, transliteration; keep the raw legal string as a separate feature | undoes 60 %+ of name noise; legal *surface* ("Limited" vs "Ltd") separates same-name S1 | Christen 2012 (Data Matching, Springer); Aksharantar/IndicXlit (Madhani et al., EMNLP Findings 2023) |
| field likelihoods | Fellegi–Sunter m/u weights per field and per operator, with Winkler string comparators, as features | calibrated, density-free evidence per field | Fellegi & Sunter 1969 (JASA); Winkler 1990 |
| blocking | multi-pass union (exact keys, q-gram TF-IDF, dense ANN), adaptive k, meta-blocking | recall 98.8 % → 99.5 %+ | Christen 2012 (IEEE TKDE survey); Papadakis et al. 2020 (ACM CSUR); DeepBlocker, Thirumuruganathan et al. 2021 (PVLDB) |
| pair scorer | GBDT on rich comparators; ranking loss (LambdaMART) per record | strong tabular baseline, rank-aware | Chen & Guestrin 2016 (KDD); Burges 2010 (MSR-TR) |
| neural matcher | cross-encoder on raw text, hard-negative augmentation | content evidence that transfers across density and country | Ditto, Li et al. 2021 (PVLDB); Brunner & Stockinger 2020 (EDBT); Peeters & Bizer 2021 (PVLDB) dual-objective; Peeters & Bizer 2022 (WWW) supervised contrastive |
| multilingual encoders | XLM-R-large (MIT), bge-reranker-v2-m3 / BGE-M3 (Apache), multilingual-e5 (MIT) | Indic scripts, French accents | Conneau et al. 2020 (ACL); Chen et al. 2024 (BGE M3, arXiv 2402.03216); Wang et al. 2024 (mE5, arXiv 2402.05672) |
| collective ER | iterative collective classification with record–record (sibling) evidence | hard positives and empty-address records via siblings | Bhattacharya & Getoor 2007 (ACM TKDD); Singla & Domingos 2006 (ICDM, Markov logic); Rastogi et al. 2011 (PVLDB) |
| multi-source clustering | clean-source constraint: S1 is duplicate-free, each record joins ≤ 1 S1 | global consistency, twin handling | FAMER, Saeedi et al. 2018; Christophides et al. 2020 (ACM CSUR overview) |
| assignment / decoding | constrained assignment (≤ 5 S2 / ≤ 6 S3 per S1) + expected-F decoding | fewer confusions between same-name S1 | Kuhn 1955 (Hungarian method); Jansche 2007 (ACL); Nan et al. 2012 (ICML) |
| calibration under shift | isotonic in the *target-like* universe; prior-shift correction | fixes the US inflation | Zadrozny & Elkan 2002 (KDD); Saerens et al. 2002 (Neural Computation); Lipton et al. 2018 (ICML, BBSE) |
| unseen country | domain adaptation, self-training with conservative pseudo-labels | France | Ganin & Lempitsky 2015 (ICML, DANN); DADER, Tu et al. 2022 (SIGMOD); Xie et al. 2020 (CVPR, Noisy Student) |
| LLM matcher (≤ 8 B, hard band only) | generative model as judge on the few thousand hardest pairs | last-mile precision | Peeters, Steiner & Bizer 2025 (EDBT, "Entity matching using LLMs") |
| zero-label generative | mixture model of match / non-match features | sanity check under shift | ZeroER, Wu et al. 2020 (SIGMOD) |

## 5. French ↔ English, in depth

Observed in the France test samples (S1 vs S2/S3):

* **Case and accents:** S2 upper case ("RUE DU MARÉCHAL FOCH"). The generator also *adds* accents to the first
  letter ("Çhasse", "Àmis", "Ècole", "Màrchands"). The same operator exists in US/India ("Gároa", "Límited").
  `anyascii` removes it, but "Çhasse" → "chasse" only works because Ç maps to C; keep folding before tokenising.
* **Number prefixes and suffixes:** `N° 74`, `Nº 26`, `NO. 10`, `No 52`, `# 3`, `(45)`, `41B`, `9 bis`, `4bis`.
  Bug: `N°` becomes the token `ndeg` (° → "deg"), which stays in the address and dilutes TF-IDF and jaccard.
  Fix: strip `n°|nº|no\.?|#` before a number, and map `bis/ter/quater` and trailing letters to one suffix class.
* **Street types:** R/R./RUE, AV/AV./AVE, BD, IMP/IMP., ALL/ALLÉE, PL, RTE, CHE/CHEM, QU, SQ, CRS, FBG, CITÉ,
  PASS, SENT. Most are in `normalize.ADDR_MAP`. Missing or ambiguous: `CITE`, `QU`, `RES`, and typos
  ("COUR2" → "cours 2"; our parser reads "2" as the house number).
* **Administrative level:** S1 uses regions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire); records use
  departments (Nord, Gironde, Loire-Atlantique) or nothing. Normalisation v2 maps departments → region
  (`state_cmp` for France went from 0 to 0.64).
* **Elisions and articles:** "Rue de l'Yser" / "RUE YSER", "d'Auray", "du/de la/des" dropped or kept, "Saint-" /
  "St-" / "ST ", hyphenated cities ("La Teste-de-Buch" / "LA TESTE DE BUCH"). Remove articles and elided `l'`/`d'`
  in the street core; treat `saint/st/ste` as one token.
* **Legal forms:** SAS, SASU, S.A.S.U., SARL, EURL, EI, SCI, SA, SNC, "Ets"/"Établissements", "& Fils",
  "& Frères", "Cie". These map to families (`LEGAL_FAMILY`). A SARL ↔ EURL swap is noise; SARL ↔ SAS is weak
  evidence of a different entity.
* **Generic filler words the generator adds for France:** "Groupe", "Participations", "Holding",
  "Développement", "International", "France", "(France)", "Services". They're the French equivalents of
  "Holdings/Services/Center" in US/India. Add "participations", "developpement", "gestion" to `GENERIC`, so the
  name *key* ignores them the same way. ("groupe", "holding", "france", "international", "services" are already
  there.)
* **Translation (meaning change):** a record *could* translate name words ("École" ↔ "School", "Pharmacie" ↔
  "Pharmacy", "Lycée" ↔ "High School", "Collège" ↔ "College", where the meaning actually changes from middle school
  to university, "Frères" ↔ "Brothers", "Fils" ↔ "Sons", "Compagnie/Cie" ↔ "Company/Co", "Établissements" ↔
  "Establishments"). The 50-name samples per source show **no translations**: English words there are
  international ("Services", "International", "Holding"). A 150-word FR↔EN lexicon canonicaliser is cheap
  insurance. Measure first: share of France top-candidate pairs whose keys become equal after the lexicon.
* **Why France is structurally hard:** S1 names come from a small vocabulary ("Nantes Club SAS", "Lille Compagnie
  SASU", "Association de Formation") in ~20 dense cities. The name carries little evidence; street core + number
  decides. Name-specificity and street-core features were built for this (V5).

## 6. V6 architecture (build in this order; each step has a gate)

| # | component | gate before keeping it |
|---|---|---|
| 0 | **Submit "V3 without GNN"** (already computed) | leaderboard ≥ V2 |
| 1 | **Density-matched training universe** (`simulate_universe.py`, `--stage v6a`): V2 recipe trained and calibrated at test density for the US | on the density universe, V6a beats the unchanged V2 chain; test/val ratio for US within ±0.3 % |
| 2 | normalisation v3: `ndeg` fix, number-prefix/suffix classes, French generic words, elisions, raw-legal-surface feature | France self-consistency (key equality of top pairs) ↑; val not worse |
| 3 | blocking v3: exact-key passes, all same-key S1 for empty-address records (capped), sibling expansion | candidate recall ≥ 99.5 %; val not worse |
| 4 | **collective sibling stage**: for each (S1, record), support from other records of that S1, split same-source / cross-source: identical normalised address, shared non-S1 house number, name/address similarity × sibling probability; iterate twice | on density universe folds 0 and 4: FN ↓, FP not ↑ |
| 5 | stronger cross-encoder (bge-reranker-v2-m3 or XLM-R-large) on a wider band, stacked like V3-without-GNN | LOCO India→US not worse; val ↑ |
| 6 | constrained assignment (per-source caps, count prior) + expected-F decoding; unseen-country γ from LOCO | val ↑, consistency holds |

## 7. What to run now (H100)

```bash
bash run_all.sh --stage v6diag   # 1) data probe  2) V3-without-GNN submission file  3) density universe  4) V2 chain on it
bash run_all.sh --stage v6a      # V2 recipe trained and calibrated at test density -> results/final/final_v6a/output/
```

Send back `results/v6/*.json`, `results/xgboost/xgb_v6a_s2/decode_eval.json` and `results/final/*/test_stats.json`.
Suggested submission order: `final_v3nognn` now; `final_v6a` if the density experiment confirms the effect;
V5 only if `v5_choice.json` passes the new consistency gate.

## 8. Honest odds (leaderboard, public 20 %)

| target | needs | my estimate |
|---|---|---|
| > 0.9833 (beat V2) | V3-without-GNN or V6a | 65–75 % |
| 0.986–0.988 | density fix works as the ablation suggests | 40–50 % |
| 0.990–0.993 | + collective siblings + stronger cross-encoder | 15–25 % |
| ≥ 0.995 | + blocking ≥ 99.5 % and sibling/count signals as strong at scale as in the samples | 5–10 % |
| 0.999 | near-perfect on content-ambiguous records and twin distractors | < 1 % |
