---
type: "Reference"
title: "Instruments"
description: "The four drift instruments — Ghost Lexicon, Behavioral Footprint, Semantic Drift, and opt-in Hedging Language — their algorithms, thresholds, configuration, and the permutation-test significance layer."
tags: [instruments, ghost-lexicon, behavioral, semantic, hedging, significance]
---

# Instruments

drift-monitor uses three complementary instruments to detect different dimensions of behavioral drift, plus an opt-in fourth (Hedging Language) for confidence erosion. Each operates independently on the same dual-window observations and produces a score from 0.0 (no drift) to 1.0 (complete drift).

## Ghost Lexicon

**Source:** `drift_monitor/instruments/ghost_lexicon.py`

Detects loss of low-frequency, high-precision vocabulary after compression. When an agent loses access to specialized terms like "idempotent", "memoization", or "eigendecomposition" and starts using generic alternatives, Ghost Lexicon catches it.

### Algorithm

1. **Tokenize** all text in each window using a regex tokenizer (`[^\W\d_][\w-]*(?:'[\w]+)?`, Unicode-aware — non-English text tokenizes correctly, and pure-ASCII input yields the same tokens as the historical `[a-zA-Z]` pattern), lowercased.
2. **Extract specialized vocabulary** from the anchor window: terms that appear ≥ `min_freq` times (default 2), are ≥ `min_length` characters (default 4), and are not stop words. The stop word list is a hardcoded frozenset of ~100 common English words (`_STOP_WORDS`).
3. **Compute decay:** For each anchor specialized term, check if it appears in the recent window (using `min_freq=1` for recent — only presence matters). Score = fraction of anchor specialized terms missing from recent. With `idf_weighting=True`, each lost term is instead weighted by its within-window inverse document frequency, so losing a rare (concentrated) jargon term moves the score more than losing a term that appears in every observation.

### Configuration

```python
GhostLexicon(
    window_size=50,   # max observations per window
    min_freq=2,       # minimum frequency in anchor to be "specialized"
    min_length=4,     # minimum term length
    idf_weighting=False,  # opt-in rarity-weighted decay (see below)
)
```

### Thresholds

| Threshold | Value |
|---|---|
| `high_threshold` | 0.3 |
| `moderate_threshold` | 0.1 |

### Reading Details

The `read()` method returns details including:
- `anchor_specialized_count`: number of specialized terms in anchor
- `lost_count`: number of specialized terms missing from recent
- `lost_terms`: list of terms present in anchor but missing from recent

### Score Interpretation

| Score | Meaning |
|---|---|
| 0.0 | All specialized terms retained |
| 0.5 | Half of specialized vocabulary lost |
| 1.0 | Complete vocabulary collapse |

## Behavioral Footprint

**Source:** `drift_monitor/instruments/behavioral.py`

Tracks shifts in tool-call ratios and response shape. If an agent was using `search`, `code`, and `analyze` tools evenly but post-compression only uses `search`, that's behavioral drift.

### Algorithm

1. **Build a `BehaviorFingerprint`** for each window containing:
   - `tool_distribution`: normalized frequency of each tool name from `metadata["tools"]`
   - `avg_response_length`: mean character length of texts
   - `response_length_std`: standard deviation of response lengths
   - `total_observations`: count
2. **Compute tool distribution distance:** Total variation distance between anchor and recent tool distributions: `Σ|p_a(k) - p_r(k)| / 2`, bounded [0, 1].
3. **Compute length shift:** `|avg_anchor - avg_recent| / max(avg_anchor, avg_recent, 1.0)`, bounded [0, 1].
4. **Combine:** `score = tool_weight × tool_dist + length_weight × length_dist` (default weights 0.6 and 0.4). If no tools were observed in either window, score is based entirely on length shift.

### Configuration

```python
BehavioralFootprint(
    window_size=50,
    tool_weight=0.6,    # weight for tool distribution distance
    length_weight=0.4,  # weight for response length shift
)
```

### Thresholds

| Threshold | Value |
|---|---|
| `high_threshold` | 0.3 |
| `moderate_threshold` | 0.1 |

### Reading Details

Includes `anchor_fingerprint`, `recent_fingerprint` (each with `tool_distribution`, `avg_response_length`, `response_length_std`, `total_observations`), `tool_distance`, and `length_shift`.

### Metadata Format

Pass tool usage as a list in metadata:

```python
instrument.observe(text, metadata={"tools": ["read_file", "grep", "edit_file"]})
```

If `tools` is a string instead of a list, it's treated as a single tool.

## Semantic Drift

**Source:** `drift_monitor/instruments/semantic.py`

Measures movement of the conceptual center-of-gravity. Detects when an agent's focus shifts from, say, "architecture exploration" to "learning rate tuning."

### Two Modes

#### Keyword Overlap (zero-dependency default)

1. Build keyword frequency distributions for anchor and recent texts (excluding stop words, tokens ≤ 2 chars).
2. Compute cosine similarity between the two distributions.
3. Score = `1.0 - similarity` (bounded [0, 1]).

#### Embedding Centroid (optional, requires `sentence-transformers`)

1. Encode all anchor and recent texts using a `SentenceTransformer` model (default: `all-MiniLM-L6-v2`).
2. Compute the centroid (mean vector) for each window.
3. Compute cosine similarity between centroids.
4. Score = `1.0 - similarity` (bounded [0, 1]).

### Configuration

```python
SemanticDrift(
    window_size=50,
    model_name="all-MiniLM-L6-v2",  # only used with embeddings
    use_embeddings=None,  # None = auto-detect; True/False to force
    idf_weighting=False,  # opt-in tf-idf weighting (keyword mode only)
)
```

`idf_weighting=True` turns the keyword-overlap distribution into **tf-idf**: each term's frequency is multiplied by its within-window inverse document frequency, so rare high-information terms dominate the similarity instead of common non-stop words. It has no effect in embedding mode.

Install embedding support:

```bash
pip install "drift-monitor[embeddings]"
# or
pip install sentence-transformers>=2.0.0
```

### Thresholds

Semantic drift uses **lower thresholds** than the other instruments — it's more sensitive to subtle shifts:

| Threshold | Value |
|---|---|
| `high_threshold` | 0.15 |
| `moderate_threshold` | 0.05 |

### Reading Details

Includes `method`: either `"embedding"` or `"keyword"`.

### Auto-Detection

If `use_embeddings` is `None` (default), the instrument auto-detects whether `sentence_transformers` is importable. The CLI and harness default to `use_embeddings=False` for deterministic zero-dependency operation.

## When to Use Which Mode

| Scenario | Recommended Mode |
|---|---|
| Quick CLI analysis, CI/CD | Keyword overlap (zero deps) |
| Sensitive detection of subtle topic shifts | Embedding centroid |
| Production experiment loops (harness) | Keyword overlap (default in `DriftHarness`) |

## Hedging Language (opt-in)

**Source:** `drift_monitor/instruments/hedging.py`

Detects erosion of assertive language: "must" becomes "could potentially", "requires" becomes "might benefit from". The surface trio stays silent on this (vocabulary preserved, topic unchanged), so it is a separate, opt-in instrument — it is **not** part of `DriftMonitor`'s default set, and including its readings in a `DriftScorer` composite requires passing explicit weights (unknown instruments default to equal weight, which dilutes the trio).

- **Score:** one-sided — only an *increase* in hedge density (hedges per word) from anchor to recent counts. A rise of 5 hedges per 100 words saturates at 1.0. Growing confidence is not treated as drift.
- **Hedge list:** multi-word qualifiers and hedging adverbs only; the bare modals "may"/"might"/"could" are deliberately **excluded** because they are ubiquitous in ordinary technical prose (and "May" would collide with the month under case-insensitive matching). This keeps real-text false positives down while still catching FRAMING's "could potentially" / "might benefit" substitutions.
- **Scope note:** this is a lexical detector for hedging *style*. `DriftMode.FRAMING`'s invisibility guarantee applies to the three surface instruments; FRAMING's synthetic substitutions are literally hedging qualifiers, so this instrument fires on them (by design).

> **Research item (open):** detection *precision* on real text is unvalidated. The lexical hedge list — even after dropping the bare modals — is exercised only against synthetic framing substitutions. Whether "count the multi-word qualifiers" is the right model for confidence erosion, and how often ordinary agent prose trips it, are open questions. Before leaning on this instrument in production: build a real-text benchmark (pre/post genuine agent outputs) and compare against an embedding/classifier baseline.

### Configuration

```python
HedgingLanguage(window_size=50)
```

### Thresholds

| Threshold | Value |
|---|---|
| `high_threshold` | 0.5 |
| `moderate_threshold` | 0.2 |

## Significance Testing

**Source:** `drift_monitor/significance.py`

Composites over small windows can be noise. `permutation_test()` pools the anchor + recent observations, reshuffles them across a sham boundary `n_permutations` times, recomputes the composite, and reports `p_value` — the probability of drift this strong if the boundary were arbitrary. `monitor_significance(monitor)` is the convenience wrapper; `drift-monitor run --permutations 200` adds a `significance` block to the CLI output. Null recomputations use keyword-mode semantics; this is an offline analysis tool, not a hot-path call. `monitor_significance` only accepts a monitor whose composite it can faithfully reproduce (the default trio in keyword mode); a custom instrument set or embedding-mode semantics raises `ValueError`, since a silently mismatched `observed_composite` would make the p-value meaningless — use `permutation_test()` directly in those cases.

## Adding a New Instrument

1. Create a new file in `drift_monitor/instruments/`.
2. Subclass `Instrument` from `drift_monitor/instruments/base.py`.
3. Set `name`, `high_threshold`, `moderate_threshold`.
4. Implement `observe(text, metadata)`, `score() → float`, `reset()`.
5. Optionally override `read()` to include custom details.
6. Add the instrument to `DriftScorer` weights and the public API in `__init__.py`.
7. Add tests in `tests/`.
8. Add a drift mode to `drift_monitor/simulate.py` if applicable.
