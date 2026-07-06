# Instruments

drift-monitor uses three complementary instruments to detect different dimensions of behavioral drift. Each operates independently on the same dual-window observations and produces a score from 0.0 (no drift) to 1.0 (complete drift).

## Ghost Lexicon

**Source:** `drift_monitor/instruments/ghost_lexicon.py`

Detects loss of low-frequency, high-precision vocabulary after compression. When an agent loses access to specialized terms like "idempotent", "memoization", or "eigendecomposition" and starts using generic alternatives, Ghost Lexicon catches it.

### Algorithm

1. **Tokenize** all text in each window using a regex tokenizer (`[a-zA-Z][a-zA-Z0-9_-]*(?:'[a-zA-Z]+)?`), lowercased.
2. **Extract specialized vocabulary** from the anchor window: terms that appear ≥ `min_freq` times (default 2), are ≥ `min_length` characters (default 4), and are not stop words. The stop word list is a hardcoded frozenset of ~100 common English words (`_STOP_WORDS`).
3. **Compute decay:** For each anchor specialized term, check if it appears in the recent window (using `min_freq=1` for recent — only presence matters). Score = fraction of anchor specialized terms missing from recent.

### Configuration

```python
GhostLexicon(
    window_size=50,   # max observations per window
    min_freq=2,       # minimum frequency in anchor to be "specialized"
    min_length=4,     # minimum term length
)
```

### Thresholds

| Threshold | Value |
|---|---|
| `high_threshold` | 0.3 |
| `moderate_threshold` | 0.1 |

### Reading Details

The `read()` method returns details including:
- `anchor_vocab_size`: number of specialized terms in anchor
- `recent_vocab_size`: number of specialized terms in recent
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
)
```

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

## Adding a New Instrument

1. Create a new file in `drift_monitor/instruments/`.
2. Subclass `Instrument` from `drift_monitor/instruments/base.py`.
3. Set `name`, `high_threshold`, `moderate_threshold`.
4. Implement `observe(text, metadata)`, `score() → float`, `reset()`.
5. Optionally override `read()` to include custom details.
6. Add the instrument to `DriftScorer` weights and the public API in `__init__.py`.
7. Add tests in `tests/`.
8. Add a drift mode to `drift_monitor/simulate.py` if applicable.
