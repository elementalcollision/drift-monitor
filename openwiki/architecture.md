# Architecture

This page covers the core architecture of drift-monitor: the dual-window system, instrument base class, composite scoring, and compression-type classification.

## Data Flow

```
Observations → DualWindow → [Anchor Window (pre-boundary), Recent Window (post-boundary)]
                                     ↓                        ↓
                              ┌──────┼──────┐
                         GhostLexicon  BehavioralFootprint  SemanticDrift
                              └──────┼──────┘
                                     ↓
                              DriftScorer (weighted composite)
                                     ↓
                              DriftReport
                          ├── composite_score (0.0–1.0)
                          ├── compression_type (enum)
                          ├── severity (LOW/MODERATE/HIGH)
                          └── readings[] (per-instrument detail)
```

## DualWindow

**Source:** `drift_monitor/window.py`

The `DualWindow` manages two `ObservationWindow` instances — `anchor` and `recent` — that hold observations from before and after a compression boundary.

- Observations flow into the **anchor** window until `mark_boundary()` is called.
- After the boundary is marked, subsequent observations flow into the **recent** window.
- Each window is a fixed-size `deque` (default 50) — older observations age out automatically.
- `has_enough_data(min_observations=3)` checks that both windows have sufficient data for comparison.

```python
class DualWindow:
    def __init__(self, window_size: int = 50) -> None
    def add(self, text: str, metadata: dict | None = None) -> None
    def mark_boundary(self) -> None
    def reset(self) -> None
    def has_enough_data(self, min_observations: int = 3) -> bool
```

An `Observation` is a simple dataclass holding `text: str` and `metadata: dict[str, Any]`.

## Instrument Base Class

**Source:** `drift_monitor/instruments/base.py`

All instruments inherit from the `Instrument` ABC. Each instrument:

1. **Observes** text + metadata via `observe(text, metadata)`.
2. **Marks the boundary** via `mark_boundary()` (delegates to its internal `DualWindow`).
3. **Scores** drift via `score() → float` (0.0 = no drift, 1.0 = complete drift).
4. **Reads** a full `InstrumentReading` via `read()`, which includes severity classification.

```python
class Instrument(ABC):
    name: str = ""
    high_threshold: float = 0.3
    moderate_threshold: float = 0.1

    @abstractmethod
    def observe(self, text: str, metadata: dict | None = None) -> None
    @abstractmethod
    def score(self) -> float
    @abstractmethod
    def reset(self) -> None

    def read(self) -> InstrumentReading  # calls score() + _classify()
    def _classify(self, score: float) -> Severity
```

### Severity Classification

Each instrument classifies its score using two thresholds:

| Severity | Condition |
|---|---|
| `LOW` | score ≤ `moderate_threshold` |
| `MODERATE` | `moderate_threshold` < score ≤ `high_threshold` |
| `HIGH` | score > `high_threshold` |

An instrument is considered to have **"fired"** when its severity is not `LOW` (i.e., score exceeds the moderate threshold). The firing pattern determines the compression type.

### InstrumentReading

```python
@dataclass
class InstrumentReading:
    instrument: str          # e.g. "ghost_lexicon"
    score: float             # 0.0–1.0
    severity: Severity       # LOW / MODERATE / HIGH
    details: dict[str, Any]  # instrument-specific data
```

## DriftScorer

**Source:** `drift_monitor/scoring.py`

The `DriftScorer` takes a list of `InstrumentReading` objects and produces a `DriftReport`.

### Composite Score

Weighted average of instrument scores:

```python
composite = Σ(reading.score × weight) / Σ(weight)
```

Default weights:

| Instrument | Weight |
|---|---|
| `ghost_lexicon` | 0.35 |
| `behavioral_footprint` | 0.35 |
| `semantic_drift` | 0.30 |

Weights are configurable: `DriftScorer(weights={...})`.

### Overall Severity

| Severity | Composite Score |
|---|---|
| `LOW` | ≤ 0.1 |
| `MODERATE` | 0.1 < score ≤ 0.3 |
| `HIGH` | > 0.3 |

### CompressionType Classification

The classification is based on **which instruments fired** (severity ≠ LOW):

```python
fired = {r.instrument for r in readings if r.severity != Severity.LOW}
```

| Condition | CompressionType |
|---|---|
| None fired | `NONE` |
| All three fired | `FULL_BOUNDARY` |
| Behavioral only (no Ghost) | `INFRASTRUCTURE` |
| Ghost + Behavioral (no Semantic) | `OPERATIONAL` |
| Ghost only (no Behavioral, no Semantic) | `VOCABULARY_ONLY` |
| Semantic only (no Ghost, no Behavioral) | `SEMANTIC_ONLY` |
| Any other combination | `FULL_BOUNDARY` (fallback) |

> **Note:** The README's classification table says `OPERATIONAL` = "Ghost Lexicon + Behavioral," but the source code in `_classify_compression()` also requires Semantic to *not* fire for `OPERATIONAL`. If all three fire, it's `FULL_BOUNDARY`. See `drift_monitor/scoring.py` lines 113–147.

## DriftReport

```python
@dataclass
class DriftReport:
    composite_score: float
    compression_type: CompressionType
    readings: list[InstrumentReading]
    severity: Severity
    details: dict[str, Any]

    @property
    def fired(self) -> list[str]  # names of instruments that exceeded moderate threshold
    def to_dict(self) -> dict     # JSON-serializable
```

## Public API

**Source:** `drift_monitor/__init__.py`

The package exports:

```python
from drift_monitor import GhostLexicon, BehavioralFootprint, SemanticDrift, DriftScorer, DriftReport, CompressionType
```

## Where to Start When Modifying

- **Adding a new instrument:** Subclass `Instrument`, implement `observe`, `score`, `reset`. Add it to the scorer's weights. See `drift_monitor/instruments/base.py` for the ABC.
- **Changing classification logic:** Edit `DriftScorer._classify_compression()` in `drift_monitor/scoring.py`.
- **Changing severity thresholds:** Edit `high_threshold` / `moderate_threshold` on the instrument class, or the composite thresholds in `DriftScorer.score()`.
- **Changing window behavior:** Edit `DualWindow` in `drift_monitor/window.py`.
