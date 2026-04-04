"""Basic usage example: measure drift between pre/post compression windows."""

from drift_monitor import GhostLexicon, BehavioralFootprint, SemanticDrift, DriftScorer

# Simulate agent outputs before compression
# Note: ghost lexicon needs repeated specialized terms across the window
pre_compression = [
    "The idempotent retry handler uses memoization to cache results. "
    "When backpressure exceeds the threshold, apply throttle logic.",
    "The idempotent guarantee ensures memoization works across retries. "
    "Monitor backpressure to decide when throttle kicks in.",
    "Without idempotent writes, the memoization cache may return stale data. "
    "The backpressure signal triggers throttle on the upstream queue.",
    "Check idempotent status before memoization lookup to avoid duplicates. "
    "If backpressure is high, the throttle prevents cascade failures.",
    "The idempotent handler retries with memoization, using backpressure "
    "metrics to decide the throttle rate for downstream consumers.",
]

# Simulate agent outputs after compression (specialized terms lost)
post_compression = [
    "The handler uses caching for results. "
    "When load exceeds the limit, apply rate limiting.",
    "The guarantee ensures caching works across retries. "
    "Monitor load to decide when limiting kicks in.",
    "Without proper writes, the cache may return stale data. "
    "The load signal triggers limiting on the upstream queue.",
    "Check status before cache lookup to avoid duplicates. "
    "If load is high, the limiter prevents cascade failures.",
    "The handler retries with caching, using load "
    "metrics to decide the rate for downstream consumers.",
]

# Create instruments
ghost = GhostLexicon()
behavioral = BehavioralFootprint()
semantic = SemanticDrift(use_embeddings=False)

instruments = [ghost, behavioral, semantic]

# Feed pre-compression observations
for text in pre_compression:
    for instr in instruments:
        instr.observe(text, {"tools": ["read_file", "grep"]})

# Mark the compression boundary
for instr in instruments:
    instr.mark_boundary()

# Feed post-compression observations
for text in post_compression:
    for instr in instruments:
        instr.observe(text, {"tools": ["read_file", "grep"]})

# Get individual readings
readings = [instr.read() for instr in instruments]

# Get composite report
scorer = DriftScorer()
report = scorer.score(readings)

print("=== Individual Instruments ===")
for reading in readings:
    print(f"  {reading.instrument:25s} score={reading.score:.4f}  severity={reading.severity.value}")

print(f"\n=== Composite ===")
print(f"  Score: {report.composite_score:.4f}")
print(f"  Type:  {report.compression_type.value}")
print(f"  Fired: {report.fired}")

# Show ghost lexicon details
ghost_reading = ghost.read()
if ghost_reading.details.get("lost_terms"):
    print(f"\n=== Lost Terms ===")
    for term in ghost_reading.details["lost_terms"]:
        print(f"  - {term}")
