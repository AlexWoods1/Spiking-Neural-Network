# * Deterministic sub-seeds derived from one experiment ``seed``.
MODEL_SEED_OFFSET = 0
ENCODING_SEED_TRAIN_OFFSET = 1
ENCODING_SEED_VAL_OFFSET = 2
ENCODING_SEED_TEST_OFFSET = 3
SHUFFLE_SEED_OFFSET = 4


def derived_seed(seed: int, offset: int) -> int:
    """Return a deterministic sub-seed for an isolated RNG stream."""
    return seed + offset
