from typing import TypedDict, Optional


class Sample(TypedDict):
    name: str
    index: list[int]
    pressure: list[float]


# GIXOS fitting does not use a separate water reference, but kept for parity
REFERENCE: Optional[Sample] = None

SAMPLES: list[Sample] = [
    {"name": "azotrans", "index": [49, 53, 57, 61], "pressure": [0.5, 10, 20, 30]},
    {"name": "azocis", "index": [77, 81, 85, 89], "pressure": [5, 10, 20, 30]},
    {"name": "azocis02", "index": [105, 109], "pressure": [3.3, 30]},
    {"name": "azocis03", "index": [114, 118], "pressure": [0.1, 30]},
    {"name": "dopc", "index": [10, 15, 19, 23], "pressure": [0.1, 10, 20, 30]},
    {"name": "redazo", "index": [127, 131, 135, 139], "pressure": [0.1, 10, 20, 30]},
]
SAMPLES_TEST: list[Sample] = [
    {"name": "azotrans", "index": [49], "pressure": [0.5]},
]


def get_samples(test: bool = False) -> list[Sample]:
    """
    Returns a list of samples to process.

    Args:
        test: If True, returns a single sample for testing.

    Returns:
        A list of samples.
    """
    return SAMPLES_TEST if test else SAMPLES

