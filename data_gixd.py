from typing import TypedDict


class Sample(TypedDict):
    name: str
    full_name: str
    index: list[int]
    pressure: list[float]


WATER: Sample = {"name": "water", "full_name": "water", "index": [44], "pressure": [0]}
SAMPLES: list[Sample] = [
    {
        "name": "azotrans",
        "full_name": "AzoPC-trans",
        "index": [54, 58, 62],
        "pressure": [10, 20, 30],
    },
    {
        "name": "azocis",
        "full_name": "AzoPC-cis",
        "index": [78, 82, 86, 90, 106, 110, 115, 119],
        "pressure": [5, 10, 20, 30, 3.3, 30, 0.1, 30],
    },
    # {
    #     "name": "dopc",
    #     "full_name": "DOPC",
    #     "index": [16, 12, 20, 24],
    #     "pressure": [0.1, 10, 20, 30],
    # },
    # {
    #     "name": "redazo",
    #     "full_name": "Red AzoPC",
    #     "index": [128, 132, 136, 140],
    #     "pressure": [0.1, 10, 20, 30],
    # },
]
SAMPLES_TEST: list[Sample] = [
    {
        "name": "azotrans",
        "full_name": "AzoPC-trans",
        "index": [54, 58, 62],
        "pressure": [10, 20, 30],
    },
]

# Test toggle - set to True to use SAMPLES_TEST, False to use SAMPLES
IS_TEST = False


def get_samples() -> list[Sample]:
    """
    Returns a list of samples to process.
    Uses IS_TEST flag to determine which sample list to return.

    Returns:
        A list of samples.
    """
    return SAMPLES_TEST if IS_TEST else SAMPLES


def get_water() -> Sample:
    """
    Returns the reference sample (water).

    Returns:
        The water sample.
    """
    return WATER


ROI_IQ = [0.7, 2.0, 0, 10]  # [q_min, q_max, tau_min, tau_max]
ROI_ITAU = [1.25, 1.5, 0, 50]  # [q_min, q_max, tau_min, tau_max]
