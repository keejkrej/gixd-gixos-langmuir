from typing import TypedDict


class Sample(TypedDict):
    name: str
    index: list[int]
    pressure: list[float]


WATER: Sample = {
    "name": "water",
    "index": [44],
    "pressure": [0],
}
SAMPLES: list[Sample] = [
    {"name": "azotrans", "index": [54, 58, 62], "pressure": [10, 20, 30]},
    {"name": "azocis", "index": [78, 82, 86, 90], "pressure": [5, 10, 20, 30]},
    {"name": "azocis02", "index": [106, 110], "pressure": [3.3, 30]},
    {"name": "azocis03", "index": [115, 119], "pressure": [0.1, 30]},
    {"name": "dopc", "index": [16, 12, 20, 24], "pressure": [0.1, 10, 20, 30]},
    {"name": "redazo", "index": [128, 132, 136, 140], "pressure": [0.1, 10, 20, 30]},
]
