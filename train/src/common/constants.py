from enum import Enum
from typing import List


class WeightsInitializer(str, Enum):
    Zeros = "zeros"
    He = "he"
    Xavier = "xavier"

    @staticmethod
    def all_names() -> List[str]:
        return [v.value for v in WeightsInitializer.__members__.values()]

    @staticmethod
    def is_valid(value: str) -> bool:
        return value in WeightsInitializer.all_names()
