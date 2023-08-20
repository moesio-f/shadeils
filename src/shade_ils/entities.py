from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class FitnessFunction(ABC):
    def __call__(self, population: np.ndarray) -> np.ndarray:
        return self.call(population)

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def call(self, population: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def info(self) -> dict:
        """Retorna informações da função, sempre
        vão existir pelo menos 3 chaves:

        - 'lower': float
        - 'upper': float
        - 'dimension': int
        """


@dataclass(frozen=True)
class EAResult:
    fitness: np.ndarray
    solution: np.ndarray
    evaluations: int
    best_index: int | None = None
