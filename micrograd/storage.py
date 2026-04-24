# micrograd/storage.py
import numpy as np
from abc import ABC, abstractmethod


class Storage(ABC):
    """Abstract storage layer for unified CPU/GPU data access interface"""

    @property
    @abstractmethod
    def device(self) -> str:
        """Device type: 'cpu' or 'cuda'"""
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Data shape"""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Data type"""
        pass

    @abstractmethod
    def numpy(self) -> np.ndarray:
        """Get numpy array"""
        pass

    @abstractmethod
    def copy_to(self, other: "Storage"):
        """Copy data to another Storage"""
        pass

    @property
    def numel(self) -> int:
        """Number of elements"""
        return int(np.prod(self.shape))

    @abstractmethod
    def ptr(self) -> int:
        """Memory pointer"""
        pass


class CPUStorage(Storage):
    """CPU storage: numpy ndarray"""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def numpy(self) -> np.ndarray:
        return self._data

    def ptr(self) -> int:
        return self._data.ctypes.data

    def copy_to(self, other: Storage):
        if other.device == "cpu":
            other._data[:] = self._data
        else:
            raise NotImplementedError("CUDA not yet implemented")