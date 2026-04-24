# micrograd/dispatch.py
from typing import Callable, Dict
import functools


class OpRegistry:
    """
    算子注册表，根据 device 自动 dispatch 到对应 kernel。

    支持两种注册方式：
    1. 装饰器: @registry.register_op("add", "cpu")
    2. 直接调用: registry.register("add", "cpu", forward_fn, backward_fn)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Dict[str, dict]] = {}
        return cls._instance

    def register_op(self, op_name: str, device: str, is_backward: bool = False):
        """
        装饰器：注册 forward 或 backward kernel。

        Usage:
            @registry.register_op("add", "cpu")
            def add_forward(a, b):
                ...

            @registry.register_op("add", "cpu", is_backward=True)
            def add_backward(grad_out, a, b):
                ...
        """
        def decorator(fn: Callable) -> Callable:
            if op_name not in self._registry:
                self._registry[op_name] = {}
            if device not in self._registry[op_name]:
                self._registry[op_name][device] = {"forward": None, "backward": None}

            key = "backward" if is_backward else "forward"
            self._registry[op_name][device][key] = fn

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    def register(self, op_name: str, device: str,
                 forward: Callable, backward: Callable):
        """直接注册 forward/backward kernel"""
        if op_name not in self._registry:
            self._registry[op_name] = {}
        self._registry[op_name][device] = {"forward": forward, "backward": backward}

    def dispatch(self, op_name: str, device: str) -> tuple[Callable, Callable]:
        """根据 op_name 和 device 选择 kernel"""
        if op_name not in self._registry:
            raise KeyError(f"Op '{op_name}' not registered")
        if device not in self._registry[op_name]:
            raise KeyError(f"Op '{op_name}' has no kernel for device '{device}'")
        entry = self._registry[op_name][device]
        return entry["forward"], entry["backward"]

    def get_devices(self, op_name: str) -> list[str]:
        """获取某算子支持的所有设备"""
        return list(self._registry.get(op_name, {}).keys())

    def is_registered(self, op_name: str, device: str) -> bool:
        """检查算子是否在某设备上注册"""
        return op_name in self._registry and device in self._registry[op_name]


# 全局单例
registry = OpRegistry()