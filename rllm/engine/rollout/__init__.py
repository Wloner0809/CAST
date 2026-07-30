# Avoid importing concrete engines at module import time to prevent circular imports
from .rollout_engine import ModelOutput, RolloutEngine

__all__ = [
    "ModelOutput",
    "RolloutEngine",
    "OpenAIEngine",
    "VerlEngine",
    "SglangEngine",
]


def __getattr__(name):
    if name == "OpenAIEngine":
        from .openai_engine import OpenAIEngine as _OpenAIEngine

        return _OpenAIEngine
    if name == "VerlEngine":
        try:
            from .verl_engine import VerlEngine as _VerlEngine

            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    if name == "SglangEngine":
        try:
            from .sglang_engine import SglangEngine as _SglangEngine

            return _SglangEngine
        except Exception:
            raise AttributeError(name) from None
    raise AttributeError(name)
