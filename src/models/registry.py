"""Decorator-based architecture registry for model discovery and instantiation.

Provides a global registry pattern where model classes are registered via
the ``@register(arch_id)`` decorator and instantiated through the
``get_model`` factory function. All project models should subclass
``BaseModel`` for consistent weight initialisation, parameter counting,
and architecture summarisation.

Example::

    @register("my_model_v1")
    class MyModel(BaseModel):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            ...

    model = get_model("my_model_v1", input_size=128, output_size=10)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[nn.Module]] = {}


def register(arch_id: str):
    """Decorator that registers a model class under *arch_id*.

    Args:
        arch_id: Unique string identifier for the architecture.

    Returns:
        The original class, unmodified.

    Raises:
        ValueError: If *arch_id* is already registered.
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if arch_id in _REGISTRY:
            raise ValueError(
                f"Duplicate arch_id '{arch_id}': "
                f"{cls.__name__} conflicts with {_REGISTRY[arch_id].__name__}"
            )
        _REGISTRY[arch_id] = cls
        return cls

    return decorator


def get_model(arch_id: str, **config) -> nn.Module:
    """Instantiate a registered model by *arch_id*.

    Args:
        arch_id: The architecture identifier used in ``@register``.
        **config: Keyword arguments forwarded to the model constructor
            (typically ``input_size``, ``output_size``, and any
            architecture-specific hyperparameters).

    Returns:
        An ``nn.Module`` instance.

    Raises:
        ValueError: If *arch_id* is not found in the registry.
    """
    if arch_id not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture: '{arch_id}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[arch_id](**config)


def list_models() -> list[str]:
    """Return a sorted list of all registered architecture IDs."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class BaseModel(nn.Module):
    """Base class for all project models.

    Provides shared utilities for weight initialisation, parameter counting,
    and human-readable architecture summaries.  Subclasses must implement
    ``forward``.
    """

    # Supported initialisation methods
    _INIT_METHODS = {
        "kaiming_normal": nn.init.kaiming_normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
    }

    def init_weights(self, method: str = "kaiming_normal") -> None:
        """Initialise all weight tensors in the module.

        Args:
            method: One of ``"kaiming_normal"``, ``"xavier_uniform"``,
                or ``"xavier_normal"``.

        Raises:
            ValueError: If *method* is not supported.
        """
        if method not in self._INIT_METHODS:
            raise ValueError(
                f"Unknown init method '{method}'. "
                f"Supported: {list(self._INIT_METHODS.keys())}"
            )
        init_fn = self._INIT_METHODS[method]
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                init_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_string(self) -> str:
        """Return a human-readable model summary.

        Includes the class name, total trainable parameters, and the
        standard ``repr`` produced by ``nn.Module``.
        """
        n_params = self.count_parameters()
        header = (
            f"{self.__class__.__name__}  "
            f"({n_params:,} trainable parameters)"
        )
        return f"{header}\n{repr(self)}"


# ---------------------------------------------------------------------------
# Placeholder model (infrastructure testing)
# ---------------------------------------------------------------------------


@register("placeholder")
class Placeholder(BaseModel):
    """Minimal feed-forward model for infrastructure testing.

    Architecture: Linear -> ReLU -> Linear

    Args:
        input_size: Dimensionality of each input sample.
        output_size: Number of output classes / dimensions.
        hidden: Width of the hidden layer (default 128).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size),
        )
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, input_size)``.

        Returns:
            Output tensor of shape ``(batch, output_size)``.
        """
        return self.net(x)
