from __future__ import annotations

import math

import pytest

from nekro_plugin_debounce.classifier import send_probability


def test_send_probability_is_stable() -> None:
    result = send_probability([[0.0, 1.0]])
    assert result == pytest.approx(1 / (1 + math.exp(-1)), rel=1e-6)


def test_send_probability_rejects_single_logit() -> None:
    with pytest.raises(ValueError):
        send_probability([[1.0]])
