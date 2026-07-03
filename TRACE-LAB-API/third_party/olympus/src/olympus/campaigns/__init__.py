#!/usr/bin/env python

# ======================================================================

from olympus.campaigns.observations import Observations
from olympus.campaigns.param_space import ParameterSpace

__all__ = ["Campaign", "Observations", "ParameterSpace"]


def __getattr__(name):
    if name == "Campaign":
        from olympus.campaigns.campaign import Campaign

        return Campaign
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
