# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Microbiome Environment.

The microbiome environment is a simple test environment that echoes back messages.
"""

from typing import List
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MicrobiomeAction(Action):
    """Action for the Microbiome environment - representing a drug dosage."""

    dosage: float = Field(..., ge=0.0, description="Drug dosage to administer (non-negative)")


class MicrobiomeObservation(Observation):
    """Observation from the Microbiome environment - representing the physiological state."""

    microbiome_abundances: List[float] = Field(default_factory=list, description="Abundance of each microbial species")
    drug_concentration: float = Field(default=0.0, description="Current drug concentration in the system")
    metabolite_concentration: float = Field(default=0.0, description="Current metabolite concentration from microbial metabolism")
    health_marker: float = Field(default=0.0, description="Health marker indicating disease severity (lower is better, 0 is ideal)")
