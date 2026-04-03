# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Microbiome Environment Implementation.

A RL environment that simulates the interaction between drug dosage,
gut microbiome composition, drug metabolism, and patient health.
"""

from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MicrobiomeAction, MicrobiomeObservation
except ImportError:
    from models import MicrobiomeAction, MicrobiomeObservation


class MicrobiomeEnvironment(Environment):
    """
    Simulates patient health based on microbiome drug metabolism using gLV equations.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the microbiome environment parameters."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Biological Parameters
        self.dt: float = 0.1
        self.num_species: int = 5
        self.max_steps: int = 200
        
        # We define constants for the biological simulation
        random.seed(42) # For reproducible but randomized initialization
        self.r = [random.uniform(0.1, 0.5) for _ in range(self.num_species)]
        self.A = [[random.uniform(-0.1, 0.1) for _ in range(self.num_species)] for _ in range(self.num_species)]
        # Self-limiting terms (carrying capacity constraint)
        for i in range(self.num_species):
            self.A[i][i] = -abs(self.A[i][i]) - 0.1
            
        self.b = [random.uniform(0.01, 0.1) for _ in range(self.num_species)]
        self.c = [random.uniform(0.1, 0.5) for _ in range(self.num_species)]
        
        self.k: float = 0.5      # Drug clearance rate
        self.km: float = 0.2     # Metabolite decay rate
        self.rho: float = 0.05   # Disease progression rate
        self.kappa: float = 0.5  # Healing rate from metabolites
        self.health_target: float = 0.0
        self.lambda_penalty: float = 0.1 # Toxicity penalty

        # State Variables
        self.abundances = []
        self.drug_concentration = 0.0
        self.metabolite_concentration = 0.0
        self.health_marker = 0.0

    def reset(self) -> MicrobiomeObservation:
        """
        Reset the environment to a new patient episode.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Initialize a patient state (can be fairly sick)
        self.health_marker = random.uniform(50.0, 80.0)
        self.drug_concentration = 0.0
        self.metabolite_concentration = 0.0
        self.abundances = [random.uniform(0.5, 1.5) for _ in range(self.num_species)]

        return MicrobiomeObservation(
            microbiome_abundances=self.abundances,
            drug_concentration=self.drug_concentration,
            metabolite_concentration=self.metabolite_concentration,
            health_marker=self.health_marker,
            done=False,
            reward=0.0,
        )

    def step(self, action: MicrobiomeAction) -> MicrobiomeObservation:  # type: ignore[override]
        """
        Execute a step in the environment, advancing biological dynamics.
        """
        self._state.step_count += 1
        
        # 1. Administer drug dosage
        dosage = max(0.0, float(action.dosage))
        self.drug_concentration += dosage
        
        # 2. Microbiome update (Generalized Lotka-Volterra)
        next_abundances = [0.0] * self.num_species
        metabolism_rate = 0.0
        
        for i in range(self.num_species):
            interaction_sum = sum(self.A[i][j] * self.abundances[j] for j in range(self.num_species))
            growth = self.r[i] + interaction_sum
            drug_effect = self.b[i] * self.drug_concentration
            
            dx = self.abundances[i] * (growth - drug_effect) * self.dt
            new_x = self.abundances[i] + dx
            # Clip between 0 and a reasonable max to prevent explosions
            next_abundances[i] = max(0.0, min(100.0, new_x))
            
            # Sum up metabolite production
            metabolism_rate += self.c[i] * self.abundances[i] * self.drug_concentration
            
        self.abundances = next_abundances
        
        # 3. Metabolite accumulation and clearance
        dm = (metabolism_rate - self.km * self.metabolite_concentration) * self.dt
        self.metabolite_concentration = max(0.0, min(500.0, self.metabolite_concentration + dm))
        
        # 4. Drug clearance
        dd = (-self.k * self.drug_concentration) * self.dt
        self.drug_concentration = max(0.0, min(100.0, self.drug_concentration + dd))
        
        # 5. Health Outcome Dynamics
        base_change = self.health_marker * self.rho
        healing = self.kappa * self.metabolite_concentration
        
        dh = (base_change - healing) * self.dt
        self.health_marker = max(0.0, min(100.0, self.health_marker + dh))
        
        # 6. Compute Reward
        reward = -abs(self.health_marker - self.health_target) - (self.lambda_penalty * self.drug_concentration)
        
        # 7. Check terminal conditions
        done = self._state.step_count >= self.max_steps or self.health_marker >= 100.0
        
        return MicrobiomeObservation(
            microbiome_abundances=self.abundances,
            drug_concentration=self.drug_concentration,
            metabolite_concentration=self.metabolite_concentration,
            health_marker=self.health_marker,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.
        """
        return self._state
