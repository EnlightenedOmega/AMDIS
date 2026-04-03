# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Microbiome Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MicrobiomeAction, MicrobiomeObservation


class MicrobiomeEnv(
    EnvClient[MicrobiomeAction, MicrobiomeObservation, State]
):
    """
    Client for the Microbiome Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MicrobiomeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.health_marker)
        ...
        ...     result = client.step(MicrobiomeAction(dosage=2.0))
        ...     print(result.observation.health_marker)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MicrobiomeEnv.from_docker_image("microbiome-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MicrobiomeAction(dosage=2.0))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MicrobiomeAction) -> Dict:
        """
        Convert MicrobiomeAction to JSON payload for step message.

        Args:
            action: MicrobiomeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "dosage": action.dosage,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MicrobiomeObservation]:
        """
        Parse server response into StepResult[MicrobiomeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MicrobiomeObservation
        """
        obs_data = payload.get("observation", {})
        observation = MicrobiomeObservation(
            microbiome_abundances=obs_data.get("microbiome_abundances", []),
            drug_concentration=obs_data.get("drug_concentration", 0.0),
            metabolite_concentration=obs_data.get("metabolite_concentration", 0.0),
            health_marker=obs_data.get("health_marker", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
