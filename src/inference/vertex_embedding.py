"""Vertex AI online prediction adapter for hosted text embeddings."""
from __future__ import annotations

import os
from typing import List, Sequence

import google.auth
from google.auth.transport.requests import AuthorizedSession
import numpy as np

from src.config.catalog import EndpointSpec, ModelSpec
from src.inference import limits
from src.inference.base import Embedder
from src.inference.types import InferenceError, Usage


class VertexEmbedding(Embedder):
    batch_size = 32

    def __init__(self, key: str, spec: ModelSpec, endpoint: EndpointSpec):
        super().__init__(key, spec)
        self.endpoint = endpoint
        self.usage = Usage()
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.session = AuthorizedSession(credentials)

    def embed(self, texts: Sequence[str], kind: str = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefix = self.spec.query_instruction if kind == "query" else None
        inputs = [f"{prefix}{text}" if prefix else text for text in texts]
        dns = os.getenv(self.endpoint.base_url_env or "", "").strip().rstrip("/")
        endpoint_id = os.getenv(self.endpoint.endpoint_id_env or "", "").strip()
        project = os.getenv(self.endpoint.project_env or "GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.getenv(self.endpoint.location_env or "GOOGLE_CLOUD_LOCATION", self.endpoint.default_location).strip()
        if not dns or not endpoint_id or not project:
            raise InferenceError(f"{self.key}: Vertex embedding endpoint configuration is incomplete")
        base_url = dns if dns.startswith(("http://", "https://")) else f"https://{dns}"
        url = f"{base_url}/v1/projects/{project}/locations/{location}/endpoints/{endpoint_id}:predict"
        vectors: List[List[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            try:
                with limits.inflight():
                    response = self.session.post(url, json={"instances": [{"inputs": text} for text in inputs[start : start + self.batch_size]]}, timeout=self.endpoint.timeout_s)
                    response.raise_for_status()
                    payload = response.json()
                vectors.extend(payload["predictions"])
                self.usage.add(Usage(api_calls=1))
            except Exception as exc:
                raise InferenceError(f"{self.key} ({self.spec.name}) embedding request failed: {exc}") from exc
        return np.asarray(vectors, dtype=np.float32).reshape(len(inputs), -1)