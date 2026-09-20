"""
orcha.integrations.phoenix
==========================
Boundary around Phoenix / OpenTelemetry developer observability.

Two distinct surfaces, kept deliberately separate:

- ``PhoenixBoundary.status()`` / ``load_otel()`` — the Apache-2.0
  OpenInference / OTel exporter packages (e.g.
  ``arize-phoenix-otel``, ``openinference-instrumentation``) used to
  ship traces from Orcha runs. These are the only packages that may be
  bundled, under the optional ``orcha[obs]`` extra.

- The Phoenix server application itself (ELv2) is an EXTERNAL developer
  tool. It is never bundled into the Anvira runtime; developers run it
  locally or self-hosted and point the exporter at it. Nothing in this
  boundary ever sends data to the cloud.

Local-first rule: no telemetry is emitted unless a developer explicitly
configures an endpoint (default ``http://localhost:6006``).
"""
from __future__ import annotations

from ..base import Boundary, IntegrationUnavailable

__all__ = ["PhoenixBoundary"]


class PhoenixBoundary(Boundary):
    name = "phoenix"
    package = "arize-phoenix-otel"
    extra = "obs"

    def load_otel(self) -> object:
        """
        Return the OpenInference OTLP span-exporter machinery.

        Only Apache-2.0 exporter packages are touched here. The ELv2
        Phoenix server is not imported, not bundled, and never required.
        """
        try:
            return self._import("phoenix.otel")
        except IntegrationUnavailable:
            raise IntegrationUnavailable(
                "Phoenix OTel exporter is not installed. Install with "
                "`pip install \"orcha[obs]\"` and point it at your local "
                "Phoenix server (e.g. http://localhost:6006). Nothing is "
                "sent anywhere unless you configure an endpoint."
            ) from None