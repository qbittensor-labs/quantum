import os
import time
import torch
import bittensor as bt  # Added import for logging
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, ResourceAttributes

class MetricsService:
    def __init__(self, service_name="bittensor.sn63.validator", export_interval_millis=5000, validator_hotkey: str = "", network: str = "", enabled: bool = True):
        """
        Initialize the MetricsService.

        Metrics are disabled if the OTEL_EXPORTER_OTLP_ENDPOINT environment variable is not set.

        :param service_name: Name of the service for OTel identification.
        :param export_interval_millis: Interval for exporting metrics in milliseconds.
        :param validator_hotkey: The SS58 address of the validator's hotkey to attach as a global attribute to all metrics.
        :param enabled: Programmatically enable or disable metrics.
        """
        # Determine if metrics should be enabled based on env vars and the enabled flag.
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.enabled = enabled and bool(otlp_endpoint)

        # If not enabled, log a warning and exit initialization.
        if not self.enabled:
            bt.logging.warning("Metrics are not enabled. Set OTEL_EXPORTER_OTLP_ENDPOINT to enable.")
            return

        bt.logging.info(f"Metrics sending to: {otlp_endpoint}")

        # Initialize OTel metrics provider with resource attributes
        self.validator_hotkey = validator_hotkey
        resource = Resource(attributes={ResourceAttributes.SERVICE_NAME: validator_hotkey, ResourceAttributes.DEPLOYMENT_ENVIRONMENT: network}) if validator_hotkey else Resource.create()
        # The exporter automatically uses env vars like OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS
        exporter = OTLPMetricExporter()
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=export_interval_millis)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)

        # Get a meter for your metrics
        self.meter = metrics.get_meter(service_name)

        # Create metrics instruments
        self.circuits_sent_counter = self.meter.create_counter(
            name="total_circuits_sent",
            description="Total circuit solutions sent to miners, by type and miner",
            unit="1"
        )
        self.solutions_received_counter = self.meter.create_counter(
            name="total_solutions_received",
            description="Total circuit solutions received from miners, by type and miner",
            unit="1"
        )
        self.difficulty_gauge = self.meter.create_gauge(
            name="current_difficulty",
            description="Current difficulty set for hstab or peaked circuits, by type and miner",
            unit="1"
        )
        self.last_weights_set_overall_gauge = self.meter.create_gauge(
            name="last_weights_set_timestamp_overall",
            description="Unix timestamp of the last time weights were set overall",
            unit="s"
        )
        self.validator_heartbeat_gauge = self.meter.create_gauge(
            name="validator_heartbeat_timestamp",
            description="Unix timestamp for the validators heartbeat",
            unit="s"
        )
        self.miner_weight_gauge = self.meter.create_gauge(
            name="current_miner_weight",
            description="Current weight set for each miner",
            unit="1"
        )
        self.miner_last_weight_timestamp_gauge = self.meter.create_gauge(
            name="last_weights_set_timestamp_per_miner",
            description="Unix timestamp when weights were last set",
            unit="s"
        )

    def record_circuit_sent(self, circuit_type: str, miner_uid: int, miner_hotkey: str):
        if not self.enabled:
            return
        attributes = {
            "circuit_type": circuit_type,
            "miner_uid": str(miner_uid),
            "miner_hotkey": miner_hotkey
        }
        self.circuits_sent_counter.add(1, attributes)

    def record_solution_received(self, circuit_type: str, miner_uid: int, miner_hotkey: str):
        if not self.enabled:
            return
        attributes = {
            "circuit_type": circuit_type,
            "miner_uid": str(miner_uid),
            "miner_hotkey": miner_hotkey
        }
        self.solutions_received_counter.add(1, attributes)

    def set_circuit_difficulty(self, circuit_type: str, miner_uid: int, miner_hotkey: str, difficulty: float):
        if not self.enabled:
            return
        attributes = {
            "circuit_type": circuit_type,
            "miner_uid": str(miner_uid),
            "miner_hotkey": miner_hotkey
        }
        self.difficulty_gauge.set(difficulty, attributes)

    def record_weights_set_complete(self):
        if not self.enabled:
            return
        self.last_weights_set_overall_gauge.set(time.time())

    def set_miner_weights(self, weights: torch.Tensor, hotkeys: list[str]):
        if not self.enabled:
            return
        if weights.shape[0] != len(hotkeys):
            raise ValueError("Weights tensor size must match the number of hotkeys")

        self.miner_last_weight_timestamp_gauge.set(time.time())

        # hotkeys list is aligned with metagraph indices, so index == uid
        for uid, hotkey in enumerate(hotkeys):
            value = float(weights[uid].item())
            # Some exporters prefer strings; ints are OK by spec. If needed, str(uid).
            attrs = {"miner_uid": uid, "miner_hotkey": hotkey}

            self.miner_weight_gauge.set(value, attributes=attrs)

    def record_heardbeat(self, version: str):
        if not self.enabled:
            return
        attributes = {
            "version": version
        }
        self.validator_heartbeat_gauge.set(time.time(), attributes)

    def shutdown(self):
        """
        Shuts down the OpenTelemetry MeterProvider, ensuring all buffered metrics are exported.
        
        This should be called during application cleanup.
        """
        if not self.enabled:
            return

        bt.logging.info("Shutting down metrics service and flushing final metrics...")
        provider = metrics.get_meter_provider()
        
        # The global provider could be a NoOpMeterProvider if initialization failed.
        # Checking for the shutdown method is the safest way to proceed.
        if hasattr(provider, 'shutdown'):
            provider.shutdown()
        bt.logging.info("Metrics service shutdown complete. ✅")