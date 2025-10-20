import os
import time
import torch
import bittensor as bt
import requests
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from .jwt_manager import JWTManager
import queue
import threading

class MetricsService:
    def __init__(self, keypair, service_name="bittensor.sn63.validator", export_interval_millis=5000, network: str = "", max_queue_size=1000, batch_size=10, retry_attempts=3, retry_delay=1):
        """
        Initialize the MetricsService.
        Metrics are disabled if the METRICS_API_URL environment variable is not set or keypair is missing.
        :param keypair: Bittensor Keypair required for JWT authentication.
        :param service_name: Name of the service for request headers.
        :param export_interval_millis: Flush interval in ms (for background sending).
        :param network: Deployment network (logged but not used in requests).
        :param max_queue_size: Max size of the internal queue before dropping items.
        :param batch_size: Number of items to batch per send (if API supports; otherwise 1).
        :param retry_attempts: Max retries per send.
        :param retry_delay: Initial retry delay in seconds (exponential backoff).
        """
        self.base_url = os.environ.get("METRICS_API_URL", "https://telemetry.openquantum.com")
        self.service_name = service_name
        self.validator_hotkey = keypair.ss58_address
        self.network = network
        self.keypair = keypair
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.flush_interval = export_interval_millis / 1000.0  # Convert to seconds

        bt.logging.info(f"Metrics sending to: {self.base_url}")
        self.jwt_manager = JWTManager(self.keypair)
        self.jwt = None
        self.session = requests.Session()
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._start_background_worker()

    def _to_python_scalar(self, x: Any) -> Any:
        """Convert NumPy or Torch scalars to JSON-serializable Python types."""
        if x is None:
            return None
        if isinstance(x, (int, float, str)):
            return x
        if hasattr(x, 'item'):  # Handles torch.Tensor scalars and NumPy arrays
            return x.item()
        if isinstance(x, (np.integer, np.floating, np.number)):
            return x.item()
        return str(x)  # Fallback for other types

    def _start_background_worker(self):
        """Start the background thread for flushing the queue."""
        def worker():
            while not self._stop_event.is_set():
                try:
                    # Flush every interval or when batch_size reached
                    start_time = time.time()
                    batch = []
                    while len(batch) < self.batch_size and not self._stop_event.is_set():
                        try:
                            item = self.queue.get(timeout=0.1)
                            batch.append(item)
                        except queue.Empty:
                            break
                    if batch:
                        self._flush_batch(batch)
                    sleep_time = max(0, self.flush_interval - (time.time() - start_time))
                    if sleep_time > 0:
                        self._stop_event.wait(sleep_time)
                except Exception as e:
                    bt.logging.error(f"Background worker error: {e}")
                    time.sleep(self.retry_delay)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _flush_batch(self, batch: list[Dict[str, Any]]) -> None:
        """Flush a batch of datapoints as a single API request (wrap in 'datapoints' array)."""
        # Construct batch payload once, send in one request (instead of one-by-one)
        datapoints = []
        for item in batch:
            payload_item = {
                "type": item['type'],
                "timestamp": item['timestamp'],
            }
            if item.get('miner_uid') is not None:
                payload_item["minerUid"] = item['miner_uid']
            if item.get('miner_hotkey'):
                payload_item["minerHotkey"] = item['miner_hotkey']
            if isinstance(item['value'], (int, float)):
                payload_item["numericValue"] = item['value']
            else:
                payload_item["stringValue"] = item['value']
            if item.get('attributes'):
                payload_item["attributes"] = item['attributes']
            datapoints.append(payload_item)

        # Send the full batch
        for attempt in range(self.retry_attempts):
            try:
                headers = {
                    "Authorization": f"Bearer {self._get_current_jwt()}",
                    "Content-Type": "application/json",
                    "X-Service-Name": self.service_name,
                    "X-Network": self.network,
                }
                url = f"{self.base_url}/v1/datapoints"
                response = self.session.post(url, json={"datapoints": datapoints}, headers=headers, timeout=5.0)
                response.raise_for_status()
                # Mark all as done after successful batch send
                for _ in batch:
                    self.queue.task_done()
                break
            except Exception as e:
                bt.logging.warning(f"Batch send attempt {attempt + 1} failed (size {len(batch)}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    bt.logging.error(f"Failed to send batch of {len(batch)} after {self.retry_attempts} attempts; dropping.")
                    # Mark as done even on failure to avoid stuck queue
                    for _ in batch:
                        self.queue.task_done()

    def _get_current_jwt(self) -> str:
        """Get or refresh JWT access token if expired."""
        if not self.jwt or self.jwt.expiration_date < datetime.now(timezone.utc):
            self.jwt = self.jwt_manager.get_jwt()
        return self.jwt.access_token

    def _enqueue_datapoint(self, type: str, timestamp: str, value: float | str, miner_uid: Optional[int] = None, miner_hotkey: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> bool:
        """Enqueue a datapoint; return True if enqueued, False if queue full (dropped)."""
        # CHANGE: timestamp now str (ISO)
        try:
            if self.queue.full():
                bt.logging.warning(f"Queue full (size {self.max_queue_size}); dropping datapoint {type}")
                return False
            # Convert value to ensure it's a Python scalar (handles NumPy/Torch)
            safe_value = self._to_python_scalar(value)
            if isinstance(safe_value, (int, float)):
                safe_value = float(safe_value)  # Ensure float for numericValue

            # onvert miner_uid
            safe_miner_uid = self._to_python_scalar(miner_uid) if miner_uid is not None else None
            if safe_miner_uid is not None:
                safe_miner_uid = int(safe_miner_uid)

            # Convert attributes values
            safe_attributes = None
            if attributes:
                safe_attributes = {
                    k: self._to_python_scalar(v)
                    for k, v in attributes.items()
                }

            item = {
                'type': type,
                'timestamp': timestamp,
                'value': safe_value,
                'miner_uid': safe_miner_uid,
                'miner_hotkey': miner_hotkey,
                'attributes': safe_attributes,
            }
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            bt.logging.warning(f"Queue full; dropping datapoint {type}")
            return False

    def record_circuit_sent(self, circuit_type: str, miner_uid: int, miner_hotkey: str):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            self._enqueue_datapoint(f"{circuit_type}_circuits_sent", timestamp, 1.0, miner_uid, miner_hotkey)
        except Exception as e:
            bt.logging.debug(f"Failed to enqueue circuit_sent {circuit_type}: {e}")  # Non-critical

    def record_solution_received(self, circuit_type: str, miner_uid: int, miner_hotkey: str, nqubits: Optional[int] = None):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            attributes = {'difficulty': nqubits} if nqubits is not None else {}
            self._enqueue_datapoint(f"{circuit_type}_solutions_received", timestamp, 1.0, miner_uid, miner_hotkey, attributes=attributes)
        except Exception as e:
            bt.logging.debug(f"Failed to enqueue solution_received {circuit_type}: {e}")

    def record_certificate_received(self, circuit_type: str, miner_uid: int, miner_hotkey: str, nqubits: Optional[int] = None):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            attributes = {'difficulty': nqubits} if nqubits is not None else {}
            self._enqueue_datapoint(f"{circuit_type}_certificates_received", timestamp, 1.0, miner_uid, miner_hotkey, attributes=attributes)
        except Exception as e:
            bt.logging.debug(f"Failed to enqueue certificate_received {circuit_type}: {e}")

    def set_circuit_difficulty(self, circuit_type: str, miner_uid: int, miner_hotkey: str, difficulty: float):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            self._enqueue_datapoint(f"{circuit_type}_difficulty", timestamp, difficulty, miner_uid, miner_hotkey)
        except Exception as e:
            bt.logging.debug(f"Failed to enqueue difficulty {circuit_type}: {e}")

    def record_weights_set_complete(self):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            self._enqueue_datapoint("weights_set", timestamp, float(datetime.now(timezone.utc).timestamp()))  # Value remains Unix-like numeric for compatibility, or change to 0 if needed
        except Exception as e:
            bt.logging.debug(f"Failed to enqueue weights_set: {e}")

    def set_miner_weights(self, weights: torch.Tensor, hotkeys: list[str]):
        try:
            if weights.shape[0] != len(hotkeys):
                raise ValueError("Weights tensor size must match the number of hotkeys")
            # CHANGE: Use ISO string instead of Unix int
            timestamp = datetime.now(timezone.utc).isoformat()
            for uid, (weight, hotkey) in enumerate(zip(weights, hotkeys)):
                self._enqueue_datapoint("miner_weight", timestamp, float(weight.item()), uid, hotkey)
        except Exception as e:
            bt.logging.debug(f"Failed to enqueue miner_weights: {e}")

    def record_heartbeat(self, version: str):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            # Record version as string
            self._enqueue_datapoint("heartbeat_version", timestamp, version)
        except Exception as e:
            bt.logging.debug(f"Failed to enqueue heartbeat: {e}")

    def shutdown(self):
        """
        Shuts down the requests session and flushes the queue.
        This should be called during application cleanup.
        """
        try:
            bt.logging.info("Shutting down metrics service...")
            self._stop_event.set()
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)  # Wait up to 5s for flush
            # Force flush remaining
            batch = []
            while not self.queue.empty():
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._flush_batch(batch)
            self.session.close()
            bt.logging.info("Metrics service shutdown complete. ✅")
        except Exception as e:
            bt.logging.warning(f"Error during shutdown: {e}")