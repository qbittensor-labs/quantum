import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables from .env (for METRICS_API_URL, etc.)
from dotenv import load_dotenv
load_dotenv()

# Explicitly disable legacy OpenTelemetry exporters to avoid duplicate metrics export
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"
for _var in (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
):
    os.environ.pop(_var, None)

import logging
logging.getLogger("torch.autograd").setLevel(logging.ERROR)

import torch
torch.autograd.set_detect_anomaly(False)

import bittensor as bt
import signal
import sys
from pathlib import Path

from qbittensor.base.validator import BaseValidatorNeuron
from qbittensor.validator.services.metrics import MetricsService
import qbittensor

import torch
import time


CLEANUP_FLAG = Path("/tmp/validator_cleanup_done")

def _graceful_shutdown(signum, frame):
    bt.logging.info(f"[validator] Received signal {signum}, shutting down gracefully...")
    try:
        CLEANUP_FLAG.write_text("done")
    except Exception as e:
        bt.logging.warning(f"[validator] Could not write cleanup flag: {e}")
    try:
        global _VALIDATOR_SINGLETON
        if _VALIDATOR_SINGLETON is not None:
            if hasattr(_VALIDATOR_SINGLETON, 'is_running') and _VALIDATOR_SINGLETON.is_running:
                try:
                    _VALIDATOR_SINGLETON.stop_run_thread()
                except Exception:
                    pass
    except Exception:
        bt.logging.error("[validator] error during graceful shutdown", exc_info=True)
    finally:
        import os as _os
        _os._exit(0)


class Validator(BaseValidatorNeuron):
    """Simplified validator that only sets weights to UID 235 at 100%."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_weight_time = 0.0
        self.weight_setting_interval = 30 * 60  # 30 minutes

    def forward(self):
        """Simplified forward - just set weights to UID 235 at 100%"""
        now = time.time()
        
        # Only set weights every 30 minutes
        if now - self.last_weight_time < self.weight_setting_interval:
            time.sleep(10)
            return
            
        try:
            # Send heartbeat with version number
            try:
                self.metrics_service.record_heartbeat(qbittensor.__version__)
                bt.logging.info(f"📡 Heartbeat sent - version {qbittensor.__version__}")
            except Exception as e:
                bt.logging.warning(f"Failed to send heartbeat: {e}")
            
            target_uid = 235
            
            # Verify target UID exists in metagraph
            if target_uid >= len(self.metagraph.uids):
                bt.logging.error(f"Target UID {target_uid} does not exist in metagraph (max UID: {len(self.metagraph.uids)-1})")
                self.last_weight_time = now
                return
            
            weights = torch.zeros(len(self.metagraph.uids))
            weights[target_uid] = 1.0
            
            target_hotkey = self.metagraph.hotkeys[target_uid]
            
            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=self.metagraph.uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            
            if result and result[0]:
                bt.logging.info(f"✅ Weight extrinsic submitted")
                bt.logging.info(f"Sleeping for {self.weight_setting_interval/60:.0f} minutes until next weight setting")
            else:
                bt.logging.error(f"Weight extrinsic failed: {result}")
                
        except Exception as exc:
            if "NeuronNoValidatorPermit" in str(exc):
                bt.logging.warning("No validator permit")
            else:
                bt.logging.error(f"Weight-setting error: {exc}", exc_info=True)
        finally:
            self.last_weight_time = now


    def run(self):
        # sanity-check hotkey
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error("Hotkey not registered, aborting")
            return

        self.metrics_service = MetricsService(keypair=self.wallet.hotkey, network=self.subtensor.network)

        global _VALIDATOR_SINGLETON
        _VALIDATOR_SINGLETON = self
        try:
            while True:
                try:
                    self.metagraph.sync(subtensor=self.subtensor)
                    self.forward()
                except KeyboardInterrupt:
                    bt.logging.warning("Shutting down…")
                    break
                except Exception as e:
                    bt.logging.error(f"loop error: {e}", exc_info=True)
                    time.sleep(10)  # Wait before retrying
        finally:
            # Keep metrics service running for heartbeat/version reporting
            bt.logging.info("Metrics service still active for heartbeat/version reporting")
            
            if self.is_running:
                self.stop_run_thread()
            
            bt.logging.info("Validator shutdown complete")


_VALIDATOR_SINGLETON = None

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    
    try:
        if CLEANUP_FLAG.exists():
            CLEANUP_FLAG.unlink()
    except Exception:
        pass

    Validator().run()

