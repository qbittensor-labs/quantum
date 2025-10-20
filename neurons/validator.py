import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
logging.getLogger("torch.autograd").setLevel(logging.ERROR)

import torch
torch.autograd.set_detect_anomaly(False)

import bittensor as bt
import signal
import sys
from pathlib import Path

import qbittensor as qbt
from qbittensor.base.validator import BaseValidatorNeuron
from qbittensor.validator.forward import forward, shutdown
from qbittensor.validator.services.metrics import MetricsService
from qbittensor.validator.utils.auto_updater import start_updater, stop_updater

# import OOM error type for targeted catch
from qbittensor.validator.utils.challenge_utils import ValidatorOOMError

# used for GPU reset
import subprocess
import shlex
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
            import os
            timeout_env = os.getenv("VALIDATOR_SHUTDOWN_TIMEOUT_S")
            timeout_s = float(timeout_env) if timeout_env else None
            shutdown(_VALIDATOR_SINGLETON, timeout_s=timeout_s)
    except Exception:
        bt.logging.error("[validator] error during graceful shutdown", exc_info=True)
    finally:
        # waiting causes hangs
        import os as _os
        _os._exit(1)


# for GPU reset sequence
def _run_cmd(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timeout: {e}"
    except Exception as e:
        return 125, "", f"error: {e}"

# GPU reset sequence; requires sudo NOPASSWD for nvidia-smi/rmmod/modprobe
def run_gpu_reset_sequence(gpu_id: int = 0, do_mig_toggle: bool = True) -> None:
    cmds: list[list[str]] = [
        ["sudo", "nvidia-smi", "--gpu-reset", "-i", str(gpu_id)],
        ["sudo", "rmmod", "nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia"],
        ["sudo", "modprobe", "nvidia"],
    ]
    if do_mig_toggle:
        cmds += [
            ["sudo", "nvidia-smi", "-i", str(gpu_id), "-mig", "0"],
            ["sudo", "nvidia-smi", "-i", str(gpu_id), "-mig", "1"],
            ["sudo", "nvidia-smi", "-i", str(gpu_id), "-mig", "0"],
        ]

    bt.logging.info("[gpu-reset] starting")
    for cmd in cmds:
        rc, out, err = _run_cmd(cmd, timeout=60)
        bt.logging.info(f"[gpu-reset] $ {' '.join(shlex.quote(c) for c in cmd)} -> rc={rc}")
        if out:
            bt.logging.debug(f"[gpu-reset] stdout:\n{out}")
        if err:
            bt.logging.warning(f"[gpu-reset] stderr:\n{err}")
    bt.logging.info("[gpu-reset] done")


class Validator(BaseValidatorNeuron):
    """Thin wrapper around BaseValidatorNeuron with a synchronous forward loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.step = 0 

    def forward(self):
        return forward(self)


    def run(self):
        # sanity-check hotkey
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error("Hotkey not registered, aborting")
            return

        bt.logging.info("Validator starting (sync mode)")

        self.metrics_service = MetricsService(keypair=self.wallet.hotkey, network=self.subtensor.network)

        # Validator git repo update worker
        #start_updater(check_interval_minutes=5)

        global _VALIDATOR_SINGLETON
        _VALIDATOR_SINGLETON = self
        try:
            while True:
                try:
                    self.metrics_service.record_heartbeat(qbt.__version__)
                    #bt.logging.info(f"step={self.step} uid={self.uid}")
                    self.metagraph.sync(subtensor=self.subtensor)
                    self.forward() # sync now
                    #self.step += 1
                except KeyboardInterrupt:
                    bt.logging.warning("Shutting down…")
                    break
                # catch explicit OOM signal, reset GPU, then restart cleanly
                except ValidatorOOMError as e:
                    bt.logging.error(f"OOM detected: {e}. Attempting GPU reset before restart")
                    try:
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        except Exception:
                            pass
                        try:
                            subprocess.run(["pkill", "-f", "gen_worker.py"], check=False)
                        except Exception:
                            pass
                        import os
                        gpu_id = int(os.getenv("VALIDATOR_GPU_ID", "0"))
                        script = os.getenv("GPU_RESET_SCRIPT", "gpu-reset-and-exec.sh")
                        cmd = [script, sys.executable, *sys.argv]
                        env = os.environ.copy(); env["GPU_ID"] = str(gpu_id)
                        os.execvpe(script, cmd, env)
                    finally:
                        _graceful_shutdown(signal.SIGTERM, None)
                    break
                except Exception as e:
                    bt.logging.error(f"loop error: {e}", exc_info=True)
        finally:
            self.metrics_service.shutdown()

            bt.logging.info("Stopping auto-updater...")
            #stop_updater()
            if self.is_running:
                self.stop_run_thread()
            try:
                shutdown(self, timeout_s=1800.0)
            except Exception:
                pass


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

