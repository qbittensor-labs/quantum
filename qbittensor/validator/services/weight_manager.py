"""
Weight and scoring management for the validator.
"""
import time
from typing import TYPE_CHECKING
import torch
import bittensor as bt
import asyncio

if TYPE_CHECKING:
    from qbittensor.validator import Validator

# Constants
SCORING_INTERVAL = 900  # Score every 3 seconds
WEIGHT_SETTING_INTERVAL = 1800  # Set weights every 30 minutes
MIN_WEIGHT = 0.0001  # Minimum weight for active miners


class WeightManager:
    """Manages scoring updates and weight setting for the validator."""

    def __init__(self, validator: "Validator"):
        self.validator = validator
        self.last_scoring_time = 0
        self.last_weight_time = 0

    async def update(self) -> None:
        """Check and update scoring/weights if needed."""
        current_time = time.time()

        # Update scoring periodically
        if current_time - self.last_scoring_time > SCORING_INTERVAL:
            await self._update_scoring(current_time)

        # Set weights periodically
        if current_time - self.last_weight_time > WEIGHT_SETTING_INTERVAL:
            await self._set_weights(current_time)

    async def _update_scoring(self, current_time: float) -> None:
        """Update daily scoring history."""
        try:
            self.validator._scoring_mgr.update_daily_score_history()
            self.last_scoring_time = current_time
            bt.logging.info("✅ Updated daily scoring history")
        except Exception as e:
            bt.logging.error(f"❌ Scoring update failed: {e}")

    async def _set_weights(self, current_time: float) -> None:
        """Calculate and set weights based on decayed scores."""

        try:
            try:
                uid = self.validator.metagraph.hotkeys.index(
                    self.validator.wallet.hotkey.ss58_address
                )
            except ValueError:
                bt.logging.warning("⚠️  Hotkey not found in metagraph")
                return  # handled by finally

            scores = self.validator._scoring_mgr.calculate_decayed_scores(
                lookback_days=2
            )
            weights = torch.zeros(len(self.validator.metagraph.uids))
            for uid_, score in scores.items():
                if 0 <= uid_ < len(weights):
                    weights[uid_] = max(MIN_WEIGHT, score)
            total = weights.sum()
            if total > 0:
                weights /= total

            bt.logging.info(f"setting weights: {weights}")

            try:
                result = self.validator.subtensor.set_weights(
                    wallet=self.validator.wallet,
                    netuid=self.validator.config.netuid,
                    uids=self.validator.metagraph.uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False
                )

                if result[0]:
                    bt.logging.info(
                        f"✅ Set weights: "
                        f"{(weights > 0).sum().item()} miners weighted"
                    )
                    for uid_, score in sorted(
                        scores.items(), key=lambda x: x[1], reverse=True
                    )[:5]:
                        bt.logging.info(f"  UID {uid_}: {score:.4f}")
                else:
                    bt.logging.error(
                        f"❌ Failed to submit weight extrinsic. Result: {result}"
                    )

            except Exception as e:
                if "NeuronNoValidatorPermit" in str(e):
                    bt.logging.warning("⚠️  No validator permit – postponing 5 h")
                    # push the *next* attempt 5 h into the future
                    self.last_weight_time = current_time + WEIGHT_SETTING_INTERVAL * 10
                    return  # handled by finally
                bt.logging.error(f"❌ Weight submission error: {e}")

        except Exception as e:
            bt.logging.error(f"❌ Unexpected weight-setting error: {e}")

        finally:
            if self.last_weight_time < current_time:
                self.last_weight_time = current_time

    async def cleanup_scoring_data(self, retention_days: int = 7) -> None:
        """Clean up old scoring data."""
        try:
            self.validator._scoring_mgr.cleanup(retention_days=retention_days)
            bt.logging.info("✅ Cleaned up old scoring data")
        except Exception as e:
            bt.logging.error(f"❌ Scoring cleanup failed: {e}")
