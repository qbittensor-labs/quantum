from qbittensor.validator.utils.reward_crypto import verify_reward_signature


def reward_is_valid(syn):
    try:
        return verify_reward_signature(
            signature_hex=syn.validator_signature,
            validator_hotkey_ss58=syn.validator_hotkey,
            challenge_id=syn.challenge_id,
            entanglement_entropy=syn.entanglement_entropy,
        )
    except Exception:
        return False
