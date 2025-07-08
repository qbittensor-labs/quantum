import bittensor

# Load your wallet
wallet = bittensor.wallet(name="local_subnet", hotkey="localhotkey")

# Sign a message
message = b"hello world"
signature = wallet.hotkey.sign(message)

# To verify (e.g., by someone else)
wallet.hotkey.ss58_address == bittensor.Keypair(ss58_address=wallet.hotkey.ss58_address).ss58_address
bittensor.Keypair(ss58_address=wallet.hotkey.ss58_address).verify(message, signature)
