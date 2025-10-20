import base64
import json
from typing import Dict
import bittensor as bt
from dotenv import load_dotenv
import os
import sys
from time import time
from pydantic import BaseModel, Field
import requests
from datetime import datetime, timedelta, timezone

from qbittensor.validator.utils.request import make_session

JWT_ENDPOINT: str = "token"

class OQJWT(BaseModel):
    access_token: str
    expires_in: int

class JWT(OQJWT):
    expiration_date: datetime

class JWTManager:

    def __init__(self, keypair: bt.Keypair) -> None:
        self._keypair: bt.Keypair = keypair
        self._timeout: float = 7.0
        self._session: requests.Session = make_session(allowed_methods=["GET"])
        self._tensorauth_url: str = os.environ.get("TENSORAUTH_URL", "https://tensorauth.openquantum.com")

    def get_jwt(self) -> JWT:
        """Fetch JWT from tensorauth service using signed header"""
        bt.logging.trace(f" ☎️  Contacting tensorauth service for a JWT")
        now: datetime = datetime.now(timezone.utc)
        response = self._session.get(f"{self._tensorauth_url}/{JWT_ENDPOINT}", headers=self._get_signed_header(), timeout=self._timeout)
        response.raise_for_status()
        token_data = response.json()
        if not isinstance(token_data, dict):
            bt.logging.error(f"❌ ERROR: JWT response is not a dictionary: {token_data}")
            raise ValueError("JWT response is not a dictionary")
        try:
            token: OQJWT = OQJWT(**{str(k): v for k, v in token_data.items()})
        except Exception as e:
            bt.logging.error(f"❌ ERROR: Failed to parse JWT response: {e}")
            raise e
        bt.logging.trace(f"✅ Received JWT from {self._tensorauth_url}/{JWT_ENDPOINT}")
        expiration_date: datetime = now + timedelta(seconds=token.expires_in)
        bt.logging.trace(f"    - Token expires at {expiration_date.isoformat()} (in {token.expires_in} seconds)")
        return JWT(**token.model_dump(by_alias=True), expiration_date=expiration_date)

    def _get_signed_header(self) -> Dict[str, str]:
        """Create request header with signature, timestamp, hotkey"""
        timestamp = str(int(time()))
        signature_bytes: bytes = self._keypair.sign(self._keypair.ss58_address.encode("utf-8"))
        signature_b64: str = base64.b64encode(signature_bytes).decode("utf-8")
        token_json: dict = {
            "hotkey": self._keypair.ss58_address,
            "timestamp": timestamp,
            "signature": signature_b64
        }
        token: str = base64.b64encode(json.dumps(token_json).encode("utf-8")).decode('utf-8')
        return {
            "Authorization": f"Bearer {token}",
        }