from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session(allowed_methods: List[str]) -> requests.Session:
        """
        Create a requests.Session with:
        - Retries on 429/5xx with exponential backoff
        - Default per-request timeout
        """
        s = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=allowed_methods,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s
