# coding: utf-8
"""
Compression utilities for shors circuit data transmission
"""
from __future__ import annotations

import gzip
import base64
import bittensor as bt


# Compression prefix to identify compressed data
COMPRESSION_PREFIX = "GZIP_B64:"


def compress_circuit_data(qasm: str) -> str:
    """
    Compress QASM circuit data using gzip and encode as base64.
    
    Args:
        qasm: The QASM string to compress
        
    Returns:
        Compressed and base64-encoded string with prefix
    """
    if not qasm:
        return qasm
    
    try:
        # Compress using gzip
        compressed = gzip.compress(qasm.encode('utf-8'))
        # Encode as base64 for safe transmission
        b64_encoded = base64.b64encode(compressed).decode('ascii')
        # Add prefix to identify compressed data
        return f"{COMPRESSION_PREFIX}{b64_encoded}"
    except Exception as e:
        bt.logging.warning(f"Failed to compress circuit data: {e}, sending uncompressed")
        return qasm


def decompress_circuit_data(data: str) -> str:
    """
    Decompress circuit data if it was compressed, otherwise return as-is.
    Handles backward compatibility with uncompressed data.
    
    Args:
        data: The probably compressed circuit data
        
    Returns:
        Decompressed QASM string or original data if not compressed
    """
    if not data:
        return data
    
    # Check if data is compressed (has prefix)
    if not data.startswith(COMPRESSION_PREFIX):
        return data
    
    try:
        # Remove prefix
        b64_data = data[len(COMPRESSION_PREFIX):]
        # Decode from base64
        compressed = base64.b64decode(b64_data)
        # Decompress
        decompressed = gzip.decompress(compressed).decode('utf-8')
        return decompressed
    except Exception as e:
        bt.logging.error(f"Failed to decompress circuit data: {e}, returning original")
        return data

