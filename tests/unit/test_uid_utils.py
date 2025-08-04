import pytest
from qbittensor.validator.utils.uid_utils import as_int_uid

# ---------------- VALID INT TESTS ----------------
def test_as_int_uid_with_valid_int():
    assert as_int_uid(0) == 0
    assert as_int_uid(255) == 255
    assert as_int_uid(128) == 128

# ---------------- INVALID INT TESTS ----------------
def test_as_int_uid_with_invalid_int():
    with pytest.raises(ValueError, match=r"outside 0-255 range"):
        as_int_uid(-1)
    with pytest.raises(ValueError, match=r"outside 0-255 range"):
        as_int_uid(256)

# ---------------- VALID BYTES AND BYTEARRAY TESTS ----------------
@pytest.mark.parametrize("data,expected", [
    (b'\x00', 0),           # Single byte, valid in both endiannesses
    (b'\xff', 255),         # Single byte, valid in both endiannesses
    (b'\x80', 128),         # Single byte, valid in both endiannesses
    (b'\x00\x01', 1),       # Little-endian valid (1)
    (b'\xff\x00', 255),     # Little-endian valid (255)
    (b'\x01\x00', 1),       # Little-endian invalid (256), big-endian valid (1)
    (bytearray(b'\x02'), 2),        # Bytearray single byte
    (bytearray([10]), 10),          # Bytearray single byte (list syntax)
    (bytearray(b'\x01\x00'), 1),    # Bytearray big-endian fallback
])
def test_as_int_uid_with_valid_bytes_and_bytearray(data, expected):
    assert as_int_uid(data) == expected

# ---------------- INVALID BYTES AND BYTEARRAY TESTS ----------------
@pytest.mark.parametrize("data", [
    b'\x01\x01',    # Both endiannesses invalid (257)
    b'\xff\xff',    # Both endiannesses invalid (65535)
    bytearray(b'\x01\x01'),  # Bytearray both endiannesses invalid
])
def test_as_int_uid_with_invalid_bytes_and_bytearray(data):
    with pytest.raises(ValueError, match=r"Cannot interpret bytes"):
        as_int_uid(data)

# ---------------- EMPTY INPUTS TEST ----------------
def test_as_int_uid_with_empty_bytes_and_bytearray():
    with pytest.raises(ValueError, match=r"Empty bytes or bytearray cannot be interpreted as UID"):
        as_int_uid(b'')
    with pytest.raises(ValueError, match=r"Empty bytes or bytearray cannot be interpreted as UID"):
        as_int_uid(bytearray())