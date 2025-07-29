def as_int_uid(x: int | bytes | bytearray) -> int:
    """Return UID as int accepting little endian bytes too for past sol's"""
    if isinstance(x, (bytes, bytearray)):
        return int.from_bytes(x, byteorder="little")
    return int(x)