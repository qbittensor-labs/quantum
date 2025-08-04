import numpy as np

def as_int_uid(x: int | bytes | bytearray) -> int:
    """Return UID as int (0-255), adjusting endianness for bytes if necessary."""
    # handle built-in int or NumPy integer
    if isinstance(x, (int, np.integer)):  # accept numpy scalars too
        uid = int(x) # cast to plain int
    else:  # bytes or bytearray
        if len(x) == 0:
            raise ValueError("Empty bytes or bytearray cannot be interpreted as UID")
        # Try little-endian first
        uid = int.from_bytes(x, byteorder="little")
        if not (0 <= uid <= 255):
            # Try big-endian
            uid = int.from_bytes(x, byteorder="big")
            if not (0 <= uid <= 255):
                raise ValueError(f"Cannot interpret bytes {x.hex()} as UID (0-255) with either endianness.")
    # Final check for all cases
    if not (0 <= uid <= 255):
        raise ValueError(f"Input {x} results in UID {uid}, which is outside 0-255 range.")
    return uid