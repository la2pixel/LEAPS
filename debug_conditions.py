"""Debug what's actually in a conditions .mat file."""
import sys
import struct
import numpy as np
from pathlib import Path

# Usage: python3 debug_conditions.py /path/to/conditions.mat
filepath = sys.argv[1]

with open(filepath, "rb") as fh:
    raw = fh.read()

print(f"File: {filepath}")
print(f"Size: {len(raw)} bytes")
print(f"Header (first 128 bytes text): {raw[:128]}")
print()

# MATLAB v5 header is 128 bytes, then elements follow
ws = raw[128:]

# Scan all 4-byte type tags at 4-byte offsets
print("=== Type tag scan (every 4 bytes) ===")
type_names = {
    1: "miINT8", 2: "miUINT8", 3: "miINT16", 4: "miUINT16(chars)",
    5: "miINT32", 6: "miUINT16", 7: "miUINT32", 8: "miINT64", 9: "miDOUBLE",
    10: "miUINT64", 14: "miMATRIX", 15: "miCOMPRESSED",
}
for i in range(0, min(len(ws) - 8, 4000), 4):
    tag = int.from_bytes(ws[i:i+4], "little")
    if tag in type_names:
        nbytes = int.from_bytes(ws[i+4:i+8], "little")
        print(f"  offset {i:5d}: type={tag} ({type_names[tag]}), nbytes={nbytes}")
        if tag == 6 and nbytes > 0 and nbytes <= 200:
            raw_chars = ws[i+8:i+8+nbytes]
            try:
                s = raw_chars.decode("utf-16-le").rstrip("\x00").strip()
                print(f"           UTF16: '{s}'")
            except Exception:
                pass
        elif tag == 14 and nbytes > 0:
            # peek into the matrix
            pass

print()

# Look for known strings as raw ASCII
print("=== ASCII substring search ===")
for needle in [b"slow", b"Slow", b"SLOW", b"normal", b"Normal", b"NORMAL",
               b"fast", b"Fast", b"FAST",
               b"102", b"127", b"152", b"178",
               b"5.2", b"7.8", b"9.2", b"11", b"12.4", b"18"]:
    idx = raw.find(needle)
    if idx >= 0:
        ctx = raw[max(0,idx-10):idx+len(needle)+10]
        print(f"  Found {needle!r} at byte {idx}: ...{ctx!r}...")

print()

# Look for UTF-16 strings by scanning every 2 bytes
print("=== UTF-16LE scan (every 2 bytes) ===")
found_utf16 = []
for i in range(0, len(ws) - 20, 2):
    # check if looks like printable ASCII in UTF-16
    chars = []
    j = i
    while j + 1 < len(ws):
        lo = ws[j]
        hi = ws[j+1]
        if hi == 0 and 32 <= lo < 128:
            chars.append(chr(lo))
            j += 2
        else:
            break
    if len(chars) >= 3:
        s = "".join(chars).strip()
        if s and s not in found_utf16:
            found_utf16.append(s)
            print(f"  offset {i:5d}: '{s}'")

print()
print(f"Total UTF-16 strings found: {len(found_utf16)}")
print("Strings matching labels:", [s for s in found_utf16
    if s.lower() in {"slow","normal","fast","102","127","152","178","5.2","7.8","9.2","11","12.4","18"}])
