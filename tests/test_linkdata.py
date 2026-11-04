# Synthetic archives: no game assets or keys.
import importlib.util
import os
import random
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'nxnative/dm/gfx/tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('linkdata_split', TOOLS / 'shader_split.py')
ss = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {'DM_GAME': '010015100B514000'}):
    spec.loader.exec_module(ss)


def packed(raw, size):
    chunks = []
    for i in range(0, len(raw), size):
        data = raw[i:i + size]; comp = zlib.compress(data)
        chunks.append(struct.pack('<I', len(comp)) + comp if len(comp) + 4 < len(data) else data)
    out = struct.pack('<III', size, len(chunks), len(raw)) + struct.pack(f'<{len(chunks)}I', *map(len, chunks))
    out += bytes(-len(out) % 128)
    for data in chunks:
        out += data; out += bytes(-len(out) % 128)
    return out


class Linkdata(unittest.TestCase):
    def test_variable_sizes_stored_chunks_and_truncation(self):
        raw = b'x' * 4096 + random.Random(7).randbytes(4096) + b'z' * 77
        for size in (4096, 65536, 131072, 262144, 2097152):
            with self.subTest(size=size):
                data = packed(raw, size)
                self.assertEqual(ss.kt_gz(data), raw)
                self.assertIsNone(ss.kt_gz(data[:130]))

    def test_indexed_members_and_invalid_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'resources.BIN'; idx = data.with_suffix('.IDX')
            raw = b'\x91\x68\x86\x19' + bytes(4096); comp = packed(raw, 131072)
            data.write_bytes(comp + raw)
            idx.write_bytes(struct.pack('<8Q', 0, len(raw), len(comp), 1, len(comp), len(raw), 0, 0))
            found = []
            with patch.object(ss, 'glslc_shaders', side_effect=lambda d, label, **kw: [(label, d)]):
                for path, rel, span in ss.linkdata_jobs(str(data), data.name):
                    ss.linkdata_shaders(path, rel, span, found)
            self.assertEqual([d for _, d in found], [raw, raw])
            idx.write_bytes(struct.pack('<4Q', 2**63, 1, 1, 1))
            self.assertIsNone(ss.linkdata_jobs(str(data), data.name))
            idx.unlink()
            self.assertIsNone(ss.linkdata_jobs(str(data), data.name))
