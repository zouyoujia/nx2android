# Exercise the actual audio parser with synthetic SDK command lists and PCM.
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'darwin' and shutil.which('cc'), 'Mac C compiler required')
class AudioLayouts(unittest.TestCase):
    def test_pcm_and_startup_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            subprocess.run([sys.executable, str(ROOT / 'nxnative/dm/games/game.py'), 'header',
                            '010015100B514000', str(out / 'game_profile.h')], check=True, capture_output=True)
            subprocess.run(['cc', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-I', str(out),
                            str(ROOT / 'tests/c/audren_layout.c'), '-o', str(out / 'audio')], check=True, capture_output=True)
            run = subprocess.run([str(out / 'audio')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
