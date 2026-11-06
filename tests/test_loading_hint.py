"""Verify the opt-in startup hint budget and foreground gating from actual C source."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class StartupLoadingHint(unittest.TestCase):
    def test_bounded_window_and_background(self):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc:
            self.skipTest('a C compiler is required')
        source = (Path(__file__).resolve().parents[1] / 'nxnative/dm/android/front_android.c').read_text()
        functions = source[source.index('static int startup_loading_duration'):source.index('static void *startup_loading_thread')]
        harness = '#include <stdint.h>\n#include <stdlib.h>\n#include <assert.h>\n' + functions + r'''
int main(void) {
    assert(startup_loading_duration(NULL) == 0);
    assert(startup_loading_duration("") == 0);
    assert(startup_loading_duration("0") == 0);
    assert(startup_loading_duration("-1") == 0);
    assert(startup_loading_duration("90x") == 0);
    assert(startup_loading_duration("90") == 90);
    assert(startup_loading_duration("3600") == 120);
    assert(startup_loading_wanted(10, 90, 0));
    assert(!startup_loading_wanted(10, 90, 1));
    assert(startup_loading_wanted(20, 90, 0));
    assert(!startup_loading_wanted(90, 90, 0));
    assert(!startup_loading_wanted(100, 90, 0));
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as d:
            src, exe = Path(d) / 'hint.c', Path(d) / 'hint'
            src.write_text(harness)
            p = subprocess.run([cc, '-std=c11', '-Wall', '-Wextra', '-Werror', str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            p = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
