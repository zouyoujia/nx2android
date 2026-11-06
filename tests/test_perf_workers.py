"""Exercise actual-source ADPF worker selection and wall-time scan cadence."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'nxnative/dm/android/perf_android.c'


class PerformanceWorkers(unittest.TestCase):
    def test_sampling_and_worker_membership(self):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc:
            self.skipTest('a C compiler is required')
        source = SOURCE.read_text(encoding='utf-8')
        functions = source[source.index('typedef struct { int tid; uint64_t ticks; } hint_thread_sample;'):
                           source.index('// DM_PIN_GAME=1:')]
        targets = source[source.index('static uint64_t hint_cpu_targets'):source.index('static uint64_t hint_cpu_read')]
        harness = '#include <assert.h>\n#include <stdint.h>\n' + functions + targets + r'''
int main(void) {
    assert(hint_scan_due(0, 0, 1000000000, 0));
    assert(!hint_scan_due(0, 299, 9000000000, 1000000000));
    assert(hint_scan_due(0, 300, 9000000000, 1000000000));
    assert(hint_scan_due(1, 0, 1000000000, 0));
    assert(!hint_scan_due(1, 300, 1499999999, 1000000000));
    assert(hint_scan_due(1, 1, 1500000000, 1000000000));
    // A multi-second loading frame must not delay worker refresh until frame 300.
    assert(hint_scan_due(1, 7, 5000000000, 1500000000));
    hint_thread_sample prev[] = {{1,100}, {10,10}, {20,0}, {40,100}, {50,100},
                                  {60,1}, {70,1}, {80,1}, {90,1}};
    hint_thread_sample cur[] = {{1,200}, {10,30}, {20,30}, {30,999}, {40,90},
                                 {50,100}, {60,25}, {70,20}, {80,19}, {90,18}};
    int32_t pick[8] = {0}; pick[6] = 12345;
    int n = hint_pick_threads(1, cur, 10, prev, 9, 2.5, 1, pick);
    assert(n == 6 && pick[0] == 1 && pick[1] == 20 && pick[2] == 60);
    assert(pick[3] == 10 && pick[4] == 70 && pick[5] == 80 && pick[6] == 12345);
    // Unknown history, counter regressions, and idle workers are never selected.
    for (int i=0; i<n; i++) assert(pick[i] != 30 && pick[i] != 40 && pick[i] != 50);
    n = hint_pick_threads(1, cur, 10, prev, 9, 2.5, 0, pick);
    assert(n == 6 && pick[1] == 60 && pick[5] == 90); // legacy zero-tick filter
    hint_thread_sample zero[] = {{2,0}, {3,0}}, next[] = {{2,3}, {3,2}};
    n = hint_pick_threads(1, next, 2, zero, 2, 2.5, 1, pick);
    assert(n == 2 && pick[1] == 2); // 5% of 0.5 s at CLK_TCK=100
    assert(hint_pick_threads(1, next, 2, zero, 2, 7.5, 1, pick) == 1);
    assert(hint_pick_threads(1, next, 2, zero, 0, 2.5, 1, pick) == 1);
    n = hint_pick_threads(0, cur, 10, prev, 9, 2.5, 1, pick);
    assert(n == 5 && pick[0] == 1 && pick[1] == 20); // busiest overall, including presenter
    uint64_t hw[] = {2000,3000,3000,3300}, lim[] = {2000,3000,3000,3300}, cap[] = {400,800,800,1024};
    assert(hint_cpu_targets(hw, lim, cap, 4) == 8);
    uint64_t pair[2]; hint_cpu_pair(hw, lim, cap, 4, pair);
    assert(pair[0] == 8 && pair[1] == 6); // one prime: second thread uses big cores
    lim[3] = 1300; // prime capped: prefer both uncapped big cores
    assert(hint_cpu_targets(hw, lim, cap, 4) == 6);
    hint_cpu_pair(hw, lim, cap, 4, pair); assert(pair[0] == 6 && pair[1] == 6); // two equally fast cores
    cap[1] = 0; // partial capacity metadata: use frequency consistently for all cores
    assert(hint_cpu_targets(hw, lim, cap, 4) == 6);
    lim[1] = lim[2] = lim[3] = 0; assert(hint_cpu_targets(hw, lim, cap, 4) == 1);
    lim[0] = 0; assert(hint_cpu_targets(hw, lim, cap, 4) == 0);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as folder:
            src, exe = Path(folder) / 'workers.c', Path(folder) / 'workers'
            src.write_text(harness, encoding='utf-8')
            build = subprocess.run([cc, '-std=c11', '-Wall', '-Wextra', '-Werror', str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
