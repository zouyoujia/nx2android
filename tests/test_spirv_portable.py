# Synthetic GLSL exercises function inlining without any game files.
import importlib.util
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('spirv_portable', ROOT / 'nxnative/dm/android/tools/spirv_portable.py')
portable = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(portable)
TOOL_DIRS = list((ROOT / 'tools').glob('**/shader-tools/*'))


def tool(name):
    return shutil.which(name) or next((str(p / (name + ('.exe' if os.name == 'nt' else ''))) for p in TOOL_DIRS if (p / (name + ('.exe' if os.name == 'nt' else ''))).is_file()), None)


@unittest.skipUnless(all(tool(n) for n in ('glslc', 'spirv-opt', 'spirv-val')), 'SPIR-V tools required')
class SubgroupFunctions(unittest.TestCase):
    def check_shader(self, helper, expression, expected):
        source = '#version 460\n#extension GL_KHR_shader_subgroup_shuffle : require\nlayout(local_size_x=32) in;\nlayout(set=0,binding=1,std430) buffer Data {float v[];} data;\n'
        source += helper + '\nvoid main(){uint lane=gl_GlobalInvocationID.x; data.v[lane]=' + expression + ';}\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); shader = root / 'shader.comp'; shader.write_text(source)
            original = root / 'shader.spv'
            subprocess.run([tool('glslc'), '--target-env=vulkan1.1', '-O0', str(shader), '-o', str(original)], check=True, capture_output=True)
            before = original.read_bytes()
            self.assertEqual(portable.subgroup_helpers(before), expected)
            after = portable.inline_subgroup_helpers(before, tool('spirv-opt'), root / 'portable.spv')
            self.assertFalse(portable.subgroup_helpers(after))
            if not expected: self.assertEqual(before, after)
            # The rewrite must preserve descriptor bindings and the workgroup's execution mode.
            def interface(blob):
                w = struct.unpack(f'<{len(blob)//4}I', blob); out = []; i = 5
                while i < len(w):
                    n, op = w[i] >> 16, w[i] & 65535
                    if op == 16: out.append((op, *w[i + 2:i + n]))
                    if op == 71 and w[i + 2] in (33, 34): out.append((op, *w[i + 2:i + n]))
                    i += n
                return sorted(out)
            self.assertEqual(interface(before), interface(after))
            original.write_bytes(after)
            subprocess.run([tool('spirv-val'), '--target-env', 'vulkan1.1', str(original)], check=True, capture_output=True)

    def test_shuffle_helper_is_inlined(self):
        self.check_shader('float helper(float x){return subgroupShuffle(x,gl_SubgroupInvocationID^1u);}', 'helper(data.v[lane])', True)

    def test_entry_point_shuffle_is_unchanged(self):
        self.check_shader('', 'subgroupShuffle(data.v[lane],gl_SubgroupInvocationID^1u)', False)

    def test_ordinary_helper_is_unchanged(self):
        self.check_shader('float helper(float x){return x*2.0;}', 'helper(data.v[lane])', False)
