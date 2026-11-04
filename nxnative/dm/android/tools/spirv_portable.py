# Normalize subgroup operations into entry points before a mobile driver compiles them.
# Adreno 750's compiler dereferences NULL for a subgroup shuffle in a called function:
# reproduced with a 1.4 KB GLSL shader, independently of the game. Inlining alone fixes
# both that reproducer and the original compute shader; scalar/SSA/dead-code passes do not.
import struct
import subprocess
from pathlib import Path


def subgroup_helpers(spv):
    words = struct.unpack(f'<{len(spv) // 4}I', spv)
    entries, body, function = set(), [], None
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xffff
        if not count or i + count > len(words): raise ValueError('malformed SPIR-V instruction')
        if op == 15: entries.add(words[i + 2])   # OpEntryPoint
        elif op == 54: function = words[i + 2]   # OpFunction
        elif op == 56: function = None          # OpFunctionEnd
        elif function is not None and (333 <= op <= 366 or op == 4431): body.append(function)
        i += count
    return bool(set(body) - entries)


def inline_subgroup_helpers(spv, optimizer, path):
    if not subgroup_helpers(spv): return spv
    path = Path(path); path.write_bytes(spv)
    out = path.with_suffix('.inlined.spv')
    result = subprocess.run([str(optimizer), '--inline-entry-points-exhaustive', '--eliminate-dead-functions', str(path), '-o', str(out)], capture_output=True, text=True)
    if result.returncode: raise RuntimeError(f'subgroup function inlining failed: {result.stderr.strip()}')
    result = out.read_bytes()
    if subgroup_helpers(result): raise RuntimeError('subgroup operations remain in a helper after inlining')
    return result
