"""Exercise the real port stages: stock-driver opt-in, APK payload, and startup env."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from porter import cli, pipeline
from porter.upstream import game, android_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "010043B013C5C000"


class GpuPerformancePackaging(unittest.TestCase):
    def test_default_is_unchanged(self):
        self.assertFalse(cli._parser().parse_args(["port", "x"]).gpu_perf)
        self.assertFalse(pipeline.Options(Path(".")).gpu_perf)

    def test_stock_and_custom_driver_payloads(self):
        for perf, custom in ((False, False), (True, False), (False, True), (True, True), (False, "renamed"), (True, "renamed")):
            with self.subTest(perf=perf, custom=custom), tempfile.TemporaryDirectory(prefix="nx2a_gpu_pack_") as td:
                t = Path(td)
                drv = t / ("test_driver.so" if custom == "renamed" else "libtest_driver.so") if custom else None
                if drv: drv.write_bytes(b"synthetic external driver")
                port = pipeline.Port.__new__(pipeline.Port)
                port.o = pipeline.Options(folder=t, out=t / "out", gpu_perf=perf, vulkan_driver=drv, jobs=1)
                port.aw = t / "android"; port.logs = t / "logs"; port.logs.mkdir()
                port.tools = {"ndk": t / "ndk"}; port.env = dict(os.environ)
                port.g = game.load(PROFILE); port.ap = android_profile.load(PROFILE)
                port.app_name = "Synthetic game"; port.pkg_ver = "v0"; port.display = "0"
                port.package = "org.nx2a.synthetic"; port.result = {}
                host_commands = []

                def build_host(cmd, *_args, **_kwargs):
                    host_commands.append([str(x) for x in cmd])
                    out = Path(cmd[cmd.index("--out") + 1]); out.mkdir(parents=True)
                    (out / "libdmhost.so").write_bytes(b"\x7fELF" + bytes(60))
                    if "--adrenotools" in cmd:
                        for name in pipeline.ADRENOTOOLS_LIBS:
                            (out / name).write_bytes(("synthetic " + name).encode())

                with mock.patch.object(pipeline.nsp, "run", side_effect=build_host):
                    port.host()
                self.assertEqual("--adrenotools" in host_commands[0], bool(perf or custom))
                port.icon = lambda: []; port.theme = lambda: []
                with mock.patch.object(pipeline.paths, "signing_key", return_value=t / "signing.json"):
                    port.apk()
                with zipfile.ZipFile(port.apk_path) as apk:
                    libs = {Path(n).name for n in apk.namelist() if n.startswith("lib/arm64-v8a/")}
                    expected = {"libdmhost.so"}
                    if perf or custom: expected.update(pipeline.ADRENOTOOLS_LIBS)
                    if custom: expected.add(pipeline.driver_lib_name(drv))
                    self.assertEqual(libs, expected)
                    if perf and not custom:
                        self.assertNotIn("libvulkan_freedreno.so", libs)

                port.patched = t / "patched"; port.patched.mkdir()
                (port.patched / "main.patched.bin").write_bytes(b"synthetic module")
                port.game = t / "game"; (port.game / "RomFS").mkdir(parents=True)
                (port.game / "RomFS" / "fixture.bin").write_bytes(b"synthetic data")
                port.dmsv = t / "shaders.dmsv"; port.dmsv.write_bytes(b"DMSV")
                port.aoc = None; port.sysdata = None; port.lang = "en-US"; port.fps_env = []; port.fps_files = {}
                port.button_font = lambda: ([], {})
                port.data()
                env = (port.data_path / "env.txt").read_text()
                self.assertEqual("DM_GPU_PERF=1" in env, perf)
                self.assertEqual("DM_VK_DRIVER=" in env, bool(custom))
                if custom: self.assertIn("DM_VK_DRIVER=" + pipeline.driver_lib_name(drv), env)
                self.assertNotIn("DM_GPU_TURBO=", env)


if __name__ == "__main__":
    unittest.main()
