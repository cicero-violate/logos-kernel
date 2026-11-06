"""Independent-process isolation regression probes. Only safe non-mutating probes."""
import errno
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("isolator", HERE / "isolation.py")
isolator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolator)

class IsolationTests(unittest.TestCase):
    def test_read_write_network_denial_and_crash(self):
        assert isolator.landlock_abi() >= 5
        program = """
import os,socket,sys
assert open('/usr/share/zoneinfo/UTC','rb').read(1)
for code in (lambda:socket.socket(),
             lambda:os.listdir('/home'),
             lambda:os.open('/workspace/experiments/logos-kernel/AGENTS.md',os.O_WRONLY)):
    try: code()
    except OSError as e: assert e.errno in (1,13),e
    else: raise AssertionError('SECURITY_ESCAPE')
print('NET_DENIED READ_DENIED WRITE_DENIED',flush=True)
"""
        r = isolator.run(["-c", program], timeout=15)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NET_DENIED READ_DENIED WRITE_DENIED", r.stdout)
        fixture=HERE/"scratch"/"test_metadata_probe"
        fixture.write_text("immutable within sandbox")
        try:
            change=isolator.run(["-c",
              "import os; os.chmod('/workspace/experiments/logos-kernel/artifacts/NC-028/scratch/test_metadata_probe',0o600)"
              ],timeout=15)
            self.assertNotEqual(change.returncode,0)
            self.assertIn("PermissionError",change.stderr)
        finally:
            fixture.unlink(missing_ok=True)
        crashed = isolator.run(["-c", "raise RuntimeError('deliberate crash')"], timeout=15)
        self.assertNotEqual(crashed.returncode, 0)
        self.assertIn("deliberate crash", crashed.stderr)

if __name__ == "__main__":
    unittest.main()
