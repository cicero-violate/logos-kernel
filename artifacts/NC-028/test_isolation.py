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
    def test_fork_denied_and_declared_resource_limits(self):
        # os.fork routes through clone on modern glibc; a child must never
        # escape the tracked process after a timeout.
        program = """
import errno,os,resource,threading
try:
    child=os.fork()
except OSError as e:
    assert e.errno in (errno.EPERM,errno.ENOSYS), e
else:
    if child==0:os._exit(88)
    os.waitpid(child,0)
    raise AssertionError('FORK_ESCAPE')
assert resource.getrlimit(resource.RLIMIT_NPROC)==(2048,2048)
assert resource.getrlimit(resource.RLIMIT_FSIZE)==(4*1024**2,4*1024**2)
out=[]
t=threading.Thread(target=lambda:out.append('THREAD_OK'))
t.start()
t.join()
assert out==['THREAD_OK']
print('FORK_DENIED PROCESS_CAP FSIZE_MATCH THREAD_OK',flush=True)
"""
        cp=isolator.run(["-c",program], timeout=15)
        self.assertEqual(cp.returncode,0,cp.stderr)
        self.assertIn("FORK_DENIED PROCESS_CAP FSIZE_MATCH THREAD_OK",cp.stdout)

    def test_posix_spawn_denied_and_output_budget(self):
        # posix_spawn may use clone(CLONE_VM|CLONE_VFORK); still a child.
        cp=isolator.run(["-c", "import os; os.posix_spawn('/usr/bin/true',['true'],{})"],
                        timeout=15)
        self.assertNotEqual(cp.returncode,0)
        self.assertNotIn("SECURITY_ESCAPE",cp.stdout)
        # RLIMIT_FSIZE must cap regular-file-backed stdout; pipes bypass it.
        flood=isolator.run(["-c",
               "import os; chunk=b'x'*1048576; [os.write(1,chunk) for _ in range(7)]"],
               timeout=15)
        self.assertNotEqual(flood.returncode,0,"output beyond the cap escaped")
        self.assertLessEqual(len(flood.stdout),4*1024**2)

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
