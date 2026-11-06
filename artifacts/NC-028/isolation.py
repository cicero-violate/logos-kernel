"""Fail-closed Linux Landlock + seccomp runner; no namespaces or GPU.

Dependencies: Linux Landlock ABI >=5, libseccomp.so.2, Python 3 stdlib.
Read-only allowlist is trusted code/weight paths; only private scratch is writable.
This is OS-enforced filesystem+socket isolation, not a VM or UID boundary.
"""
import ctypes
import errno
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RO = [
    "/usr", "/lib", "/lib64", "/etc/ld.so.cache",
    "/dev/urandom", "/dev/null", "/proc/cpuinfo",
    "/opt/intel/oneapi/mkl/latest/lib/intel64",
    ROOT / "artifacts/NC-000",
    ROOT / "artifacts/NC-001",
    ROOT / "artifacts/NC-004",
    ROOT / "artifacts/NC-005",
    ROOT / "artifacts/NC-006",
    ROOT / "artifacts/NC-007",
    ROOT / "artifacts/NC-028",
    Path("/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt"),
    Path("/workspace/experiments/latticeforge_neural_isa_test"),
    ROOT / "artifacts/NC-028/scratch",
]
# These syscalls are unavailable to a contained program, including any imported code.
DENY = (
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg",
    "io_uring_setup", "io_uring_enter", "io_uring_register",
    "ptrace", "process_vm_readv", "process_vm_writev",
    "kill", "tkill", "tgkill", "pidfd_open", "pidfd_getfd",
    "bpf", "perf_event_open", "mount", "umount2", "unshare", "setns",
    "fork", "vfork",
    # Landlock does not control all inode metadata operations. Block those
    # syscalls explicitly rather than treating denied write() as immutable.
    "chmod", "fchmod", "fchmodat", "fchmodat2",
    "chown", "fchown", "lchown", "fchownat",
    "utime", "utimes", "futimesat", "utimensat",
    "setxattr", "lsetxattr", "fsetxattr",
    "removexattr", "lremovexattr", "fremovexattr",
    "name_to_handle_at", "open_by_handle_at",
)
ALLOW = 0x7fff0000
DENY_ERRNO = 0x00050000 | errno.EPERM
DENY_UNSUPPORTED = 0x00050000 | errno.ENOSYS
SCMP_CMP_MASKED_EQ = 7
CLONE_THREAD = 0x00010000
MAX_USER_TASKS = 2048

class SeccompArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_uint),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]
O_PATH = getattr(os, "O_PATH", 0o10000000)
READ_EXEC = (1 << 0) | (1 << 2) | (1 << 3)
SCRATCH_ACCESS = READ_EXEC | (1<<1) | (1<<4) | (1<<5) | (1<<7) | (1<<8) | (1<<13) | (1<<14)
FS_HANDLED = (1 << 16) - 1  # ABI5: includes ioctl-dev; ABI6 on this host.

class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]
class PathBeneath(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

def _check(result, label):
    if result < 0:
        raise OSError(ctypes.get_errno(), label)
    return result

def landlock_abi():
    lib = ctypes.CDLL(None, use_errno=True)
    return _check(lib.syscall(444, ctypes.c_void_p(0), 0, 1), "landlock ABI")

def _lock_fs():
    libc = ctypes.CDLL(None, use_errno=True)
    if landlock_abi() < 5:
        raise RuntimeError("Landlock ABI >=5 required")
    # no_new_privs prevents weakening the filesystem/security policy with exec.
    _check(libc.prctl(38, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
    rule = Ruleset(FS_HANDLED)
    fd = _check(libc.syscall(444, ctypes.byref(rule), ctypes.sizeof(rule), 0), "create_ruleset")
    try:
        for path in RO:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(path)
            item = os.open(str(path), O_PATH | os.O_CLOEXEC)
            try:
                allowed = (SCRATCH_ACCESS if path == ROOT / "artifacts/NC-028/scratch"
                           else READ_EXEC if path.is_dir()
                           else ((1 << 1) | (1 << 2)) if str(path)=="/dev/null"
                           else (1 << 2))
                entry = PathBeneath(allowed, item)
                _check(libc.syscall(445, fd, 1, ctypes.byref(entry), 0), f"landlock add {path}")
            finally:
                os.close(item)
        _check(libc.syscall(446, fd, 0), "landlock restrict_self")
    finally:
        os.close(fd)

def _block_network_and_escape():
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                           ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(SeccompArgCmp)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(ALLOW)
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in DENY:
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr < 0:  # absent on this architecture
                continue
            rc = lib.seccomp_rule_add(ctx, DENY_ERRNO, nr, 0)
            if rc:
                raise OSError(-rc, f"seccomp rule {name}")
        # fork() may be implemented through clone(SIGCHLD), not SYS_fork.
        # Allow only thread-group clones; kernel rejects CLONE_THREAD without
        # CLONE_VM/CLONE_SIGHAND. Never permit a child process to escape our
        # wall-timeout by outliving the tracked subprocess.
        clone = lib.seccomp_syscall_resolve_name(b"clone")
        if clone < 0:
            raise RuntimeError("clone syscall unavailable: cannot prove fork containment")
        cmp = SeccompArgCmp(0, SCMP_CMP_MASKED_EQ, CLONE_THREAD, 0)
        rc = lib.seccomp_rule_add_array(ctx, DENY_ERRNO, clone, 1, ctypes.byref(cmp))
        if rc:
            raise OSError(-rc, "seccomp clone process denial")
        # glibc may probe clone3 before falling back to clone for pthreads.
        clone3 = lib.seccomp_syscall_resolve_name(b"clone3")
        if clone3 >= 0:
            rc = lib.seccomp_rule_add(ctx, DENY_UNSUPPORTED, clone3, 0)
            if rc:
                raise OSError(-rc, "seccomp clone3 denial")
        rc = lib.seccomp_load(ctx)
        if rc:
            raise OSError(-rc, "seccomp_load")
    finally:
        lib.seccomp_release(ctx)

def _child_policy():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (24, 25))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4*1024**2, 4*1024**2))
    # Per-real-UID task cap, not a cgroup limit: insufficient against
    # globally shared UID pressure or privileged processes on its own.
    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_USER_TASKS, MAX_USER_TASKS))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    _lock_fs()
    _block_network_and_escape()

def run(argv, payload=None, timeout=35):
    env = {
        "PATH": "/usr/bin", "LANG": "C", "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(ROOT / "artifacts/NC-028/scratch"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
    }
    if not (0 < timeout <= 35):
        raise ValueError("sandbox timeout must be within 35 seconds")
    scratch = ROOT / "artifacts/NC-028/scratch"
    # A pipe captured by subprocess.run has no RLIMIT_FSIZE protection and
    # can exhaust host memory. Regular scratch-backed file descriptors are
    # constrained by the child's 4 MiB RLIMIT_FSIZE; bound parent reads too.
    with tempfile.TemporaryFile(mode="w+b", dir=scratch) as stdout_file, \
         tempfile.TemporaryFile(mode="w+b", dir=scratch) as stderr_file:
        cp = subprocess.run(
            [sys.executable, "-I", "-B", *argv],
            input=payload, text=True, stdout=stdout_file, stderr=stderr_file,
            cwd=str(ROOT), env=env, close_fds=True,
            preexec_fn=_child_policy, timeout=timeout,
        )
        stdout_file.seek(0)
        stderr_file.seek(0)
        limit = 4 * 1024**2
        out = stdout_file.read(limit + 1)
        err = stderr_file.read(limit + 1)
        if len(out) > limit or len(err) > limit:
            raise RuntimeError("sandbox output exceeded file-size budget")
        return subprocess.CompletedProcess(cp.args, cp.returncode,
                       out.decode("utf-8", "replace"),
                       err.decode("utf-8", "replace"))
