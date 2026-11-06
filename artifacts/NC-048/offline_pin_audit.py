"""Read-only exact-Git-object audit of locally supplied research mirrors.

Dependencies: Python 3 standard library and installed Git CLI.
No downloads, package installs, checkout, candidate execution, or Git writes.
The local remote URL is *metadata*, not cryptographic repository authentication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

EXPECTED_IDS = frozenset(("alphageometry", "tscl", "alp-gmm",
                          "dreamcoder", "stitch", "z3"))
SHA40 = re.compile(r"[a-f0-9]{40}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
MAX_FILE_BYTES = 1024 * 1024  # no unbounded local blob reads
# For reproducibility, only project files; no globs, traversal, absolute paths.
SAFE_PART = re.compile(r"[a-zA-Z0-9_.+-]+\Z")


def safe_file(p):
    if not isinstance(p, str) or not p or "\\" in p:
        raise ValueError(f"unsafe file path: {p!r}")
    if any(not SAFE_PART.fullmatch(part) or part in (".", "..")
           for part in p.split("/")):
        raise ValueError(f"unsafe file path: {p!r}")
    if p.startswith(".git/") or p == ".git":
        raise ValueError("Git internals not part of source audit")
    return p


def sha(data):
    return hashlib.sha256(data).hexdigest()


def no_duplicates(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate JSON key {k}")
        d[k] = v
    return d


def git(mirror, *args):
    env = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1",
               GIT_CONFIG_NOSYSTEM="1",
               GIT_CONFIG_GLOBAL=os.devnull,
               GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    p = subprocess.run(["git", "-C", str(mirror), *args], env=env,
                       capture_output=True, check=False, timeout=15)
    if p.returncode:
        raise ValueError(f"local Git object unavailable ({' '.join(args[:2])})")
    return p.stdout


def bounded_blob(mirror, revision, path):
    size_raw = git(mirror, "cat-file", "-s", f"{revision}:{path}")
    size = int(size_raw.decode().strip())
    if size > MAX_FILE_BYTES:
        raise ValueError(f"oversized blob: {path} ({size} bytes)")
    return git(mirror, "show", f"{revision}:{path}")


def verify_reference(row, cache):
    name = row["id"]
    revision = row.get("immutable_git_commit_sha40")
    license_sha = row.get("license_file_sha256")
    license_path = row.get("license_path")
    result = {"id": name, "status": "HOLD", "source_executed": False,
              "network_access": False, "reviewer_identity_authenticated": False}
    if not isinstance(revision, str) or not SHA40.fullmatch(revision):
        result["reason"] = "missing exact 40-hex commit pin"
        return result
    if not isinstance(license_sha, str) or not SHA256.fullmatch(license_sha):
        result["reason"] = "missing exact 64-hex LICENSE SHA256"
        return result
    try:
        safe_file(license_path)
        submodules_expected = row.get("submodule_commit_shas")
        if not isinstance(submodules_expected, dict):
            raise ValueError("submodule manifest required, including explicit {} when empty")
        for subpath, subcommit in submodules_expected.items():
            safe_file(subpath)
            if not isinstance(subcommit, str) or not SHA40.fullmatch(subcommit):
                raise ValueError("invalid submodule commit")
        source_paths = row.get("source_paths_to_verify")
        if not isinstance(source_paths, list) or not source_paths:
            raise ValueError("nonempty source paths required")
        for path in source_paths:
            safe_file(path)
        if name not in EXPECTED_IDS:
            raise ValueError("unknown source ID")
        if not isinstance(row.get("repository_url"), str) or not row["repository_url"].startswith("https://github.com/"):
            raise ValueError("repository identity URL missing")
        # No writes or fetches. Resolved paths must stay under caller-given local cache.
        base = Path(cache).resolve(strict=True)
        mirror = (base / name).resolve(strict=True)
        if not mirror.is_relative_to(base) or mirror == base or not mirror.is_dir():
            raise ValueError("mirror escapes cache or is not directory")
        typ = git(mirror, "cat-file", "-t", revision).decode().strip()
        if typ != "commit":
            raise ValueError("pin is not a commit")
        exact = git(mirror, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
        if exact != revision:
            raise ValueError("pin resolves to a different commit")
        remote = git(mirror, "config", "--get", "remote.origin.url").decode().strip()
        if remote.rstrip("/").removesuffix(".git") != row["repository_url"].rstrip("/").removesuffix(".git"):
            raise ValueError("recorded remote URL differs from expected identity")
        tree = git(mirror, "ls-tree", "-r", revision).decode()
        submodules_found = {}
        for line in tree.splitlines():
            meta, sep, path = line.partition("\t")
            if not sep:
                raise ValueError("malformed Git tree row")
            mode, typ, obj = meta.split(" ")
            if mode == "160000":
                submodules_found[path] = obj
        if submodules_found != submodules_expected:
            raise ValueError("submodule Git-link pins mismatch")
        license_bytes = bounded_blob(mirror, revision, license_path)
        if sha(license_bytes) != license_sha:
            raise ValueError("pinned LICENSE byte hash mismatch")
        checked_sources = {}
        for path in source_paths:
            if path.endswith("/"):
                raise ValueError("source path must be an exact file, not directory")
            raw = bounded_blob(mirror, revision, path)
            checked_sources[path] = sha(raw)
        result.update(status="LOCALLY_PINNED", reason="exact commit, remote metadata, license and Git-link blobs match",
                      commit=revision, license_sha256=license_sha, source_sha256=checked_sources,
                      submodule_commit_shas=submodules_found,
                      note="No external origin authentication or license legal clearance; independent review required")
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
        result["reason"] = str(exc)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="artifacts/NC-048/teacher_reference_manifest.json")
    p.add_argument("--cache", required=True, help="existing local read-only mirror root")
    a = p.parse_args()
    try:
        m = json.loads(Path(a.manifest).read_text(), object_pairs_hook=no_duplicates)
        rows = m["references"]
        if not isinstance(rows, list) or {r["id"] for r in rows} != EXPECTED_IDS or len(rows) != len(EXPECTED_IDS):
            raise ValueError("expected precisely six unique source IDs")
        if not Path(a.cache).is_dir():
            raise ValueError("cache root does not exist: refusing to fetch")
        results = [verify_reference(row, a.cache) for row in rows]
        verdict = "ALL_LOCALLY_PINNED" if all(r["status"] == "LOCALLY_PINNED" for r in results) else "HOLD"
        print(json.dumps({"status": verdict, "count": len(results), "results": results,
                          "scope": "read-only local source integrity, not independent verification"}, indent=2))
        return 0 if verdict == "ALL_LOCALLY_PINNED" else 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        p.exit(2, f"HOLD: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
