"""Verify the supported LLVM distribution and fingerprint the Carbon build inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tree_digest(root):
    rows = [
        (str(p.relative_to(root)), digest(p))
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def verify(prefix, lock):
    for name, expected in lock["files"].items():
        if digest(prefix / name) != expected:
            raise ValueError(f"toolchain pin mismatch: {name}")
    for name, expected in lock["trees"].items():
        if tree_digest(prefix / name) != expected:
            raise ValueError(f"toolchain pin mismatch: {name}")
    for option, expected in (
        ("--version", lock["llvm"]["version"]),
        ("--build-mode", lock["distribution"]["build_mode"]),
        ("--host-target", lock["distribution"]["host_target"]),
    ):
        if command(str(prefix / "bin/llvm-config"), option) != expected:
            raise ValueError(f"toolchain pin mismatch: llvm-config {option}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().with_name("toolchain.json"),
    )
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--metal-cpp", type=Path, required=True)
    parser.add_argument("--build-type", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    verify(args.llvm.resolve(), lock)
    linked = {}
    for name in ("lib/libMLIR.dylib", "lib/libLLVM.dylib"):
        for line in command("otool", "-L", str(args.llvm / name)).splitlines()[1:]:
            path = Path(line.strip().split(" (")[0])
            if str(path).startswith("/opt/homebrew/"):
                linked[str(path)] = digest(path)
    manifest = {
        "schema": 1,
        "toolchain_id": digest(args.lock),
        "llvm": lock["llvm"],
        "distribution": lock["distribution"]["name"],
        "verified_llvm_prefix": str(args.llvm.resolve()),
        "linked_libraries": linked,
        "cxx": str(Path(args.cxx).resolve()),
        "cxx_version": command(args.cxx, "--version"),
        "sdk": command("xcrun", "--sdk", "macosx", "--show-sdk-version"),
        "sdk_path": command("xcrun", "--sdk", "macosx", "--show-sdk-path"),
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "python": sys.version,
        "nanobind": importlib.metadata.version("nanobind"),
        "metal_cpp_sha256": tree_digest(args.metal_cpp),
        "build_type": args.build_type,
        "cmake": command("cmake", "--version").splitlines()[0],
    }
    manifest["build_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode(),
    ).hexdigest()
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["build_id"])


if __name__ == "__main__":
    main()
