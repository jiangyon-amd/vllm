#!/usr/bin/env python3
"""Check AMD GPU availability. Returns first free GPU ID or -1."""
import subprocess
import re
import sys
import time


def get_gpu_status():
    """Return list of (gpu_id, use_pct, vram_used_gb, vram_total_gb)."""
    gpus = []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        use_lines = re.findall(r"GPU\[(\d+)\].*GPU use \(%\): (\d+)", result.stdout)
        vram_used = re.findall(r"GPU\[(\d+)\].*VRAM Total Used Memory \(B\): (\d+)", result.stdout)
        vram_total = re.findall(r"GPU\[(\d+)\].*VRAM Total Memory \(B\): (\d+)", result.stdout)

        used_map = {int(g): int(v) for g, v in vram_used}
        total_map = {int(g): int(v) for g, v in vram_total}

        for gid_str, use_str in use_lines:
            gid = int(gid_str)
            use = int(use_str)
            used_gb = used_map.get(gid, 0) / 1e9
            total_gb = total_map.get(gid, 1) / 1e9
            gpus.append((gid, use, used_gb, total_gb))
    except Exception as e:
        print(f"Error querying GPU: {e}", file=sys.stderr)
    return gpus


def find_free_gpu(exclude=None):
    """Find a GPU with 0% usage and <10% VRAM used."""
    exclude = exclude or set()
    gpus = get_gpu_status()
    for gid, use, used_gb, total_gb in gpus:
        if gid in exclude:
            continue
        vram_pct = used_gb / total_gb * 100 if total_gb > 0 else 100
        if use == 0 and vram_pct < 10:
            return gid
    return -1


def main():
    gpus = get_gpu_status()
    print(f"{'GPU':>4} {'Use%':>5} {'VRAM Used':>10} {'VRAM Total':>11} {'Status':>8}")
    print("-" * 42)
    for gid, use, used_gb, total_gb in gpus:
        vram_pct = used_gb / total_gb * 100 if total_gb > 0 else 100
        status = "FREE" if use == 0 and vram_pct < 10 else "BUSY"
        print(f"{gid:>4} {use:>4}% {used_gb:>8.1f}GB {total_gb:>9.1f}GB {status:>8}")

    free = find_free_gpu()
    if free >= 0:
        print(f"\nFree GPU found: {free}")
    else:
        print("\nNo free GPU available")
    return free


if __name__ == "__main__":
    if "--wait" in sys.argv:
        poll_interval = 60
        print(f"Waiting for free GPU (polling every {poll_interval}s)...")
        while True:
            free = find_free_gpu()
            if free >= 0:
                print(f"GPU {free} is free!")
                sys.exit(free)
            time.sleep(poll_interval)
    else:
        result = main()
        sys.exit(0 if result >= 0 else 1)
