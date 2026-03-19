#!/usr/bin/env python3
"""MoE Benchmark: RTN vs Separated vs Fused on Qwen3-30B."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

PORT = 8200
GPU_UTIL = "0.85"
OPENAI_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
CONCURRENCIES = [1, 4, 16, 32]
PYTHONPATH = "/data/jiangyon/vllm_rotation"

MODES = {
    "rtn": {
        "model": "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn",
        "env": {},
    },
    "separated": {
        "model": "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2",
        "env": {},
    },
    "fused": {
        "model": "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2",
        "env": {
            "VLLM_MOE_FUSED_ROTATION": "1",
        },
    },
    "hip_mfma": {
        "model": "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2",
        "env": {
            "VLLM_MOE_FUSED_ROTATION": "1",
            "VLLM_MOE_HIP_MFMA": "1",
        },
    },
}

BASE_ENV = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1",
}


def cleanup():
    os.system('pkill -9 -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1')
    os.system('pkill -9 -f "VLLM::EngineCore" >/dev/null 2>&1')
    time.sleep(6)


def wait_ready(timeout_s=360):
    for _ in range(timeout_s):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def server_warmup():
    for i in range(8):
        payload = json.dumps({
            "model": "placeholder",
            "messages": [{"role": "user", "content": f"warmup {i}"}],
            "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception:
            pass


def correctness_check(model_name):
    checks = []
    prompts = [
        ("只回答数字：1+1等于几？", "2"),
        ("只回答数字：3*3等于几？", "9"),
    ]
    for prompt, expected in prompts:
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"].strip()
                ok = expected in txt
                has_garbage = any(ord(c) > 0xFFFF for c in txt) or len(txt) > 50
                if has_garbage:
                    ok = False
                checks.append({"prompt": prompt, "expected": expected, "got": txt[:30], "ok": ok})
        except Exception as e:
            checks.append({"prompt": prompt, "expected": expected, "got": f"ERR:{type(e).__name__}", "ok": False})
    return checks


def run_bench(model_name, c, num_prompts, num_warmups):
    p = subprocess.run(
        [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "openai",
            "--base-url", f"http://localhost:{PORT}",
            "--model", model_name,
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(c),
            "--request-rate", "inf",
            "--dataset-name", "random",
            "--random-input-len", "512",
            "--random-output-len", "256",
            "--random-range-ratio", "0.2",
            "--num-warmups", str(num_warmups),
            "--seed", "42",
            "--extra-body", json.dumps(OPENAI_EXTRA_BODY),
        ],
        capture_output=True, text=True, timeout=1800,
    )

    def pick(key):
        for line in p.stdout.splitlines():
            if key in line:
                return line.split(":", 1)[1].strip()
        return "N/A"

    return {
        "tok_s": pick("Output token throughput"),
        "tpot": pick("Mean TPOT"),
        "ttft": pick("Mean TTFT"),
        "rc": str(p.returncode),
    }


def write_issues(results, gpu_id, output_dir):
    """Write issue report if any mode failed. Returns issue file path or None."""
    issues = []
    for mode_name, mode_results in results.items():
        # Check correctness
        checks = mode_results.get("correctness", [])
        if checks and not all(c.get("ok", False) for c in checks):
            issues.append({
                "mode": mode_name,
                "type": "correctness",
                "details": f"Correctness check failed: {[c for c in checks if not c.get('ok')]}",
                "checks": checks,
            })

        # Check timeouts
        for c in CONCURRENCIES:
            perf = mode_results.get(c, {})
            if perf.get("rc") == "timeout" or perf.get("tok_s") == "N/A":
                issues.append({
                    "mode": mode_name,
                    "type": "timeout",
                    "details": f"Server timeout or benchmark failure at c={c}",
                    "concurrency": c,
                    "result": perf,
                })
                break

            # Check garbage performance
            try:
                tok_s = float(perf.get("tok_s", "0"))
                tpot = float(perf.get("tpot", "999"))
                if tok_s <= 0 or tpot > 50:
                    issues.append({
                        "mode": mode_name,
                        "type": "performance_anomaly",
                        "details": f"Abnormal performance at c={c}: tok_s={tok_s}, tpot={tpot}",
                        "concurrency": c,
                        "result": perf,
                    })
            except (ValueError, TypeError):
                pass

    if issues:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        issue_path = os.path.join(output_dir, f"issues_{ts}.json")
        report = {
            "timestamp": datetime.now().isoformat(),
            "gpu_id": gpu_id,
            "issues": issues,
            "modes_tested": list(results.keys()),
            "fix_hints": {
                "timeout": "Check GPU memory (gpu_memory_utilization), .item() calls blocking CUDAGraph, or triton compile errors in server stderr",
                "correctness": "Check scale formula (0x200000 vs 0x400000), sorted_ids indexing, MAX_Q size",
                "performance_anomaly": "Check CUDAGraph compatibility, kernel launch overhead, or environment variable settings",
                "compile_error": "Check AMDMFMALayout instr_shape format ([16,16,32] for triton 3.6+)",
            },
            "key_files": {
                "kernel": "/data/jiangyon/vllm_rotation/vllm/model_executor/layers/quantization/quark/fused_rotation_mxfp4_quant_moe_sort.py",
                "gluon_v2": "/data/jiangyon/vllm_rotation/vllm/model_executor/layers/quantization/quark/fused_rotation_quant_gluon_v2.py",
                "gluon_kw8": "/data/jiangyon/vllm_rotation/vllm/model_executor/layers/quantization/quark/fused_rotation_quant_gluon_v2_kw8.py",
                "environment_rules": "/data/jiangyon/.cursor/rules/environment.mdc",
            },
        }
        with open(issue_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n⚠️  Issues detected! Report: {issue_path}")
        print(f"   {len(issues)} issue(s) found across modes: {set(i['mode'] for i in issues)}")
        return issue_path
    return None


def write_report(results, gpu_id, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# MoE Benchmark Results — {now}\n",
        "## Environment",
        f"- GPU: {gpu_id} (AMD MI300X)",
        "- Model: Qwen3-30B MoE MXFP4",
        f"- Concurrencies: {CONCURRENCIES}",
        "",
        "## Correctness",
        "| Mode | Q1 (1+1) | Q2 (3*3) | Status |",
        "|------|----------|----------|--------|",
    ]
    for mode in ("rtn", "separated", "fused"):
        checks = results.get(mode, {}).get("correctness", [])
        if not checks:
            lines.append(f"| {mode} | N/A | N/A | SKIP |")
        else:
            q1 = checks[0] if len(checks) > 0 else {"got": "N/A", "ok": False}
            q2 = checks[1] if len(checks) > 1 else {"got": "N/A", "ok": False}
            status = "PASS" if all(c["ok"] for c in checks) else "FAIL"
            lines.append(f"| {mode} | {q1['got']} | {q2['got']} | {status} |")

    lines += [
        "",
        "## Performance (tok/s)",
        "| c | RTN | Separated | Fused | Sep vs RTN | Fused vs Sep |",
        "|---|-----|-----------|-------|------------|--------------|",
    ]
    for c in CONCURRENCIES:
        rtn = results.get("rtn", {}).get(c, {}).get("tok_s", "N/A")
        sep = results.get("separated", {}).get(c, {}).get("tok_s", "N/A")
        fus = results.get("fused", {}).get(c, {}).get("tok_s", "N/A")
        try:
            sep_vs_rtn = f"{(float(sep) - float(rtn)) / float(rtn) * 100:+.1f}%"
        except:
            sep_vs_rtn = "N/A"
        try:
            fus_vs_sep = f"{(float(fus) - float(sep)) / float(sep) * 100:+.1f}%"
        except:
            fus_vs_sep = "N/A"
        lines.append(f"| c={c} | {rtn} | {sep} | {fus} | {sep_vs_rtn} | {fus_vs_sep} |")

    lines += [
        "",
        "## Performance (TPOT ms)",
        "| c | RTN | Separated | Fused |",
        "|---|-----|-----------|-------|",
    ]
    for c in CONCURRENCIES:
        rtn = results.get("rtn", {}).get(c, {}).get("tpot", "N/A")
        sep = results.get("separated", {}).get(c, {}).get("tpot", "N/A")
        fus = results.get("fused", {}).get(c, {}).get("tpot", "N/A")
        lines.append(f"| c={c} | {rtn} | {sep} | {fus} |")

    lines += [
        "",
        "## Performance (TTFT ms)",
        "| c | RTN | Separated | Fused |",
        "|---|-----|-----------|-------|",
    ]
    for c in CONCURRENCIES:
        rtn = results.get("rtn", {}).get(c, {}).get("ttft", "N/A")
        sep = results.get("separated", {}).get(c, {}).get("ttft", "N/A")
        fus = results.get("fused", {}).get(c, {}).get("ttft", "N/A")
        lines.append(f"| c={c} | {rtn} | {sep} | {fus} |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4, help="GPU ID to use")
    parser.add_argument("--modes", nargs="+", default=["rtn", "fused", "hip_mfma"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gpu_id = args.gpu
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"/data/jiangyon/vllm_rotation/bench_results/moe_benchmark_{ts}.md"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {}
    print(f"=== MoE Benchmark on GPU {gpu_id} ===", flush=True)

    for mode_name in args.modes:
        mode = MODES.get(mode_name)
        if not mode:
            print(f"Unknown mode: {mode_name}", flush=True)
            continue

        cleanup()
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(mode["env"])
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = PYTHONPATH + ":" + env.get("PYTHONPATH", "")

        model = mode["model"]
        print(f"\n[{mode_name}] Starting server (model={os.path.basename(model)})...", flush=True)

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model,
                "--port", str(PORT),
                "--trust-remote-code",
                "--disable-log-requests",
                "--max-model-len", "4096",
                "--gpu-memory-utilization", GPU_UTIL,
                "--host", "0.0.0.0",
            ],
            env=env, stdout=subprocess.DEVNULL,
            stderr=open(f"/tmp/vllm_{mode_name}.log", "w"),
        )

        try:
            if not wait_ready():
                print(f"[{mode_name}] Server timeout!", flush=True)
                results[mode_name] = {"correctness": [], **{c: {"tok_s": "N/A", "tpot": "N/A", "ttft": "N/A", "rc": "timeout"} for c in CONCURRENCIES}}
                continue

            server_warmup()

            # Correctness
            checks = correctness_check(model)
            results[mode_name] = {"correctness": checks}
            all_ok = all(c["ok"] for c in checks)
            print(f"[{mode_name}] Correctness: {'PASS' if all_ok else 'FAIL'} — {[c['got'] for c in checks]}", flush=True)

            # Benchmark
            for c in CONCURRENCIES:
                np_val = 20 if c <= 4 else (40 if c <= 16 else 60)
                nw_val = 4 if c <= 8 else 6
                _ = run_bench(model, c, max(8, np_val // 3), 2)
                print(f"[{mode_name}] bench c={c}...", flush=True)
                results[mode_name][c] = run_bench(model, c, np_val, nw_val)
                print(f"[{mode_name}] c={c} -> {results[mode_name][c]}", flush=True)

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    cleanup()
    write_report(results, gpu_id, args.output)

    # Check for issues and write issue report
    issue_path = write_issues(results, gpu_id, os.path.dirname(args.output))
    if issue_path:
        print(f"\n{'!'*60}")
        print(f"ISSUES DETECTED — see {issue_path}")
        print(f"To auto-fix: spawn a subagent with the issue file path")
        print(f"{'!'*60}")

    # Print summary
    print("\n=== Summary (tok/s) ===", flush=True)
    print(f"{'c':<5}", end="")
    for m in args.modes:
        print(f" {m:>12}", end="")
    print()
    for c in CONCURRENCIES:
        print(f"c={c:<3}", end="")
        for m in args.modes:
            v = results.get(m, {}).get(c, {}).get("tok_s", "N/A")
            print(f" {v:>12}", end="")
        print()


if __name__ == "__main__":
    main()
