#!/usr/bin/env python3
# PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
# Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

"""Splice Lean-4-emitted GPU kernel images into include/sim/verified_lean_kernels.hpp.

Source of truth: ../rust-mmap-engine/kernels/*, produced by
`lake build emit_gpu && ./lake/build/bin/emit_gpu kernels` in rust-mmap-engine
(lean_rules: Rules.QrlobGpu, Rules.MarketSubgameGpu - INV-003 verified, ptxas
sm_89 compiled).

Embedding policy:
  - Lean-emitted qrlob kernels  -> PTX 8.0 raw strings (JIT-clean, readable).
  - nvcc-emitted LOB kernels    -> sm_89 CUBIN byte arrays. Their nvcc 13.3
    PTX declares ISA 9.3, which the driver JIT rejects with
    CUDA_ERROR_INVALID_IMAGE on drivers older than the toolkit; cubins are
    arch-bound and load wherever sm_89 is supported.

Marker-based splice: everything between each BEGIN/END GENERATED pair is
rewritten; hand-written launcher code is preserved.

Usage: python3 scripts/sync_lean_kernels.py [--check]
  --check  verify the embedded payloads match the sources (CI diff guard),
           exit 1 on drift without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PTX_ROOT = REPO.parent / "rust-mmap-engine" / "kernels"
HEADER = REPO / "include" / "sim" / "verified_lean_kernels.hpp"

PTX_BEGIN, PTX_END = "// BEGIN GENERATED PTX", "// END GENERATED PTX"
CUBIN_BEGIN, CUBIN_END = "// BEGIN GENERATED CUBIN", "// END GENERATED CUBIN"

PTX_PAYLOADS = [
    ("QrlobRolloutPtx", "qrlob_gpu_batch_qr_hawkes_rollout.ptx"),
    ("QrlobOzakiPtx", "qrlob_gpu_ozaki_score_fisher.ptx"),
]

CUBIN_PAYLOADS = [
    ("LobSubgameCubin", "verified_market_subgame.cubin"),
    ("LobRegretCubin", "online_regret_subgame.cubin"),
]


def build_ptx_section() -> str:
    lines = [PTX_BEGIN]
    for name, fname in PTX_PAYLOADS:
        src = PTX_ROOT / fname
        if not src.exists():
            sys.exit(f"missing PTX source: {src} (run emit_gpu in rust-mmap-engine first)")
        ptx = src.read_text(encoding="ascii").rstrip("\n")
        if ")LEANPTX" in ptx:
            sys.exit(f"PTX payload contains the raw-string terminator: {src}")
        lines.append(f'inline constexpr char k{name}[] =')
        lines.append(f'    R"LEANPTX({ptx})LEANPTX";')
    lines.append(PTX_END)
    return "\n".join(lines)


def build_cubin_section() -> str:
    lines = [CUBIN_BEGIN]
    for name, fname in CUBIN_PAYLOADS:
        src = PTX_ROOT / fname
        if not src.exists():
            sys.exit(f"missing CUBIN source: {src} (run emit_gpu in rust-mmap-engine first)")
        data = src.read_bytes()
        hexbytes = ", ".join(f"0x{b:02x}" for b in data)
        wrapped = "\n    ".join(
            hexbytes[i:i + 96] for i in range(0, len(hexbytes), 96)
        )
        lines.append(f"inline constexpr unsigned char k{name}[] = {{")
        lines.append(f"    {wrapped}}};")
    lines.append(CUBIN_END)
    return "\n".join(lines)


def splice(text: str, begin: str, end: str, section: str) -> str:
    b = text.find(begin)
    e = text.find(end)
    if b < 0 or e < 0 or e < b:
        sys.exit(f"markers not found in {HEADER}: {begin}")
    return text[:b] + section + text[e + len(end):]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    text = HEADER.read_text(encoding="utf-8")
    new = splice(text, PTX_BEGIN, PTX_END, build_ptx_section())
    new = splice(new, CUBIN_BEGIN, CUBIN_END, build_cubin_section())
    if new == text:
        print("lean kernels header up to date")
        return 0
    if args.check:
        print("DRIFT: embedded kernel images differ from rust-mmap-engine/kernels sources",
              file=sys.stderr)
        return 1
    HEADER.write_text(new, encoding="utf-8")
    print(f"updated {HEADER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
