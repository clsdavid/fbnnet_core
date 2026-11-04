"""Synthetic Affymetrix CDF/CEL files for tests (a 4 x 4 chip with 4 probe sets)."""
from pathlib import Path

import numpy as np

COLS = 4
# probe set -> list of (x, y, probe base, target base); PM when probe base complements target base
PROBE_SETS = {
    "AFFX-control_at": [(0, 0, "A", "T"), (1, 0, "T", "T")],
    "G1_at": [(2, 0, "A", "T"), (3, 0, "T", "T"), (0, 1, "C", "G"), (1, 1, "G", "G")],
    "G2_at": [(2, 1, "G", "C"), (3, 1, "C", "C"), (0, 2, "T", "A"), (1, 2, "A", "A")],
    "G3_at": [(2, 2, "A", "T"), (3, 2, "A", "A"), (0, 3, "C", "G"), (1, 3, "C", "C")],
}


def write_cdf(path: Path) -> None:
    lines = ["[CDF]", "Version=GC3.0", "", "[Chip]", f"Cols={COLS}", f"Rows={COLS}", "", "[QC1]", "Type=15"]
    for number, (name, cells) in enumerate(PROBE_SETS.items(), start=1):
        lines += ["", f"[Unit{number}]", "Name=NONE", "", f"[Unit{number}_Block1]", f"Name={name}"]
        lines.append("CellHeader=X\tY\tPROBE\tFEAT\tQUAL\tEXPOS\tPOS\tCBASE\tPBASE\tTBASE\tATOM\tINDEX")
        for n, (x, y, probe, target) in enumerate(cells, start=1):
            lines.append(f"Cell{n}={x}\t{y}\tN\tf\tq\t0\t13\t{target}\t{probe}\t{target}\t0\t{y * COLS + x}\t-1\t-1\t99\t")
    path.write_text("\n".join(lines) + "\n")


def write_cel(path: Path, intensities: np.ndarray) -> None:
    lines = ["[CEL]", "Version=3", "", "[HEADER]", f"Cols={COLS}", f"Rows={COLS}", "", "[INTENSITY]",
             f"NumberCells={COLS * COLS}", "CellHeader=X\tY\tMEAN\tSTDV\tNPIXELS"]
    # Scrambled row order: the reader must place values by x/y, not by line number.
    for index in np.random.default_rng(0).permutation(COLS * COLS):
        y, x = divmod(int(index), COLS)
        lines.append(f"  {x}\t  {y}\t{intensities[index]:.1f}\t1.0\t 25")
    lines += ["", "[MASKS]", "NumberCells=0"]
    path.write_text("\n".join(lines) + "\n")
