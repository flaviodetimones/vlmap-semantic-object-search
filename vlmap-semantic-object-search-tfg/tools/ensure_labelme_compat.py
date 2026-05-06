#!/usr/bin/env python3
"""Patch the installed LabelMe package for NumPy compatibility if needed."""

from __future__ import annotations

from pathlib import Path


LABELME_FILE = Path("/opt/conda/envs/tfg/lib/python3.9/site-packages/labelme/_label_file.py")


def main() -> None:
    if not LABELME_FILE.exists():
        print(f"LabelMe no encontrado en: {LABELME_FILE}")
        return

    text = LABELME_FILE.read_text(encoding="utf-8")
    fixed = text.replace("NDArray[np.bool]", "NDArray[np.bool_]")

    if fixed != text:
        LABELME_FILE.write_text(fixed, encoding="utf-8")
        print("Parche de compatibilidad de LabelMe aplicado.")
    else:
        print("LabelMe ya es compatible; no hace falta parchear.")


if __name__ == "__main__":
    main()
