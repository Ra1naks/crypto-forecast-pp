"""Обновляет оглавление отчёта и экспортирует docx/pptx в PDF через установленный Microsoft Office (COM).

Запуск: powershell-совместимая среда Windows с Word и PowerPoint.
    python scripts/render_office.py docs/report.docx docs/presentation.pptx
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_ps(script: str) -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def render(path: Path) -> Path:
    path = path.resolve()
    pdf = path.with_suffix(".pdf")
    if path.suffix == ".docx":
        run_ps(
            "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
            f"$d = $w.Documents.Open('{path}'); "
            "foreach ($t in $d.TablesOfContents) { $t.Update() }; $d.Fields.Update() | Out-Null; $d.Save(); "
            f"$d.ExportAsFixedFormat('{pdf}', 17); $d.Close(); $w.Quit()"
        )
    elif path.suffix == ".pptx":
        run_ps(
            "$p = New-Object -ComObject PowerPoint.Application; "
            f"$d = $p.Presentations.Open('{path}', $true, $false, $false); "
            f"$d.SaveAs('{pdf}', 32); $d.Close(); $p.Quit()"
        )
    return pdf


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(render(Path(arg)))
