#!/usr/bin/env python3
"""Lanzador de un solo paso: descarga 60 días de velas 5m de IAG y analiza.

Pensado para ejecutarse en un entorno CON acceso de red a Yahoo (en local, o en
una sesión de Claude Code en web cuyo entorno tenga *.yahoo.com en la lista de
dominios permitidos). Hace, de una vez:

  1) descarga IAG.MC en velas de 5 minutos de los ~60 últimos días;
  2) guarda una copia limpia en  iag_5min_60d.csv;
  3) ejecuta el estudio de eventos completo (costes, stop opcional, estadística
     y figuras) dejando los resultados en  iag_intraday_results/.

Uso:
    python descargar_60d.py                 # descarga + análisis con costes por defecto
    python descargar_60d.py --stop-pct 1.0  # además, con stop-loss del 1%

Cualquier opción de iag_intraday_event_study.py (--cost-bps, --spread-bps,
--borrow-annual-pct, --stop-pct, --output) se puede pasar aquí y se reenvía.
Ejecútalo desde la raíz del repositorio.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import iag_intraday_event_study as study

FRIENDLY_CSV = "iag_5min_60d.csv"


def main() -> int:
    # Fuerza periodo/intervalo y reenvía el resto de argumentos del usuario.
    passthrough = sys.argv[1:]
    sys.argv = [
        "iag_intraday_event_study.py",
        "--ticker", "IAG.MC",
        "--period", "60d",
        "--interval", "5m",
        *passthrough,
    ]

    rc = study.main()
    if rc != 0:
        print(
            "\nLa descarga falló. Si ves un error de proxy/403, este entorno no tiene "
            "acceso a Yahoo: abre la red del entorno (*.yahoo.com) o ejecútalo en local.",
            file=sys.stderr,
        )
        return rc

    # Copia el CSV crudo a un nombre fácil de recordar/subir.
    raw = Path("iag_intraday_results") / "iag_5min_raw.csv"
    if raw.exists():
        shutil.copy(raw, FRIENDLY_CSV)
        print(f"\nCopia limpia de las velas guardada en: {Path(FRIENDLY_CSV).resolve()}")
        print(f"Puedes reanalizarla cuando quieras con:  python iag_intraday_event_study.py --csv {FRIENDLY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
