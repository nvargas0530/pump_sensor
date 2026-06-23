# -*- coding: utf-8 -*-
"""
Script de preprocesamiento de datos
===================================
Replica el tratamiento del notebook de EDA y genera un dataset limpio listo para
modelado: `data/sensor_clean.csv.gz`.

Pasos:
  1. Carga `data/sensor.csv.gz`.
  2. Elimina la columna índice redundante (`Unnamed: 0`) y `sensor_15` (100% nula).
  3. Convierte `timestamp` a índice temporal y `machine_status` a categórica.
  4. Imputa los sensores numéricos por interpolación temporal.
  5. Guarda el resultado comprimido.

Uso:   python data/preprocess.py
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
ENTRADA = BASE / "sensor.csv.gz"
SALIDA = BASE / "sensor_clean.csv.gz"

ORDEN_ESTADO = ["NORMAL", "RECOVERING", "BROKEN"]


def main() -> None:
    print(f"Leyendo {ENTRADA} ...")
    df = pd.read_csv(ENTRADA, compression="infer")
    print(f"  dimensiones originales: {df.shape}")

    # 2. Eliminar columnas sin valor analítico
    df = df.drop(columns=[c for c in ["Unnamed: 0", "sensor_15"] if c in df.columns])

    # 3. timestamp -> índice ; machine_status -> categórica
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df["machine_status"] = pd.Categorical(
        df["machine_status"], categories=ORDEN_ESTADO, ordered=True
    )

    # 4. Imputación temporal de sensores
    sensores = df.select_dtypes(include="number").columns.tolist()
    nulos_antes = int(df[sensores].isnull().sum().sum())
    df[sensores] = df[sensores].interpolate(method="time").ffill().bfill()
    nulos_despues = int(df[sensores].isnull().sum().sum())
    print(f"  nulos imputados: {nulos_antes:,} -> {nulos_despues:,}")

    # 5. Guardar
    df.to_csv(SALIDA, compression="gzip")
    print(f"Dataset limpio guardado en {SALIDA} (dimensiones {df.shape})")


if __name__ == "__main__":
    main()
