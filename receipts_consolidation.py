#!/usr/bin/env python3
"""Consolidate SAP goods-receipt data (MB51-style export) by material and month.

Pipeline:
  1. Net quantities per material/date. Movement type 102 is a reversal of
     101: the source Quantity column is signed (102 rows negative, 101
     rows positive), so netting is a plain sum per material/date.
  2. Roll the netted daily quantities up to material/month totals.
  3. Compute descriptive statistics per month across all materials.
  4. Build a material x month matrix of the netted quantities.

Expected input columns (defaults match a standard SAP MB51 export):
  Material, Posting Date, Movement Type, Quantity
Column names are configurable via CLI flags for other export layouts. If a
source instead stores unsigned quantity magnitudes, pass --no-qty-is-signed
so the sign is derived from --negative-movement-types instead.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MATERIAL_COL = "Material"
DEFAULT_DATE_COL = "Posting Date"
DEFAULT_MOVEMENT_COL = "Movement Type"
DEFAULT_QTY_COL = "Quantity"
DEFAULT_NEGATIVE_MOVEMENT_TYPES = ("102",)


def _normalize_movement_type(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_receipts(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet, dtype={DEFAULT_MOVEMENT_COL: str})
    return pd.read_csv(path, dtype={DEFAULT_MOVEMENT_COL: str})


def net_daily_quantities(
    df: pd.DataFrame,
    material_col: str,
    date_col: str,
    movement_col: str,
    qty_col: str,
    negative_movement_types: tuple[str, ...],
    dayfirst: bool,
    qty_is_signed: bool,
) -> pd.DataFrame:
    working = df[[material_col, date_col, movement_col, qty_col]].copy()

    working[date_col] = pd.to_datetime(working[date_col], dayfirst=dayfirst, errors="coerce")
    dropped_dates = working[date_col].isna().sum()
    if dropped_dates:
        print(f"Warning: dropping {dropped_dates} row(s) with an unparseable {date_col!r}")
    working = working.dropna(subset=[date_col])

    working[qty_col] = pd.to_numeric(working[qty_col], errors="coerce").fillna(0.0)

    if qty_is_signed:
        working["signed_qty"] = working[qty_col]
    else:
        movement = working[movement_col].map(_normalize_movement_type)
        sign = np.where(movement.isin(negative_movement_types), -1.0, 1.0)
        working["signed_qty"] = working[qty_col] * sign

    working["date"] = working[date_col].dt.normalize()

    net = (
        working.groupby([material_col, "date"], as_index=False)["signed_qty"]
        .sum()
        .rename(columns={"signed_qty": "net_qty"})
    )
    return net


def consolidate_by_month(net_daily: pd.DataFrame, material_col: str) -> pd.DataFrame:
    working = net_daily.copy()
    working["month"] = working["date"].dt.to_period("M").astype(str)
    monthly = (
        working.groupby([material_col, "month"], as_index=False)["net_qty"]
        .sum()
    )
    return monthly


def compute_monthly_statistics(monthly: pd.DataFrame, bottom_percentile: float) -> pd.DataFrame:
    def summarize(group: pd.Series) -> pd.Series:
        return pd.Series(
            {
                "material_count": group.count(),
                "sum": group.sum(),
                "mean": group.mean(),
                "median": group.median(),
                f"p{bottom_percentile:g}_bottom": group.quantile(bottom_percentile / 100),
                "p75": group.quantile(0.75),
                "p95": group.quantile(0.95),
                "min": group.min(),
                "max": group.max(),
            }
        )

    stats = monthly.groupby("month")["net_qty"].apply(summarize).unstack()
    return stats.sort_index()


def build_material_month_matrix(monthly: pd.DataFrame, material_col: str) -> pd.DataFrame:
    matrix = monthly.pivot_table(
        index=material_col, columns="month", values="net_qty", aggfunc="sum", fill_value=0.0
    )
    return matrix.reindex(sorted(matrix.columns), axis=1)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_receipts(input_path, sheet=args.sheet)

    net_daily = net_daily_quantities(
        df,
        material_col=args.material_column,
        date_col=args.date_column,
        movement_col=args.movement_column,
        qty_col=args.qty_column,
        negative_movement_types=tuple(args.negative_movement_types.split(",")),
        dayfirst=not args.month_first_dates,
        qty_is_signed=args.qty_is_signed,
    )
    net_daily.to_csv(output_dir / "net_daily_by_material.csv", index=False)

    monthly = consolidate_by_month(net_daily, material_col=args.material_column)
    monthly.to_csv(output_dir / "consolidated_by_month.csv", index=False)

    stats = compute_monthly_statistics(monthly, bottom_percentile=args.bottom_percentile)
    stats.to_csv(output_dir / "monthly_statistics.csv")

    matrix = build_material_month_matrix(monthly, material_col=args.material_column)
    matrix.to_csv(output_dir / "material_month_matrix.csv")

    print(f"Materials: {matrix.shape[0]}, Months: {matrix.shape[1]}")
    print(f"Wrote outputs to {output_dir}/")
    print("\nMonthly statistics:")
    print(stats.round(2).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the receipts export (.csv or .xlsx)")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index (ignored for CSV)")
    parser.add_argument("--output-dir", default="output", help="Directory to write result CSVs to")
    parser.add_argument("--material-column", default=DEFAULT_MATERIAL_COL)
    parser.add_argument("--date-column", default=DEFAULT_DATE_COL)
    parser.add_argument("--movement-column", default=DEFAULT_MOVEMENT_COL)
    parser.add_argument("--qty-column", default=DEFAULT_QTY_COL)
    parser.add_argument(
        "--negative-movement-types",
        default=",".join(DEFAULT_NEGATIVE_MOVEMENT_TYPES),
        help="Comma-separated movement types subtracted during netting (e.g. reversals)",
    )
    parser.add_argument(
        "--qty-is-signed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantity column is already signed (e.g. 102 reversals stored as negative values). "
        "Use --no-qty-is-signed if the source stores unsigned magnitudes instead, in which case "
        "--negative-movement-types is applied to derive the sign.",
    )
    parser.add_argument(
        "--month-first-dates",
        action="store_true",
        help="Set if dates are MM/DD/YYYY instead of the SAP-default DD.MM.YYYY",
    )
    parser.add_argument(
        "--bottom-percentile",
        type=float,
        default=5.0,
        help="Percentile to report as the 'bottom percentile' (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
