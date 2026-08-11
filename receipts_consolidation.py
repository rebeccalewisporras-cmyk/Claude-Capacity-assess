#!/usr/bin/env python3
"""Consolidate SAP goods-receipt data (MB51-style export) by material and month.

Pipeline:
  1. Net quantities per material/date. Movement type 102 is a reversal of
     101: the source Quantity column is signed (102 rows negative, 101
     rows positive), so netting is a plain sum per material/date.
  2. Roll the netted daily quantities up to material/month totals.
  3. Compute descriptive statistics per material (mean, median, bottom
     percentile, 75th, 95th, etc.) across that material's monthly totals.
  4. Build a material x month matrix of the netted quantities.

Also checks that each material's rows share a single unit of entry, since
mixed units (e.g. EA and KG) would make summed/netted quantities meaningless.
Materials with mixed units are flagged in a "Unit Consistency" sheet and
warned about on stdout.

Results are written to a single .xlsx workbook with one sheet per step.

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
DEFAULT_UNIT_COL = "Unit of Entry"
DEFAULT_NEGATIVE_MOVEMENT_TYPES = ("102",)


def prompt_for_input_path() -> Path:
    """Ask the user where the raw receipts data lives (.csv or .xlsx)."""
    while True:
        raw = input("Path to receipts file (.csv or .xlsx): ").strip().strip('"')
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            return path
        print(f"'{path}' does not exist or is not a file. Try again.")


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


def check_unit_consistency(df: pd.DataFrame, material_col: str, unit_col: str) -> pd.DataFrame:
    """Flag materials whose rows don't all share the same unit of entry.

    Summing/netting a material's Quantity only makes sense if every row for
    that material was recorded in the same unit; mixed units (e.g. EA and
    KG) would make the netted totals meaningless.
    """
    working = df[[material_col, unit_col]].dropna(subset=[unit_col])
    working[unit_col] = working[unit_col].astype(str).str.strip()

    grouped = working.groupby(material_col)[unit_col].agg(lambda s: sorted(set(s)))
    result = grouped.reset_index()
    result.columns = [material_col, "units_found"]
    result["unit_consistent"] = result["units_found"].apply(lambda units: len(units) <= 1)
    result["units_found"] = result["units_found"].apply(", ".join)
    return result.sort_values(material_col).reset_index(drop=True)


def consolidate_by_month(net_daily: pd.DataFrame, material_col: str) -> pd.DataFrame:
    working = net_daily.copy()
    working["month"] = working["date"].dt.to_period("M").astype(str)
    monthly = (
        working.groupby([material_col, "month"], as_index=False)["net_qty"]
        .sum()
    )
    return monthly


def compute_material_statistics(
    monthly: pd.DataFrame, material_col: str, bottom_percentile: float
) -> pd.DataFrame:
    """One row per material, one column per descriptive statistic, computed
    across that material's monthly net quantities."""

    def summarize(group: pd.Series) -> pd.Series:
        return pd.Series(
            {
                "month_count": group.count(),
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

    stats = monthly.groupby(material_col)["net_qty"].apply(summarize).unstack()
    return stats.sort_index()


def build_material_month_matrix(monthly: pd.DataFrame, material_col: str) -> pd.DataFrame:
    matrix = monthly.pivot_table(
        index=material_col, columns="month", values="net_qty", aggfunc="sum", fill_value=0.0
    )
    return matrix.reindex(sorted(matrix.columns), axis=1)


def run(args: argparse.Namespace, input_path: Path) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_receipts(input_path, sheet=args.sheet)

    unit_check = None
    if args.unit_column in df.columns:
        unit_check = check_unit_consistency(df, args.material_column, args.unit_column)
        inconsistent = unit_check[~unit_check["unit_consistent"]]
        if not inconsistent.empty:
            print(
                f"Warning: {len(inconsistent)} material(s) have inconsistent "
                f"{args.unit_column!r} values across rows:"
            )
            print(inconsistent.to_string(index=False))
        else:
            print(f"Unit check: all materials use a consistent {args.unit_column!r}.")
    else:
        print(f"Note: unit column {args.unit_column!r} not found in input; skipping unit consistency check.")

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

    monthly = consolidate_by_month(net_daily, material_col=args.material_column)

    stats = compute_material_statistics(
        monthly, material_col=args.material_column, bottom_percentile=args.bottom_percentile
    )

    matrix = build_material_month_matrix(monthly, material_col=args.material_column)

    if unit_check is not None:
        stats = stats.merge(
            unit_check.set_index(args.material_column)[["units_found", "unit_consistent"]],
            left_index=True,
            right_index=True,
            how="left",
        )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        matrix.to_excel(writer, sheet_name="Material x Month Matrix")
        stats.to_excel(writer, sheet_name="Material Statistics")
        monthly.to_excel(writer, sheet_name="Consolidated by Month", index=False)
        net_daily.to_excel(writer, sheet_name="Net Daily by Material", index=False)
        if unit_check is not None:
            unit_check.to_excel(writer, sheet_name="Unit Consistency", index=False)

    print(f"Materials: {matrix.shape[0]}, Months: {matrix.shape[1]}")
    print(f"Wrote {output_path}")
    print("\nMaterial statistics (across months):")
    print(stats.round(2).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", help="Path to the receipts export (.csv or .xlsx). Prompted for interactively if omitted."
    )
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index (ignored for CSV)")
    parser.add_argument(
        "--output",
        default="output/receipts_consolidation.xlsx",
        help="Path to the output .xlsx workbook to write",
    )
    parser.add_argument("--material-column", default=DEFAULT_MATERIAL_COL)
    parser.add_argument("--date-column", default=DEFAULT_DATE_COL)
    parser.add_argument("--movement-column", default=DEFAULT_MOVEMENT_COL)
    parser.add_argument("--qty-column", default=DEFAULT_QTY_COL)
    parser.add_argument(
        "--unit-column",
        default=DEFAULT_UNIT_COL,
        help="Column to check for a consistent unit per material (e.g. 'Unit of Entry' or "
        "'Base Unit of Measure'). Skipped if the column isn't present in the input.",
    )
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
    # Get the raw data location first, before anything else runs.
    cli_args = parse_args()
    receipts_path = Path(cli_args.input) if cli_args.input else prompt_for_input_path()

    run(cli_args, receipts_path)
