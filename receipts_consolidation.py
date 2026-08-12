#!/usr/bin/env python3
"""Consolidate SAP goods-receipt data (MB51-style export) by material and month.

Pipeline:
  1. Net quantities per material/plant/date. Movement type 102 is a
     reversal of 101: the source quantity column is signed (102 rows
     negative, 101 rows positive), so netting is a plain sum per
     material/plant/date. Plant is included because the same material can
     be stocked at multiple plants, and a reversal only cancels a receipt
     at the same plant.
  2. Roll the netted daily quantities up to material/plant/month totals.
  3. Compute descriptive statistics per material/plant (mean, median,
     bottom percentile, 75th, 95th, etc.) across that combination's
     monthly totals.
  4. Build a material+plant x month matrix of the netted quantities.

Also checks that each material/plant's rows share a single unit of entry,
since mixed units (e.g. EA and KG) would make summed/netted quantities
meaningless. Combinations mixing units from a known dimension (e.g. G and
KG, or CM and IN) are automatically converted to that dimension's base unit
(KG, M, or L) via UNIT_CONVERSIONS. Combinations mixing units with no known
conversion between them are excluded from the analysis entirely. Both
outcomes are reported in a "Unit Consistency" sheet and warned about on
stdout.

Results are written to a single .xlsx workbook with one sheet per step.

Expected input columns (defaults match a standard SAP MB51 export):
  Material, Plant, Posting Date, Movement Type, Qty in unit of entry, Unit of Entry
Column names are configurable via CLI flags for other export layouts. If a
source instead stores unsigned quantity magnitudes, pass --no-qty-is-signed
so the sign is derived from --negative-movement-types instead.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MATERIAL_COL = "Material"
DEFAULT_PLANT_COL = "Plant"
DEFAULT_DATE_COL = "Posting Date"
DEFAULT_MOVEMENT_COL = "Movement Type"
DEFAULT_QTY_COL = "Qty in unit of entry"
DEFAULT_UNIT_COL = "Unit of Entry"
DEFAULT_NEGATIVE_MOVEMENT_TYPES = ("102",)

# Standard unit conversions, grouped by dimension. Each unit maps to the
# factor that converts one of that unit into the dimension's base unit.
# Only units listed here can be reconciled automatically; anything else
# found mixed with another unit is excluded rather than guessed at.
UNIT_CONVERSIONS = {
    # Mass -> base KG
    "G": ("mass", 0.001),
    "GM": ("mass", 0.001),
    "MG": ("mass", 0.000001),
    "KG": ("mass", 1.0),
    "TO": ("mass", 1000.0),
    "T": ("mass", 1000.0),
    "LB": ("mass", 0.45359237),
    "OZ": ("mass", 0.028349523125),
    # Length -> base M
    "MM": ("length", 0.001),
    "CM": ("length", 0.01),
    "DM": ("length", 0.1),
    "M": ("length", 1.0),
    "KM": ("length", 1000.0),
    "IN": ("length", 0.0254),
    "FT": ("length", 0.3048),
    "YD": ("length", 0.9144),
    # Volume -> base L
    "ML": ("volume", 0.001),
    "CL": ("volume", 0.01),
    "L": ("volume", 1.0),
    "GAL": ("volume", 3.785411784),
}
BASE_UNIT_BY_DIMENSION = {"mass": "KG", "length": "M", "volume": "L"}


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
    plant_col: str,
    date_col: str,
    movement_col: str,
    qty_col: str,
    negative_movement_types: tuple[str, ...],
    dayfirst: bool,
    qty_is_signed: bool,
) -> pd.DataFrame:
    working = df[[material_col, plant_col, date_col, movement_col, qty_col]].copy()

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
        working.groupby([material_col, plant_col, "date"], as_index=False)["signed_qty"]
        .sum()
        .rename(columns={"signed_qty": "net_qty"})
    )
    return net


def resolve_units(
    df: pd.DataFrame, material_col: str, plant_col: str, unit_col: str, qty_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile each material/plant combination's unit of entry.

    - Combinations already using a single unit are left untouched.
    - Combinations mixing units from the same known dimension (e.g. G and
      KG) are converted to that dimension's base unit via
      UNIT_CONVERSIONS, so their quantities become summable.
    - Combinations mixing units with no known/common conversion (e.g. EA
      and KG) are excluded from the returned data entirely, since summing
      them would be meaningless.

    Returns (resolved_df, unit_report). resolved_df has qty_col converted
    in place for reconciled combinations and excluded combinations' rows
    dropped.
    """
    group_cols = [material_col, plant_col]
    working = df.copy()
    working[unit_col] = working[unit_col].astype(str).str.strip()
    normalized = working[unit_col].str.upper()

    records = []
    excluded_keys = set()
    converted_qty = working[qty_col].astype(float).copy()

    for key, idx in working.groupby(group_cols).groups.items():
        material, plant = key
        key_units = normalized.loc[idx]
        distinct_display = sorted({u for u in working.loc[idx, unit_col] if u and u.lower() != "nan"})
        distinct_norm = sorted({u for u in key_units if u and u.lower() != "nan"})

        if len(distinct_norm) <= 1:
            records.append(
                {
                    material_col: material,
                    plant_col: plant,
                    "units_found": ", ".join(distinct_display),
                    "unit_consistent": True,
                    "converted": False,
                    "target_unit": distinct_display[0] if distinct_display else "",
                    "conversion_applied": "",
                    "excluded": False,
                }
            )
            continue

        infos = {u: UNIT_CONVERSIONS.get(u) for u in distinct_norm}
        known = {u: info for u, info in infos.items() if info is not None}
        dimensions = {info[0] for info in known.values()}

        if len(known) == len(distinct_norm) and len(dimensions) == 1:
            dimension = next(iter(dimensions))
            target_unit = BASE_UNIT_BY_DIMENSION[dimension]
            factors = {u: known[u][1] for u in distinct_norm}
            for row_i in idx:
                converted_qty.loc[row_i] *= factors[key_units.loc[row_i]]
            working.loc[idx, unit_col] = target_unit
            conversion_applied = "; ".join(
                f"1 {u} = {factors[u]:g} {target_unit}" for u in distinct_norm if u != target_unit
            )
            records.append(
                {
                    material_col: material,
                    plant_col: plant,
                    "units_found": ", ".join(distinct_display),
                    "unit_consistent": True,
                    "converted": True,
                    "target_unit": target_unit,
                    "conversion_applied": conversion_applied,
                    "excluded": False,
                }
            )
        else:
            excluded_keys.add(key)
            records.append(
                {
                    material_col: material,
                    plant_col: plant,
                    "units_found": ", ".join(distinct_display),
                    "unit_consistent": False,
                    "converted": False,
                    "target_unit": "",
                    "conversion_applied": "",
                    "excluded": True,
                }
            )

    working[qty_col] = converted_qty
    excluded_mask = working.set_index(group_cols).index.isin(excluded_keys)
    resolved = working[~excluded_mask].copy()
    report = pd.DataFrame.from_records(records).sort_values(group_cols).reset_index(drop=True)
    return resolved, report


def consolidate_by_month(net_daily: pd.DataFrame, material_col: str, plant_col: str) -> pd.DataFrame:
    working = net_daily.copy()
    working["month"] = working["date"].dt.to_period("M").astype(str)
    monthly = (
        working.groupby([material_col, plant_col, "month"], as_index=False)["net_qty"]
        .sum()
    )
    return monthly


def compute_material_statistics(
    monthly: pd.DataFrame, material_col: str, plant_col: str, bottom_percentile: float
) -> pd.DataFrame:
    """One row per material/plant, one column per descriptive statistic,
    computed across that combination's monthly net quantities."""

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

    stats = monthly.groupby([material_col, plant_col])["net_qty"].apply(summarize).unstack()
    return stats.sort_index()


def build_material_month_matrix(monthly: pd.DataFrame, material_col: str, plant_col: str) -> pd.DataFrame:
    matrix = monthly.pivot_table(
        index=[material_col, plant_col], columns="month", values="net_qty", aggfunc="sum", fill_value=0.0
    )
    return matrix.reindex(sorted(matrix.columns), axis=1)


def run(args: argparse.Namespace, input_path: Path) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_receipts(input_path, sheet=args.sheet)

    group_cols = [args.material_column, args.plant_column]

    unit_check = None
    if args.unit_column in df.columns:
        df, unit_check = resolve_units(df, args.material_column, args.plant_column, args.unit_column, args.qty_column)

        converted = unit_check[unit_check["converted"]]
        excluded = unit_check[unit_check["excluded"]]
        if not converted.empty:
            print(f"Converted {len(converted)} material/plant combination(s) to a common unit via standard conversions:")
            print(converted[group_cols + ["units_found", "target_unit", "conversion_applied"]].to_string(index=False))
        if not excluded.empty:
            print(
                f"Warning: excluded {len(excluded)} material/plant combination(s) with no standard "
                f"conversion between their units (dropped from the analysis):"
            )
            print(excluded[group_cols + ["units_found"]].to_string(index=False))
        if converted.empty and excluded.empty:
            print(f"Unit check: all material/plant combinations use a consistent {args.unit_column!r}.")
    else:
        print(f"Note: unit column {args.unit_column!r} not found in input; skipping unit consistency check.")

    net_daily = net_daily_quantities(
        df,
        material_col=args.material_column,
        plant_col=args.plant_column,
        date_col=args.date_column,
        movement_col=args.movement_column,
        qty_col=args.qty_column,
        negative_movement_types=tuple(args.negative_movement_types.split(",")),
        dayfirst=not args.month_first_dates,
        qty_is_signed=args.qty_is_signed,
    )

    monthly = consolidate_by_month(net_daily, material_col=args.material_column, plant_col=args.plant_column)

    stats = compute_material_statistics(
        monthly, material_col=args.material_column, plant_col=args.plant_column, bottom_percentile=args.bottom_percentile
    )

    matrix = build_material_month_matrix(monthly, material_col=args.material_column, plant_col=args.plant_column)
    month_count = matrix.shape[1]

    if unit_check is not None:
        unit_check_indexed = unit_check.set_index(group_cols)
        stats = stats.merge(
            unit_check_indexed[["units_found", "unit_consistent", "converted", "target_unit", "conversion_applied"]],
            left_index=True,
            right_index=True,
            how="left",
        )
        unit_of_measure = unit_check_indexed["target_unit"]
        matrix.insert(0, "Unit of Measure", matrix.index.map(unit_of_measure))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # reset_index (rather than the default merge_cells=True) so Material/Plant are
        # repeated on every row instead of being blank on repeats -- blank cells read
        # back as NaN and make the sheet unusable as flat data.
        matrix.reset_index().to_excel(writer, sheet_name="Material x Month Matrix", index=False)
        stats.reset_index().to_excel(writer, sheet_name="Material Statistics", index=False)
        monthly.to_excel(writer, sheet_name="Consolidated by Month", index=False)
        net_daily.to_excel(writer, sheet_name="Net Daily by Material", index=False)
        if unit_check is not None:
            unit_check.to_excel(writer, sheet_name="Unit Consistency", index=False)

    print(f"Material/plant combinations: {matrix.shape[0]}, Months: {month_count}")
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
    parser.add_argument("--plant-column", default=DEFAULT_PLANT_COL)
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
