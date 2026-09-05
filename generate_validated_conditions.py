"""
GENERATE VALIDATED CONDITIONS
================================
Reads backtest_results.json (produced by backtest.py) and writes
validated_conditions.json — the file ConvictionGate in
wheat_monitor_pro.py already knows how to load automatically.

This closes the loop: instead of someone manually reading backtest
output and hand-editing accuracy numbers into two different gate
files (the exact kind of drift that let the old fabricated "100%"
numbers survive so long), this runs automatically after backtest.py
and produces a single source of truth both gates read from.

RULES FOR INCLUDING A CONDITION (all must hold):
  1. Not in STRUCTURAL_EXCLUSIONS — vol_low/vol_good/vol_high are
     permanently excluded regardless of what the backtest says,
     because of the confirmed Yahoo volume data lag for ZW=F
     (2026-07-10). A condition tied to broken data doesn't get
     re-included just because a number looks good.
  2. Holdout sample size >= MIN_HOLDOUT_N — small-sample holdout
     "wins" are not trustworthy (see backtest.py's own overfitting
     lesson from the original fake Tier 1).
  3. Holdout accuracy beats the real baseline UP rate — a condition
     that doesn't beat baseline adds no value even if its raw
     accuracy number looks high in isolation.
  4. UPDATED 2026-09-05, real bug found and confirmed — this only
     ever stored the accuracy NUMBER, never the direction it actually
     applied to. Every condition's accuracy here means "chance of
     UP" specifically (this project's established convention — see
     wheat_monitor_pro.py's ConvictionGate class docstring, which
     already documents bearish_month's OWN historical accuracy as
     "68.0% UP", not DOWN). A condition named with "bullish"/"bearish"
     whose real holdout best_direction doesn't match that name (e.g.
     bullish_month scoring 100% but for the DOWN direction, n=8,
     2026-09-05 run) is either mislabeled or reflects a pattern that
     flipped since it was named — either way it should not be
     silently trusted as-is. Excluded rather than included with a
     misleading name.

Usage:
  python3 generate_validated_conditions.py
  (run this right after backtest.py, same environment)
"""

import json
from pathlib import Path
from datetime import datetime

BACKTEST_RESULTS_FILE = Path("backtest_results.json")
OUTPUT_FILE = Path("validated_conditions.json")

# Permanently excluded regardless of backtest results — see docstring
STRUCTURAL_EXCLUSIONS = {'vol_low', 'vol_good', 'vol_high'}

MIN_HOLDOUT_N = 8  # don't trust a holdout "win" on fewer trades than this


def main():
    if not BACKTEST_RESULTS_FILE.exists():
        print(f"ERROR: {BACKTEST_RESULTS_FILE} not found. Run backtest.py first.")
        return

    data = json.loads(BACKTEST_RESULTS_FILE.read_text())
    baseline_up = data.get('baseline_up_full', 0.6746)

    conditions_data = data.get('individual_conditions_train_and_holdout', [])
    if not conditions_data:
        print("ERROR: backtest_results.json has no individual_conditions_train_and_holdout section.")
        print("Make sure you're using the updated backtest.py with holdout validation for individual conditions.")
        return

    validated = {}
    excluded_report = []

    for entry in conditions_data:
        name = entry.get('condition')
        if not name:
            continue

        if name in STRUCTURAL_EXCLUSIONS:
            excluded_report.append((name, "structurally excluded (unreliable data source)"))
            continue

        holdout = entry.get('holdout', {})
        n = holdout.get('n', 0)
        acc = holdout.get('best_accuracy')
        direction = holdout.get('best_direction')

        if n < MIN_HOLDOUT_N or acc is None:
            excluded_report.append((name, f"insufficient holdout sample (n={n}, need >={MIN_HOLDOUT_N})"))
            continue

        if acc <= baseline_up:
            excluded_report.append((name, f"does not beat baseline ({acc:.1%} <= {baseline_up:.1%})"))
            continue

        # Direction-name consistency check — see rule 4 above.
        name_lower = name.lower()
        if 'bullish' in name_lower and direction != 'UP':
            excluded_report.append((name, f"name implies UP but real holdout best_direction is "
                                            f"{direction} — mislabeled or pattern has flipped"))
            continue
        if 'bearish' in name_lower and direction != 'DOWN':
            excluded_report.append((name, f"name implies DOWN but real holdout best_direction is "
                                            f"{direction} — this project's own convention measures "
                                            f"UP-accuracy regardless of condition name (see "
                                            f"ConvictionGate's docstring) — flagged for manual review, "
                                            f"not silently trusted"))
            continue

        validated[name] = round(acc, 4)

    print(f"Baseline UP rate: {baseline_up:.1%}\n")
    print("VALIDATED (will be used by ConvictionGate):")
    if validated:
        for name, acc in sorted(validated.items(), key=lambda x: -x[1]):
            print(f"  {name:<15} {acc:.1%}")
    else:
        print("  (none — gate will run on baseline only)")

    print("\nEXCLUDED:")
    for name, reason in excluded_report:
        print(f"  {name:<15} {reason}")

    output = {
        'generated_at': datetime.now().isoformat(),
        'baseline_up': round(baseline_up, 4),
        'validated_conditions': validated,
        'source_backtest_run_date': data.get('run_date', 'unknown'),
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\n✅ Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
