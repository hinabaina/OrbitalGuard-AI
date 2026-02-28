import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def score_event(ev: dict):
    d = float(ev.get("distance_km", 999999))
    v = float(ev.get("velocity_km_s", 0))
    m = float(ev.get("mass_kg", 0))

    d_term = clamp(1.0 / (1.0 + d / 2000.0))
    v_term = clamp((v - 5.0) / 25.0)
    m_term = clamp((m - 500.0) / 4500.0)

    score = clamp(0.65 * d_term + 0.30 * v_term + 0.05 * m_term)
    explanation = f"dist={d:.0f}km, vel={v:.2f}km/s, mass={m:.0f}kg"
    return score, explanation


def assign_class_by_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    high_thr = df["ai_risk_score"].quantile(0.85)
    med_thr = df["ai_risk_score"].quantile(0.50)

    def class_by_thr(s):
        if s >= high_thr:
            return "HIGH"
        elif s >= med_thr:
            return "MEDIUM"
        return "LOW"

    df["ai_risk_class"] = df["ai_risk_score"].apply(class_by_thr)
    return df


def main():
    parser = argparse.ArgumentParser(description="orbitalguard ai risk scoring (mvp).")
    parser.add_argument(
        "--in",
        dest="inp",
        default=str(PROJECT_ROOT / "data" / "space_events_sample.jsonl"),
        help="input events file. default: data/space_events_sample.jsonl",
    )
    parser.add_argument(
        "--out-jsonl",
        dest="out_jsonl",
        default=str(PROJECT_ROOT / "outputs" / "scored_events.jsonl"),
        help="output scored jsonl. default: outputs/scored_events.jsonl",
    )
    parser.add_argument(
        "--out-csv",
        dest="out_csv",
        default=str(PROJECT_ROOT / "outputs" / "scored_events.csv"),
        help="output scored csv. default: outputs/scored_events.csv",
    )
    parser.add_argument(
        "--alerts",
        dest="alerts_n",
        type=int,
        default=5,
        help="how many high alerts to show. default: 5",
    )
    args = parser.parse_args()

    input_path = Path(args.inp)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"input file not found: {input_path}\n"
            f"expected default: {PROJECT_ROOT / 'data' / 'space_events_sample.jsonl'}"
        )

    events = load_jsonl(input_path)

    for ev in events:
        score, expl = score_event(ev)
        ev["ai_risk_score"] = score
        ev["ai_explanation"] = expl

    df = pd.DataFrame(events)
    df = df.sort_values("ai_risk_score", ascending=False).reset_index(drop=True)
    df = assign_class_by_percentiles(df)

    scored_events = df.to_dict(orient="records")
    save_jsonl(out_jsonl, scored_events)
    df.to_csv(out_csv, index=False)

    cols = [c for c in [
        "event_id", "object_id", "distance_km", "velocity_km_s", "mass_kg",
        "risk_level", "ai_risk_score", "ai_risk_class", "ai_explanation"
    ] if c in df.columns]

    print("\nTOP-10 MOST RISKY EVENTS (AI):\n")
    print(df.head(10)[cols].to_string(index=False))

    print("\nALERTS (HIGH risk):\n")
    alerts = df[df["ai_risk_class"] == "HIGH"].head(args.alerts_n)

    if len(alerts) == 0:
        print("No HIGH risk alerts.")
    else:
        show_console = [c for c in [
            "event_id", "object_id", "distance_km", "velocity_km_s",
            "ai_risk_score", "ai_explanation"
        ] if c in df.columns]
        print(alerts[show_console].to_string(index=False))

    alerts_md = PROJECT_ROOT / "outputs" / "alerts.md"
    alerts_md.parent.mkdir(parents=True, exist_ok=True)

    with alerts_md.open("w", encoding="utf-8") as f:
        f.write("# OrbitalGuard AI — Alerts (HIGH risk)\n\n")
        if len(alerts) == 0:
            f.write("No HIGH risk alerts.\n")
        else:
            show_md = [c for c in [
                "event_id", "object_id", "distance_km", "velocity_km_s",
                "ai_risk_score", "ai_explanation"
            ] if c in df.columns]
            f.write(alerts[show_md].to_markdown(index=False))

    print(f"\nSaved: {alerts_md}")
    print(f"Saved: {out_jsonl}")
    print(f"Saved: {out_csv}")

    if "risk_level" in df.columns:
        backend = df["risk_level"].astype(str).str.upper().replace({"MED": "MEDIUM"})
        ai = df["ai_risk_class"].astype(str).str.upper()

        same = (backend == ai).sum()
        total = len(df)
        print(f"\nAI vs backend match: {same}/{total} events")

        mism = df[backend != ai].copy()
        if len(mism) > 0:
            print("\nMISMATCHES (top):\n")
            print(
                mism.head(10)[
                    ["event_id", "distance_km", "velocity_km_s", "risk_level", "ai_risk_class", "ai_risk_score"]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()