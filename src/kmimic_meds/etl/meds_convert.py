"""
Standalone MEDS conversion pipeline for K-MIMIC SYN-ICU.

Bypasses MEDS-Extract CLI entirely (which has Windows compatibility issues).
Reads intermediate Parquet files produced by pre_meds.py and produces a
fully MEDS-compliant dataset directly using pandas + pyarrow.

Output structure:
    data/output/
    ├── data/
    │   ├── train/0.parquet
    │   ├── tuning/0.parquet
    │   └── held_out/0.parquet
    └── metadata/
        ├── codes.parquet
        ├── dataset.json
        └── subject_splits.parquet
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# MEDS schema
# ---------------------------------------------------------------------------

MEDS_SCHEMA = pa.schema([
    pa.field("subject_id", pa.int64()),
    pa.field("time", pa.timestamp("us")),
    pa.field("code", pa.string()),
    pa.field("numeric_value", pa.float32()),
])

CODES_SCHEMA = pa.schema([
    pa.field("code", pa.string()),
    pa.field("description", pa.string()),
    pa.field("parent_codes", pa.list_(pa.string())),
])

SPLITS_SCHEMA = pa.schema([
    pa.field("subject_id", pa.int64()),
    pa.field("split", pa.string()),
])

# Valeurs considérées comme vides dans les colonnes source
_EMPTY = ("", "nan", "None", "UNK", "NaN", "none", "null", "NULL")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_row(subject_id, time, code, numeric_value=None):
    """Crée un dictionnaire représentant une ligne MEDS."""
    return {
        "subject_id": subject_id,
        "time": time,
        "code": code,
        "numeric_value": float(numeric_value) if numeric_value is not None and pd.notna(numeric_value) else None,
    }


def make_code(*parts):
    """
    Construit un code MEDS en joignant les parties avec //.
    Les parties vides, nan, None, UNK sont ignorées.

    Exemple :
        make_code("CHARTEVENT", "001C_102", "회/min") -> "CHARTEVENT//001C_102//회/min"
        make_code("HOSPITAL_ADMISSION", "nan", "Home") -> "HOSPITAL_ADMISSION//Home"
    """
    clean_parts = [
        str(p).strip()
        for p in parts
        if p is not None and str(p).strip() not in _EMPTY
    ]
    return "//".join(clean_parts) if clean_parts else "UNKNOWN"


def clean(val):
    """Retourne None si la valeur est vide/nan, sinon retourne la valeur en string."""
    s = str(val).strip()
    return None if s in _EMPTY else s


# ---------------------------------------------------------------------------
# Event extractors
# Chaque fonction lit une table intermédiaire et retourne un DataFrame
# avec les colonnes [subject_id, time, code, numeric_value].
# ---------------------------------------------------------------------------

def extract_patients(df):
    rows = []
    for _, r in df.iterrows():
        sid = r["subject_id"]
        if pd.isna(sid):
            continue
        sid = int(sid)

        # naissance — on construit un timestamp à partir de year_of_birth
        yob = clean(r.get("year_of_birth", ""))
        if yob:
            try:
                rows.append(make_row(sid, pd.Timestamp(f"{yob}-01-01 00:00:01"), "MEDS_BIRTH"))
            except Exception:
                pass

        # genre — événement statique (time = NaT)
        sex = clean(r.get("sex", ""))
        if sex:
            rows.append(make_row(sid, pd.NaT, make_code("GENDER", sex)))

        # décès
        dod = r.get("dod")
        if pd.notna(dod):
            rows.append(make_row(sid, pd.Timestamp(dod), "MEDS_DEATH"))

    return pd.DataFrame(rows)


def extract_admissions(df):
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        adm_time = r.get("admittime")
        if pd.notna(adm_time):
            # admission hospitalière
            rows.append(make_row(
                sid, adm_time,
                make_code("HOSPITAL_ADMISSION", r.get("admission_type"), r.get("admission_location"))
            ))
            # données démographiques recueillies à l'admission
            for col, prefix in [
                ("insurance", "INSURANCE"),
                ("marital_status", "MARITAL_STATUS"),
                ("ethnicity", "ETHNICITY"),
            ]:
                val = clean(r.get(col, ""))
                if val:
                    rows.append(make_row(sid, adm_time, make_code(prefix, val)))

        # sortie hospitalière
        dis_time = r.get("dischtime")
        if pd.notna(dis_time):
            rows.append(make_row(
                sid, dis_time,
                make_code("HOSPITAL_DISCHARGE", r.get("discharge_location"))
            ))

        # décès en cours d'hospitalisation
        death_time = r.get("deathtime")
        if pd.notna(death_time):
            rows.append(make_row(sid, death_time, "MEDS_DEATH"))

        # urgences
        edreg = r.get("edregtime")
        if pd.notna(edreg):
            rows.append(make_row(sid, edreg, "ED_REGISTRATION"))

        edout = r.get("edouttime")
        if pd.notna(edout):
            rows.append(make_row(sid, edout, "ED_OUT"))

    return pd.DataFrame(rows)


def extract_icustays(df):
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        intime = r.get("intime")
        if pd.notna(intime):
            rows.append(make_row(sid, intime, make_code("ICU_ADMISSION", r.get("first_careunit"))))

        outtime = r.get("outtime")
        if pd.notna(outtime):
            rows.append(make_row(sid, outtime, make_code("ICU_DISCHARGE", r.get("last_careunit"))))

    return pd.DataFrame(rows)


def extract_chartevents(df):
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        t = r.get("charttime")
        if pd.isna(t):
            continue

        code = make_code("CHARTEVENT", r.get("itemid"), r.get("valueuom"))
        num = r.get("valuenum")
        rows.append(make_row(sid, t, code, num if pd.notna(num) else None))

    return pd.DataFrame(rows)


def extract_labevents(df):
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        t = r.get("charttime")
        if pd.isna(t):
            continue

        code = make_code("LAB", r.get("itemid"), r.get("valueuom"))
        num = r.get("valuenum")
        rows.append(make_row(sid, t, code, num if pd.notna(num) else None))

    return pd.DataFrame(rows)


def extract_diagnoses_icd(df):
    """
    Les diagnostics ICD sont traités comme statiques (time = NaT)
    car la table ne contient pas de timestamp précis,
    seulement un hadm_id (admission hospitalière).
    """
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        icd_code = clean(r.get("icd_code", ""))
        if not icd_code:
            continue

        code = make_code("DIAGNOSIS", r.get("icd_version"), icd_code)
        rows.append(make_row(sid, pd.NaT, code))

    return pd.DataFrame(rows)


def extract_procedures_icd(df):
    """
    chartdate est une date-only résolue à 23:59:59 par pre_meds.py
    pour éviter la fuite temporelle.
    """
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        icd_code = clean(r.get("icd_code", ""))
        if not icd_code:
            continue

        t = r.get("chartdate")
        code = make_code("PROCEDURE_ICD", r.get("icd_version"), icd_code)
        rows.append(make_row(sid, t if pd.notna(t) else pd.NaT, code))

    return pd.DataFrame(rows)


def extract_inputevents(df):
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        t = r.get("starttime")
        if pd.isna(t):
            continue

        code = make_code("INPUT_START", r.get("itemid"), r.get("amountuom"))
        num = r.get("amount")
        rows.append(make_row(sid, t, code, num if pd.notna(num) else None))

    return pd.DataFrame(rows)


def extract_outputevents(df):
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        t = r.get("charttime")
        if pd.isna(t):
            continue

        code = make_code("OUTPUT", r.get("itemid"), r.get("valueuom"))
        num = r.get("value")
        rows.append(make_row(sid, t, code, num if pd.notna(num) else None))

    return pd.DataFrame(rows)


def extract_emar(df):
    rows = []
    for _, r in df.iterrows():
        sid = r.get("subject_id")
        if pd.isna(sid):
            continue
        sid = int(sid)

        t = r.get("charttime")
        if pd.isna(t):
            continue

        code = make_code("MEDICATION", r.get("itemid"))
        rows.append(make_row(sid, t, code))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def build_codes_parquet(events_df, intermediate_dir=None):
    """
    Construit codes.parquet avec la liste des codes uniques.
    Si intermediate_dir est fourni, enrichit avec les descriptions
    issues de syn_d_items et syn_d_labitems.
    """
    codes = sorted(events_df["code"].dropna().unique())
    df = pd.DataFrame({
        "code": codes,
        "description": [None] * len(codes),
        "parent_codes": [None] * len(codes),
    })

    if intermediate_dir is not None:
        # construire un dictionnaire itemid -> label depuis les tables de dimension
        label_map = {}
        for fname in ["syn_d_items.parquet", "syn_d_labitems.parquet"]:
            path = intermediate_dir / fname
            if path.exists():
                dim = pd.read_parquet(path)
                for _, r in dim.iterrows():
                    itemid = clean(r.get("itemid", ""))
                    label = clean(r.get("label", ""))
                    if itemid and label:
                        label_map[itemid] = label

        # pour chaque code MEDS, la 2ème partie après // est l'itemid
        # ex: "CHARTEVENT//001C_102//회/min" → itemid = "001C_102"
        def find_description(code):
            parts = code.split("//")
            if len(parts) >= 2:
                return label_map.get(parts[1])
            return None

        df["description"] = df["code"].apply(find_description)

    return df


def build_subject_splits(subject_ids, train_frac=0.8, tuning_frac=0.1):
    """
    Assigne chaque patient à un split train/tuning/held_out.
    random.seed(42) garantit que les splits sont reproductibles.
    """
    import random
    random.seed(42)
    ids = sorted(subject_ids)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_tuning = int(n * tuning_frac)

    splits = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            splits[sid] = "train"
        elif i < n_train + n_tuning:
            splits[sid] = "tuning"
        else:
            splits[sid] = "held_out"

    return pd.DataFrame([{"subject_id": sid, "split": s} for sid, s in splits.items()])


def build_dataset_json(output_dir, dataset_name, dataset_version):
    meta = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "etl_name": "kmimic-meds",
        "etl_version": "0.1.0",
        "meds_version": "0.3.3",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata" / "dataset.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Conversion vers PyArrow avec le schéma MEDS strict
# ---------------------------------------------------------------------------

def to_meds_table(df):
    """
    Convertit un DataFrame pandas en table PyArrow conforme au schéma MEDS.
    Trie par subject_id puis time (NaT en premier pour les événements statiques).
    """
    if df.empty:
        df = pd.DataFrame(columns=["subject_id", "time", "code", "numeric_value"])

    df = df.copy()
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["code"] = df["code"].astype(str)
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce").astype("float32")

    # NaT en premier = événements statiques en tête du dossier patient
    df = df.sort_values(["subject_id", "time"], na_position="first").reset_index(drop=True)

    return pa.Table.from_pandas(df, schema=MEDS_SCHEMA, safe=False)


def write_parquet(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "syn_patients":       extract_patients,
    "syn_admissions":     extract_admissions,
    "syn_icustays":       extract_icustays,
    "syn_chartevents":    extract_chartevents,
    "syn_labevents":      extract_labevents,
    "syn_diagnoses_icd":  extract_diagnoses_icd,
    "syn_procedures_icd": extract_procedures_icd,
    "syn_inputevents":    extract_inputevents,
    "syn_outputevents":   extract_outputevents,
    "syn_emar":           extract_emar,
}


def run(intermediate_dir: Path, output_dir: Path, dataset_name: str, dataset_version: str):
    print("Loading intermediate Parquet files...")
    all_events = []

    for name, extractor in EXTRACTORS.items():
        path = intermediate_dir / f"{name}.parquet"
        if not path.exists():
            print(f"  WARNING: {name}.parquet not found, skipping.")
            continue
        print(f"  extracting events from {name}.parquet...")
        df = pd.read_parquet(path)
        events = extractor(df)
        if not events.empty:
            all_events.append(events)
            print(f"    -> {len(events)} events")

    print("Merging all events...")
    events_df = pd.concat(all_events, ignore_index=True)
    print(f"  total: {len(events_df)} events, {events_df['subject_id'].nunique()} subjects")

    print("Building subject splits...")
    subject_ids = events_df["subject_id"].dropna().unique().astype(int).tolist()
    splits_df = build_subject_splits(subject_ids)

    print("Writing MEDS data files...")
    for split_name in ["train", "tuning", "held_out"]:
        split_ids = set(splits_df[splits_df["split"] == split_name]["subject_id"])
        split_events = events_df[events_df["subject_id"].isin(split_ids)]
        table = to_meds_table(split_events)
        out_path = output_dir / "data" / split_name / "0.parquet"
        write_parquet(table, out_path)
        print(f"  {split_name}/0.parquet — {len(split_events)} events, {len(split_ids)} subjects")

    print("Writing metadata...")
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # codes.parquet enrichi avec les descriptions
    codes_df = build_codes_parquet(events_df, intermediate_dir)
    codes_table = pa.Table.from_pandas(codes_df, schema=CODES_SCHEMA, safe=False)
    write_parquet(codes_table, meta_dir / "codes.parquet")
    print(f"  codes.parquet — {len(codes_df)} unique codes")

    # subject_splits.parquet
    splits_table = pa.Table.from_pandas(splits_df, schema=SPLITS_SCHEMA, safe=False)
    write_parquet(splits_table, meta_dir / "subject_splits.parquet")
    print(f"  subject_splits.parquet — {len(splits_df)} subjects")

    # dataset.json
    build_dataset_json(output_dir, dataset_name, dataset_version)
    print("  dataset.json")

    print("MEDS conversion done.")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Standalone MEDS conversion for K-MIMIC SYN-ICU")
    parser.add_argument("--intermediate_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default="K-MIMIC-MEDS")
    parser.add_argument("--dataset_version", type=str, default="0.1.0")
    args = parser.parse_args()
    run(args.intermediate_dir, args.output_dir, args.dataset_name, args.dataset_version)


if __name__ == "__main__":
    main()