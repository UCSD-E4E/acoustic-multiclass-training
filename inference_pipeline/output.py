import sqlite3
from datetime import datetime
from pathlib import Path

from inference_pipeline.args import weights, LABELER_NAME, TYPE, chunk_length
from inference import prob_to_int


def get_or_create_model(cur) -> int:
    cur.execute("SELECT modelId FROM Model WHERE url = ?", (str(weights),))
    row = cur.fetchone()
    if row: 
        return row[0]
    cur.execute("INSERT INTO Model(name, type, url) VALUES (?, ?, ?)", (LABELER_NAME,TYPE,str(weights)))
    return cur.lastrowid

def get_or_create_labeler(cur, modelId: int) -> int:
    cur.execute("SELECT labelerId FROM Labeler WHERE modelId = ?", (modelId,))
    row = cur.fetchone()
    if row: 
        return row[0]
    cur.execute("INSERT INTO Labeler(name, isHuman, modelId) VALUES (?, 0, ?)", (LABELER_NAME,modelId,))
    return cur.lastrowid

def get_or_create_species(cur, code: str) -> int:
    cur.execute(
        "SELECT speciesId FROM Species WHERE species = ? OR common = ?",
        (code, code),
    )
    row = cur.fetchone()
    if row: 
        return row[0]
    cur.execute("INSERT INTO Species(species, common) VALUES (?, ?)", (code, code))
    return cur.lastrowid

def upsert_roi(cur, recording_id: int, start_s: float, end_s: float) -> int:
    cur.execute("""
        INSERT OR IGNORE INTO RegionOfInterest(recordingId, starttime, endtime)
        VALUES (?, ?, ?)
    """, (recording_id, float(start_s), float(end_s)))
    cur.execute("""
        SELECT regionId FROM RegionOfInterest
        WHERE recordingId = ? AND starttime = ? AND endtime = ?
    """, (recording_id, float(start_s), float(end_s)))
    return cur.fetchone()[0]

def save_results_to_sqlite(results, conn, rec_id_by_filename):
    cur = conn.cursor()
    model_id= get_or_create_model(cur)
    labeler_id = get_or_create_labeler(cur, model_id)
    annotations = []

    for _, row in results.iterrows():
        hits = row["_hits"]
        if not hits:
            continue
        file_name = row["FILE NAME"]
        recording_id = rec_id_by_filename.get(Path(file_name).name)
        if not recording_id:
            print(f"[WARN] no recordingId for {file_name}, skipping")
            continue

        start_s = float(row["OFFSET"])
        duration_s = float(row.get("DURATION", chunk_length))
        end_s = start_s + duration_s

        region_id = upsert_roi(cur, recording_id, start_s, end_s)
        # print(f"region_id is {region_id}")

        for h in hits:
            species_id = get_or_create_species(cur, h["species"])
            annotations.append((
                "UNVERIFIED",                 
                region_id,
                labeler_id,
                datetime.utcnow().isoformat(timespec="seconds"),
                species_id,
                prob_to_int(h["confidence"]),
                1                              # mostRecent = TRUE
            ))
    if annotations:
        cur.executemany("""
            INSERT INTO Annotation
              (verified, regionId, labelerId, annotationDate, speciesId, speciesProbability, mostRecent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, annotations)
    conn.commit()