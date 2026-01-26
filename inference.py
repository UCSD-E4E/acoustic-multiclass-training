#THINGS THAT HAVE TO GET DONE
# 1. IMMEDIATE: Implement some sort of way to filter which recordings u run inference over b/c rn, it is getting all recordings from recordings database. This is something that one of the students can do.
# 2. right now, in inference, it reads file, creates pt file in same directory and then goes over that pt file. try to make df instead so that we dont make new pt file
# 3. Remove manual_id and generally clean up code. Even cfg file isnt super necessary. 
# 4. Maybe see where you load the model below and see if u could only load it once instead of loading every data path? Look at the commonet below (where I load the model) to see why it is not an easy fix. 


from pathlib import Path
import sys
#do this before importing anything else so that we can get command line args
def pop_flag_and_value(flag: str):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            print(f"Missing value for {flag}", file=sys.stderr)
            sys.exit(2)
        val = sys.argv[i + 1]
        del sys.argv[i:i+2]  
        return val
    return None
_db = pop_flag_and_value("--db-path")
if not _db:
    print("Error: provide --db-path", file=sys.stderr)
    sys.exit(2)
#When called by desktop app, contians absolute path to db
DB_PATH = Path(_db).expanduser().resolve()
_recording_ids = pop_flag_and_value("--recording-ids")
RECORDING_IDS = _recording_ids.split(",") if _recording_ids else None



import pandas as pd
from pyha_analyzer import config
from pyha_analyzer.dataset import PyhaDFDataset
from pyha_analyzer.models.timm_model import TimmModel
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from mutagen import File
import sqlite3
from datetime import datetime
cfg = config.cfg



#CONSTANTS
#maybe threshold should be specified by user?
threshold = 0.7
chunk_length = 5  # seconds
classes = [
        'amabaw1', 'amapyo1', 'astgna1', 'baffal1', 'barant1', 'bartin2', 'batman1', 'blacar1', 'blbthr1', 'blcbec1', 'blctro1', 'blfant1', 'blfcot1', 'blfjac1', 'blfnun1', 'blgdov1', 'blhpar1', 'bltant2', 'blttro1', 'bobfly1', 'brratt1', 'bsbeye1', 'btfgle1', 'bubgro2', 'bubwre1', 'bucmot4', 'buffal1', 'butsal1', 'butwoo1', 'chwfog1', 'cinmou1', 'cintin1', 'citwoo1', 'coffal1', 'coltro1', 'compot1', 'cowpar1', 'crfgle1', 'ducatt1', 'ducfly', 'ducgre1', 'duhpar', 'dutant2', 'elewoo1', 'eulfly1', 'fasant1', 'fepowl', 'forela1', 'garkin1', 'gilbar1', 'gnbtro1', 'gocspa1', 'goeant1', 'gogwoo1', 'gramou1', 'grasal3', 'grcfly1', 'greant1', 'greibi1', 'gretin1', 'grfdov1', 'gryant1', 'gryant2', 'gycfly1', 'gycwor1', 'hauthr1', 'horscr1', 'letbar1', 'littin1', 'litwoo2', 'lobwoo1', 'lowant1', 'meapar', 'muswre2', 'olioro1', 'oliwoo1', 'partan1', 'pavpig2', 'pirfly1', 'plbwoo1', 'pltant1', 'pluant1', 'plupig2', 'plwant1', 'puteup1', 'putfru1', 'pygant1', 'rcatan1', 'rebmac2', 'renwoo1', 'rinant2', 'rinkin1', 'rinwoo1', 'royfly1', 'ruboro1', 'rucant2', 'rudpig', 'rufant3', 'ruftof1', 'ruqdov', 'scapig2', 'scbwoo5', 'scrpih1', 'sobcac1', 'specha3', 'spigua1', 'spwant2', 'squcuc1', 'stbwoo2', 'strcuc1', 'strwoo2', 'strxen1', 'stwqua1', 'tabsco1', 'thlwre1', 'undtin1', 'viotro3', 'wespuf1', 'whbtot1', 'whcspa1', 'whfant2', 'whltyr1', 'whnrob1', 'whrsir1', 'whttou1', 'whtwoo2', 'whwbec1', 'wibpip1', 'yectyr1', 'yemfly1', 'yercac1', 'yetwoo2'
        ]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  
#TODO: Change this to the path where model weights are stored if we ever actually have multiple models
weights = PROJECT_ROOT / "pyfiles" / "acoustic-multiclass-training" / "models" / "eca_nfnet_l0-20240711-0531.pt"
LABELER_NAME = "model_eca_nfnet_l0-20240711-0531"
TYPE= "eca_nfnet"

def prob_to_int(p: float) -> int:
    return int(round(100 * float(p)))  # Annotation.speciesProbability saved as integer

def row_hits(row, classes):
    above = row[classes][row[classes] > threshold]
    if above.empty: return []
    off = row["OFFSET"]
    return [{"offset": int(off), "species": col, "confidence": float(val)}
            for col, val in above.items()]

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

def run_inference_on_filenames(base_dir: Path, 
                               file_rows, #file_rows: list of rows with keys 'recordingId' and 'filename'
                               classes, 
                               cfg):
    cfg.data_path = base_dir 
    rows = []
    audio_file_paths = []
    for r in file_rows:
        #filename = r["filename"]
        filename= Path(r["url"])
        full_path = base_dir / filename
        filename = str(filename)
        # print("full path is ", full_path)
        #full_path = r["url"]
        if not full_path.exists():
            print(f"[WARN] missing file: {full_path}")
            continue
        try:
            audio = File(full_path) 
        except Exception as e:
            print(f"Error reading {str(full_path)}: {e}")
            continue

        num_chunks = int(audio.info.length // chunk_length)
        for i in range(num_chunks):
            offset = i * chunk_length
            rows.append([offset, chunk_length, "amabaw1", filename])  # MANUAL ID kept as of now but eventaully clean up code to remove this as mandatory
        audio_file_paths.append(filename)
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=["OFFSET", "DURATION", "MANUAL ID", "FILE NAME"])

    infer_dataset = PyhaDFDataset(df, train=False, species=classes, cfg=cfg)
    loader = DataLoader(infer_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=0)
    
    #Load model
    # TODO Eventually: see if u can run load the model outside loop so that it only loads once. issue is that you need to have cfg.data_path and that changes per directory. u also need to have something in df so it (to me) seems like it needs to be there
    train_dataset = PyhaDFDataset(df, train=True, species=classes, cfg=cfg)
    model = TimmModel(num_classes=len(classes), model_name=cfg.model).to(cfg.device)
    model.create_loss_fn(train_dataset)
    checkpoint = torch.load(weights, map_location=cfg.device)
    model.load_state_dict(checkpoint)
    model.to(cfg.device)

    # Inference
    log_pred = []
    model.eval()
    with torch.no_grad():
        for mels, labels in loader:
            mels = mels.to(cfg.device)
            labels = labels.to(cfg.device)
            outputs = model(mels)
            outputs = outputs.to(dtype=torch.float32)
            log_pred.append(np.array(torch.clone(F.sigmoid(outputs).cpu()).detach()))
    log_pred = np.concatenate(log_pred)


    results = pd.DataFrame(log_pred, columns=infer_dataset.classes)
    results = pd.concat([infer_dataset.samples.reset_index(drop=True), results], axis=1)
    results["_hits"] = results.apply(lambda r: row_hits(r, infer_dataset.classes), axis=1)
    if "original_file_path" in results.columns:
        results["PT FILE NAME"] = results["FILE NAME"]
        results["FILE NAME"] = results["original_file_path"].fillna(results["FILE NAME"])
    return results


with sqlite3.connect(str(DB_PATH)) as conn:
    conn.row_factory = sqlite3.Row
    # Indexes for speed
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recording_directory ON Recording(directory)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_roi_unique ON RegionOfInterest(recordingId, starttime, endtime)")

    # select directories in RECORDING_IDS if provided, else get all directories
    if RECORDING_IDS:
        placeholders = ",".join("?" * len(RECORDING_IDS))
        dirs = [r["directory"] for r in conn.execute(f"""
            SELECT DISTINCT directory
            FROM Recording
            WHERE directory IS NOT NULL
              AND recordingId IN ({placeholders})
            ORDER BY directory
        """, RECORDING_IDS).fetchall()]
    else:
         # Get directories (get the ones with most files first)
        dirs = [r["directory"] for r in conn.execute("""
            SELECT directory
            FROM Recording
            WHERE directory IS NOT NULL
            GROUP BY directory
            ORDER BY COUNT(*) DESC, directory
        """).fetchall()]


#for every unique directory, set the base path and do batched inference on those files
for directory in dirs:
    #base = Path(directory)
    base=directory
    # with sqlite3.connect(str(DB_PATH)) as conn:
    #     conn.row_factory = sqlite3.Row
    #     file_rows = conn.execute("""
    #         SELECT recordingId, filename
    #         FROM Recording
    #         WHERE directory = ?
    #         ORDER BY filename
    #     """, (directory,)).fetchall()

    #     if not file_rows:
    #         continue

    #     # filename -> recordingId map for this directory
    #     # rec_id_by_filename = {r["filename"]: r["recordingId"] for r in file_rows if r["filename"]}
    #     #replace with pt extension so that it works later on when mapping recordings to its id
    #     rec_id_by_filename = {
    #         Path(r["filename"]).with_suffix(".pt").name: r["recordingId"]
    #         for r in file_rows
    #         if r["filename"]
    #     }
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        if RECORDING_IDS:
            placeholders = ",".join("?" * len(RECORDING_IDS))
            file_rows = conn.execute(f"""
                SELECT recordingId, url
                FROM Recording
                WHERE directory = ?
                  AND recordingId IN ({placeholders})
                ORDER BY url
            """, (directory, *RECORDING_IDS)).fetchall()
        else:
            file_rows = conn.execute("""
                SELECT recordingId, url
                FROM Recording
                WHERE directory = ?
                ORDER BY url
            """, (directory,)).fetchall()

        if not file_rows:
            continue

        # Extract filename from the end of the URL ;;;;; NO LONGER DOING THIS: and replace its suffix with .pt
        rec_id_by_filename = {
            Path(r["url"]).name: r["recordingId"]
            for r in file_rows
            if r["url"]
        }


        #print("base is ", base)

        results = run_inference_on_filenames(base, file_rows, classes, cfg)
        if results is None:
            print(f"[INFO] No valid audio in {directory}")
            continue

        save_results_to_sqlite(results, conn, rec_id_by_filename)
        print(f"[OK] processed {directory}")
