import sqlite3
from pathlib import Path

from args import DB_PATH, RECORDING_IDS, classes, cfg
from inference import run_inference_on_filenames
from output import save_results_to_sqlite

def main():
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

if __name__ == "__main__":
    main()
