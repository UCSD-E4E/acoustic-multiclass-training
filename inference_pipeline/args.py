from pathlib import Path
import sys
from pyha_analyzer import config
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

cfg = config.cfg

#CONSTANTS ----------------
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
