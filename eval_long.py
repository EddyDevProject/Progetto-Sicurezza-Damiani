# eval_long.py
# Valuta un modello GIA' ALLENATO sui puzzle lunghi (n = 6..10):
#     python eval_long.py M4 0   (variante, seed)
#
# Non riallena nulla: carica il transformer dalla cache dei grafi e i pesi da best.pt.
# Come train.py va lanciato in un processo separato (problema GATConv / TimeAwareGATConv).
# Attenzione: i modelli sono allenati su n = 1..5, quindi qui sono fuori distribuzione.

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
from torch_geometric.data import Batch

from utils import *

VARIANTS = {
    "M1": ("gat_basic", False),
    "M2": ("gat_basic", True),
    "M3": ("gat_time_decay", False),
    "M4": ("gat_time_decay", True),
    "M4c": ("gat_time_decay", True),
}

variant, seed = sys.argv[1], int(sys.argv[2])
cfg = json.load(open("config.json"))
mode, use_time = VARIANTS[variant]
data, out = cfg["data_dir"], f"{cfg['runs_dir']}/{variant}_s{seed}"
device = cfg["device"]
const_time = variant == "M4c"
T_CONST = pd.read_parquet(f"{data}/train.parquet", columns=["t"])["t"].mean()

# stesso transformer usato in training (salvato nella cache dei grafi da train.py)
cache = f"{data}/graphs_{mode}_{use_time}{'_const' if const_time else ''}.pt"
tr, train_g, _ = torch.load(cache, weights_only=False)
n_features = train_g[0][0].x.shape[1]
del train_g

# stesso modello di train.py, con i pesi migliori
params = dict(num_event_features=n_features, num_embedding_features=len(CLASS_ID),
              output_dim=len(CLASS_ID), embedding_dims=64, gat_hidden_dim_event=64,
              gat_hidden_dim_embed=128, gat_hidden_dim_concat=128, num_heads=4,
              num_layers=2, dropout=0.1, use_batch_norm=True)
if mode == "gat_basic":
    from timegnn.models.gat_basic import DualGATModel
    model = DualGATModel(**params)
else:
    from timegnn.models.gat_time_decay import DualGATTimeAwareModel
    model = DualGATTimeAwareModel(lambda_decay=cfg["lambda_decay"], **params)
model.load_state_dict(torch.load(f"{out}/best.pt", map_location=device))
model = model.to(device)
model.eval()


def choose(p, board, played):
    times = [think_time(p["rating"])] * len(played)
    df = pd.DataFrame(puzzle_rows("x", p["fen"], played, times, p["rating"], p["n"]))
    if const_time:
        df["t"] = T_CONST
    g, _ = make_graphs(tr, df, mode, use_time)[0]
    with torch.no_grad():
        logits = model(Batch.from_data_list([g]).to(device))[-1].cpu()
    legal = [m.uci() for m in board.legal_moves]
    return legal[int(logits[[CLASS_ID[m] for m in legal]].argmax())]


engine = chess.engine.SimpleEngine.popen_uci(cfg["stockfish"]) if cfg["stockfish"] else None
test = pd.read_parquet(f"{data}/test_long_puzzles.parquet")
results = []
for p in test.to_dict("records"):
    t0 = time.time()
    status, first_ok, played = solve(p, choose, engine)
    results.append({"system": variant, "seed": seed, "case": p["case"], "n": p["n"],
                    "status": status, "solved": status == "solved", "first_ok": first_ok,
                    "moves": " ".join(played), "sec": time.time() - t0})
if engine:
    engine.quit()

res = pd.DataFrame(results)
res.to_csv(f"{out}/results_long.csv", index=False)
print(f"{variant} s{seed}:", res.groupby("n")["solved"].mean().round(3).to_dict())
