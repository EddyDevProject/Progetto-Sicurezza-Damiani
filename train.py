# train.py
# Allena e valuta una variante:   python train.py M4 0   (variante, seed)
#
# Lo lanciamo dal notebook come processo separato perche' con PyG >= 2.5, se nello
# stesso processo si crea un GATConv (gat_basic), il TimeAwareGATConv (gat_time_decay)
# da' errore: "propagate() got an unexpected keyword argument 'edge_attr'".

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from timegnn.data.pyg import CustomDataset, custom_collate_fn
from timegnn.models.training import train_epoch, evaluate_epoch
from timegnn.train.early_stopping import EarlyStopping

from utils import *

VARIANTS = {
    "M1": ("gat_basic", False),        # GAT senza tempo
    "M2": ("gat_basic", True),         # GAT con tempo sugli archi
    "M3": ("gat_time_decay", False),   # time-decay con tempo azzerato
    "M4": ("gat_time_decay", True),    # time-decay con tempo
}

variant, seed = sys.argv[1], int(sys.argv[2])
cfg = json.load(open("config.json"))
mode, use_time = VARIANTS[variant]
data, out = cfg["data_dir"], f"{cfg['runs_dir']}/{variant}_s{seed}"
os.makedirs(out, exist_ok=True)
device = cfg["device"]
torch.manual_seed(seed)
np.random.seed(seed)
print(f"{variant}: mode={mode}, tempo={use_time}, device={device}, seed={seed}")

# ----- grafi (li salviamo su disco: costruirli richiede qualche minuto) -------
cache = f"{data}/graphs_{mode}_{use_time}.pt"
if os.path.exists(cache):
    tr, train_g, val_g = torch.load(cache, weights_only=False)
else:
    train_df = pd.read_parquet(f"{data}/train.parquet")
    tr = fit_transformer(make_transformer(mode, use_time), train_df)
    train_g = make_graphs(tr, train_df, mode, use_time)
    val_g = make_graphs(tr, pd.read_parquet(f"{data}/val.parquet"), mode, use_time)
    torch.save((tr, train_g, val_g), cache)
print(f"grafi: {len(train_g)} train, {len(val_g)} val, {train_g[0][0].x.shape[1]} feature per nodo")


def loader(graphs, shuffle):
    ds = CustomDataset([g for g, _ in graphs], [y for _, y in graphs])
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, collate_fn=custom_collate_fn)


train_loader, val_loader = loader(train_g, True), loader(val_g, False)

# ----- modello della libreria ----------------------------------------------
params = dict(num_event_features=train_g[0][0].x.shape[1], num_embedding_features=len(CLASS_ID),
              output_dim=len(CLASS_ID), embedding_dims=64, gat_hidden_dim_event=64,
              gat_hidden_dim_embed=128, gat_hidden_dim_concat=128, num_heads=4,
              num_layers=2, dropout=0.1, use_batch_norm=True)
if mode == "gat_basic":
    from timegnn.models.gat_basic import DualGATModel
    model = DualGATModel(**params)
else:
    from timegnn.models.gat_time_decay import DualGATTimeAwareModel
    model = DualGATTimeAwareModel(lambda_decay=cfg["lambda_decay"], **params)
model = model.to(device)

# ----- training -------------------------------------------------------------
opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
stopper = EarlyStopping(patience=cfg["patience"])
history = []
for ep in range(1, cfg["epochs"] + 1):
    t0 = time.time()
    tl, ta = train_epoch(model, train_loader, opt, loss_fn, device)
    vl, va = evaluate_epoch(model, val_loader, loss_fn, device)
    history.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
    stop = stopper(vl)
    if stopper.best_loss_updated:
        torch.save(model.state_dict(), f"{out}/best.pt")
    print(f"epoca {ep}: train {tl:.3f} acc {ta:.3f} | val {vl:.3f} acc {va:.3f} | {time.time() - t0:.0f}s",
          flush=True)
    # il time-decay all'inizio resta qualche epoca fermo (predice sempre EOS):
    # per questo non fermiamo mai prima di min_epochs
    if stop and ep >= cfg["min_epochs"]:
        print("early stopping")
        break
pd.DataFrame(history).to_csv(f"{out}/history.csv", index=False)

# ----- valutazione: risolviamo i puzzle di test -----------------------------
model.load_state_dict(torch.load(f"{out}/best.pt"))
model.eval()


def choose(p, board, played):
    # ricostruiamo la sequenza giocata finora e prendiamo l'output dell'ultimo nodo
    times = [think_time(p["rating"])] * len(played)
    df = pd.DataFrame(puzzle_rows("x", p["fen"], played, times, p["rating"], p["n"]))
    g, _ = make_graphs(tr, df, mode, use_time)[0]
    with torch.no_grad():
        logits = model(Batch.from_data_list([g]).to(device))[-1].cpu()
    legal = [m.uci() for m in board.legal_moves]      # scegliamo solo tra le mosse legali
    return legal[int(logits[[CLASS_ID[m] for m in legal]].argmax())]


engine = chess.engine.SimpleEngine.popen_uci(cfg["stockfish"]) if cfg["stockfish"] else None
test = pd.read_parquet(f"{data}/test_puzzles.parquet")
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
res.to_csv(f"{out}/results.csv", index=False)
print(res.groupby("n")[["solved", "first_ok"]].mean().round(3))
