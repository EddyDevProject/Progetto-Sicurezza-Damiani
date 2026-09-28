# utils.py
# Funzioni usate sia dal notebook che da train.py

import chess
import chess.engine
import numpy as np
import pandas as pd
import torch

SQUARES = [f"sq{i}" for i in range(64)]
T0 = pd.Timestamp("2020-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# Vocabolario: tutte le mosse UCI possibili (1968). Lo fissiamo a priori cosi'
# l'encoder della libreria non trova mai mosse sconosciute in validation/test.
def all_uci_moves():
    moves = set()
    for a in range(64):
        for b in range(64):
            dx, dy = abs(a % 8 - b % 8), abs(a // 8 - b // 8)
            if a != b and (dx == 0 or dy == 0 or dx == dy or dx * dy == 2):
                moves.add(chess.square_name(a) + chess.square_name(b))
    for f in range(8):                      # promozioni
        for d in (-1, 0, 1):
            if 0 <= f + d < 8:
                for p in "qrbn":
                    moves.add(chess.square_name(48 + f) + chess.square_name(56 + f + d) + p)
                    moves.add(chess.square_name(8 + f) + chess.square_name(f + d) + p)
    return sorted(moves)


MOVES = all_uci_moves()
# il LabelEncoder della libreria ordina le classi alfabeticamente (EOS compreso)
CLASS_ID = {m: i for i, m in enumerate(sorted(MOVES + ["EOS"]))}


# ---------------------------------------------------------------------------
# Forma canonica: il risolutore gioca sempre col Bianco.
# Nel CSV Lichess al tratto c'e' l'avversario (la sua mossa e' la prima di Moves),
# quindi se al tratto c'e' il Bianco il risolutore e' il Nero e specchiamo tutto.
def mirror(uci):
    m = chess.Move.from_uci(uci)
    return chess.Move(chess.square_mirror(m.from_square),
                      chess.square_mirror(m.to_square), m.promotion).uci()


def canonical(fen, moves):
    board = chess.Board(fen)
    flip = board.turn == chess.WHITE
    if flip:
        board = board.mirror()
        moves = [mirror(m) for m in moves]
    return board.fen(), moves, flip


# ---------------------------------------------------------------------------
# Tempo simulato: i puzzle non hanno il clock. Come da traccia lo simuliamo
# dal rating (puzzle piu' difficili = si pensa di piu'), con un po' di rumore.
def think_time(rating, rng=None):
    mean = 4 * np.exp((rating - 1500) / 1000)
    if rng is None:
        return mean                           # in inferenza usiamo la media
    return float(np.clip(rng.lognormal(np.log(mean), 0.6), 0.5, 600))


# ---------------------------------------------------------------------------
# Un puzzle -> righe dell'event log (una riga = un nodo = una mossa).
# Le feature sono la scacchiera DOPO la mossa: la libreria mette sul nodo i
# la label "mossa i+1", quindi il nodo contiene la posizione da risolvere.
def puzzle_rows(pid, fen, moves, times, rating, n):
    board = chess.Board(fen)
    rows, t, solver_moves = [], 0.0, 0
    for i, m in enumerate(moves):
        board.push_uci(m)
        if i % 2 == 1:                        # mosse del risolutore: 1, 3, 5, ...
            solver_moves += 1
        t += times[i]
        row = {
            "case": pid, "ply": i, "move": m,
            "ts": T0 + pd.Timedelta(seconds=t),
            "t": min(np.log1p(times[i]) / np.log1p(600), 1.0),   # tempo scalato in [0,1]
            "check": str(int(board.is_check())),
            "mate": str(int(board.is_checkmate())),
            "left": n - solver_moves,          # mosse ancora disponibili (noto anche in inferenza)
            "rating": rating, "n": n,
        }
        for s in range(64):
            p = board.piece_at(s)
            row[SQUARES[s]] = p.symbol() if p else "."
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# TimeGNN
def make_transformer(mode, use_time):
    from timegnn import EventLogTransformer
    num = ["left", "t"] if use_time else ["left"]
    return EventLogTransformer(case_col="case", event_col="move", time_col="ts", mode=mode,
                               cat_event=SQUARES + ["check", "mate"], num_event=num,
                               seq_cols=["rating", "n"], num_seq=["rating", "n"])


def fit_transformer(tr, train_df):
    # aggiungiamo un caso finto con tutte le 1968 mosse: cosi' l'encoder conosce
    # tutto il vocabolario. I valori -1 sono trattati dalla libreria come "mancanti".
    fake = pd.DataFrame({"case": "zzz_vocab", "ply": range(len(MOVES)), "move": MOVES,
                         "ts": T0, "check": "0", "mate": "0", "left": -1, "t": -1.0,
                         "rating": -1.0, "n": -1})
    for c in SQUARES:
        fake[c] = "."
    tr.fit(pd.concat([train_df, fake], ignore_index=True))
    assert list(tr._event_encoder.classes_) == list(CLASS_ID)
    return tr


def make_graphs(tr, df, mode, use_time):
    """event log -> lista di (grafo, label). Due correzioni rispetto ai grafi della libreria:
    1) tempo: sull'arco i -> i+1 mettiamo il tempo speso sulla mossa i+1 (la libreria usa
       il tempo che manca alla fine del puzzle, che in inferenza non si conosce)
    2) self-loop per il modello time-decay: senza, un nodo non vede le proprie feature
    """
    df = df.sort_values(["case", "ply"]).reset_index(drop=True)
    cases = df["case"].unique()
    out_graphs = []
    for part in np.array_split(cases, max(1, len(cases) // 5000)):   # a blocchi per la RAM
        sub = df[df["case"].isin(part)].reset_index(drop=True)
        out = tr.transform(sub)
        times = sub.groupby("case")["t"].apply(list)                   # stesso ordine della libreria
        for g, y, t in zip(out.event_features, out.labels, times):
            n = g.num_nodes
            edges = g.edge_index.view(2, -1)
            et = torch.tensor(t[1:], dtype=torch.float) if use_time else torch.zeros(n - 1)
            if mode == "gat_basic":
                g.edge_index, g.edge_attr = edges, et.view(-1, 1)
            else:
                loops = torch.arange(n).repeat(2, 1)
                g.edge_index = torch.cat([edges, loops], dim=1)
                g.time = torch.cat([et, torch.zeros(n)])
            out_graphs.append((g, y))
    return out_graphs


# ---------------------------------------------------------------------------
# Risolvere un puzzle. choose(p, board, played) restituisce la mossa del risolutore.
# Il difensore segue la soluzione; se abbiamo deviato risponde Stockfish.
# Senza Stockfish una deviazione che non da' matto finisce come "unknown".
def solve(p, choose, engine=None):
    sol = p["moves"].split()
    board = chess.Board(p["fen"])
    board.push_uci(sol[0])
    played, on_line, first_ok = [sol[0]], True, None
    for k in range(p["n"]):
        m = choose(p, board, played)
        if first_ok is None:
            first_ok = m == sol[1]
        on_line = on_line and m == sol[2 * k + 1]
        board.push_uci(m)
        played.append(m)
        if board.is_checkmate():
            return "solved", first_ok, played
        if k == p["n"] - 1 or board.is_game_over():
            break
        if on_line:
            reply = sol[2 * k + 2]
        elif engine is not None:
            reply = engine.play(board, chess.engine.Limit(time=0.05)).move.uci()
        else:
            return "unknown", first_ok, played
        board.push_uci(reply)
        played.append(reply)
        if board.is_game_over():              # es. la risposta di Stockfish da' matto a noi
            break
    return "failed", first_ok, played