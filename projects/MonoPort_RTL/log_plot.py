import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', nargs="*")
parser.add_argument('-o', '--output', default=None, type=str)
args = parser.parse_args()

import matplotlib
if args.output:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def log_paser(file):
    """
    example line:
        04-16 21:24:16 Name: example | Epoch: 0 | 17760/21870 | Err: 0.102818 | LR: 0.001000 | Sigma: 5.00 | dataT: 0.00924 | netT: 0.95690 | ETA: 68:25
    return:
        list of dict
    """
    with open(file, "r") as f:
        lines = f.readlines()

    start_line = 0
    for idx, l in enumerate(lines):
        if "Epoch: 00(00000/" in l:
            start_line = idx

    data = []
    for idx, l in enumerate(lines):
        if idx < start_line:
            continue
        if "Epoch:" not in l:
            continue
        data.append({
            "name": l.split("|")[0].split("Name:")[-1].replace(" ", ""),
            "epoch": int(l.split("|")[1].split("Epoch: ")[-1].split("(")[0].replace(" ", "")),
            "iter": int(l.split("|")[1].split("Epoch: ")[-1].split("(")[1].split("/")[0].replace(" ", "")),
            "niter": int(l.split("|")[1].split("Epoch: ")[-1].split("(")[1].split("/")[1].replace(")", "")),
            "err": float(l.split("|")[6].split("Err:")[1].replace(" ", "")),
        })
    return data

colors = list(mcolors.TABLEAU_COLORS.keys())

for idx, f in enumerate(args.file):
    color = colors[idx]
    lines = log_paser(f)

    X = [l["epoch"] + l["iter"] / l["niter"] for l in lines]
    Y = [l["err"] for l in lines]

    plt.plot(X, Y, color=color, label=lines[0]["name"], alpha=0.7)

plt.legend()

if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()
