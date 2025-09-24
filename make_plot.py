import uproot
from particle import Particle
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from collections import Counter
# import hist
import argparse
from pathlib import Path

import json5

def main(cfg):
    print(f"Reading {cfg['files']}")
    print(f"Output to {cfg['outdir']}")
    Path(cfg['outdir']).mkdir(parents=True, exist_ok=True)
    for block in cfg['blocks']:
        print("Doing", block['suptitle'])
        plot_for_tree(cfg['files'], cfg['outdir'], block)


def plot_for_tree(fname, outdir, cfg):
    df = uproot.concatenate(f"{fname}:{cfg['tree']}", library='pd', num_workers=4)
    # Fix PDG: 
    if 'pdg' in df.columns:
        df['pdg'] = np.where(df['pdg'] > 1000000000, df['pdg'] // 10 * 10, df['pdg'])
    plt.figure(figsize=cfg["figsize"])
    plt.suptitle(cfg["suptitle"])
    iplot = 1
    for col, data in df.items():
        plt.subplot(*cfg['grid'], iplot)
        dtype = type(data[0])
        categorical_columns = ['pdg', 'code', 'algorithm', 'ropid', 'view', "TPCSetID", 'detid', 'type']
        if dtype is str or col in categorical_columns:
            data = data.to_list()
            counts = Counter(data)
            names = list(counts.keys())
            heights = list(counts.values())
            if "pdg" in col:
                names = [f"${Particle.from_pdgid(pdgcode).latex_name}$" for pdgcode in names]
            plt.bar(names, heights)

        else:
            hist_data = np.histogram(data.to_numpy(), bins=100)
            hep.histplot(hist_data)
        plt.title(col)
        plt.semilogy()
        plt.ylabel("Counts")
        if "labels" in cfg and col in cfg['labels']:
            plt.xlabel(cfg['labels'][col])
        iplot += 1
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{outdir}/{cfg['image_prefix']}.png")

    if "kinetic_energy" in cfg and cfg['kinetic_energy']:
        plt.figure()
        pdgs = df['pdg'].to_numpy()
        masses = [Particle.from_pdgid(pdg).mass for pdg in pdgs]
        masses = [masses[i] if masses[i] is not None else 0 for i in range(len(masses))]
        masses = np.asarray(masses) / 1000
        kes = df['en'].to_numpy() - masses
        binning = np.histogram(kes, bins=100)[1]
        for pdg in np.unique(pdgs):
            if pdg > 1e9:
                continue
            hist_ke = np.histogram(kes[pdgs==pdg], bins=binning)
            hep.histplot(hist_ke, label=f"${Particle.from_pdgid(pdg).latex_name}$")
        hist_ke_hadron = np.histogram(kes[np.abs(pdgs) > 1e9], bins=binning)
        hep.histplot(hist_ke_hadron, label="Hadrons")
        plt.title(f"{cfg['suptitle']} Kinetic Energy")
        plt.ylabel("Counts")
        plt.xlabel("GeV")
        plt.legend(ncol=4)
        plt.semilogy()
        plt.savefig(f"{outdir}/{cfg['image_prefix']}_KE.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    cfg = json5.load(open(args.config))
    
    # Change space around histograms
    plt.rcParams['axes.xmargin'] = 0
    
    main(cfg)
    if args.show:
        plt.show()
