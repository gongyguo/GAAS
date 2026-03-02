# Efficient GPU-Accelerated Adaptive Minimum Cost Seed Selection (GAAS)

GAAS is a C++/CUDA implementation of an adaptive algorithm for minimum cost seed selection. It utilizes GPU acceleration via CUDA to generate and manipulate randomized mRR-sets and applies under the Independent Cascade (IC) and Linear Threshold (LT) models.

## 📁 Repository Structure

```
GAAS/
├── argument.h           # argument parsing and definitions
├── command.sh           # example run script
├── GAAS.cu              # main CUDA implementation of GAAS class
├── GAAS.h               # class declarations
├── utils.cpp            # helper utilities
├── dataset/             # input graph files
│   ├── DBLP
│   ├── DBLP_cost_001DEG.txt
│   └── ...
├── realization/         # precomputed realizations
│   ├── DBLP_pw_ic0.txt
│   ├── ...
├── Makefile             # build instructions
└── README.md          
```

## 📐 Preparation

Graphs should be stored and read from text files in the `dataset/` directory. The first two integers specify the number of nodes and edges. Each subsequent line contains a source node and destination node pair (0‑indexed). Social networks including datasets [DBLP](https://snap.stanford.edu/data/com-DBLP.html), [YouTube](https://snap.stanford.edu/data/com-Youtube.html), [Flickr](http://konect.cc/networks/flickr-growth/), and [LiveJournal](https://snap.stanford.edu/data/soc-LiveJournal1.html) used in experiments.

A corresponding cost file named `<dataset>_cost_001DEG.txt` must be present; the file contains one cost value per node.

Realization files in `realization/` are used for diffusion under different models. Filenames follow the pattern `<dataset>_pw_icX.txt` or `<dataset>_pw_ltY.txt` where `X`/`Y` are realization indices.



## 🚀 Building the Project

The code requires an NVIDIA GPU with CUDA 11.6 with `nvcc` and `cub` library installed (tested with compute capability 8.6). To build:

```bash
make all
```

## 🧠 Usage

The `gaas` executable takes command–line arguments that specify the dataset, model, batch size, epsilon, and other options. Example usage:

```bash
./gaas -dataset_No <dataset> -eps <epsilon> -model <IC_or_LT> ...
```

* `-dataset_No`: path under `dataset/` (e.g., `0` for DBLP, `1` for YouTube, `2`for Flickr, and `3` for LiveJournal)
* `-eps`: estimation error of mRR-sets (e.g., `0.5`)
* `-model`: `1` for IC model, `0` for LT model
* `-eta`: target influence threshold (e.g., `2000`)
* `-batch`：batch size b (e.g., `4`)
* `-start_time`: the index of used random realization (e.g., `0`)
* `-reuse` : configure for activating reuse of mRR-sets (e.g., `1`) or regenerate mRR-sets in each round as used in experimental analysis (e.g., `0`)
* `-beta` : configure for work load balance in mRR-set update (e.g., `8`)



## 🛠 Development Notes

* Core logic resides in `GAAS.cu` and `GAAS.h`.
* `utils.cpp` contains ancillary functions (e.g., generating cost file and random realizations).
* CUDA kernels for mRR set generation, update, and seed selection are defined in the `kernel.cu` and `kernel.h`.

