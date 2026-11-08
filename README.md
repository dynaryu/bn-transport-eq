# Quakenet

## 0. What it does

Quakenet supports **post-earthquake inspection prioritisation of transport networks**.
It uses a Bayesian network to connect earthquake shaking, bridge damage, road
availability and access to key destinations, then prioritises bridge inspections
according to their expected benefit.

The browser-based GUI lets you navigate the transport network, Bayesian network
and clique tree; select a decision task; add observations; and compare results
across analysis rounds in the Roadmap and BN tabs. The main example is the
Eastern Massachusetts highway benchmark network.

## 1. Quick setup

Install **Git** and **Miniconda** (or Anaconda). On Windows, run the following
commands in Anaconda Prompt; on macOS or Linux, use a terminal with Conda available.

Download the app and its RSR inference dependency:

```sh
git clone https://github.com/jieunbyun/bn-transport-eq.git
git clone https://github.com/jieunbyun/rsr.git
cd bn-transport-eq
```

Create the Python environment and start the app:

```sh
conda env create -f environment.yml
conda activate quakenet
python -m pip install -e ../rsr
python -m src serve --open --example ema-highway/gmm
```

The app opens at **http://127.0.0.1:8765**. Keep the terminal open while using it.
Run a baseline, select the decision task, then add observations and run another
analysis. Use dropdown observations to try the app without configuring an LLM.

To start it again later, open a terminal in the `bn-transport-eq` folder and run:

```sh
conda activate quakenet
python -m src serve --open
```

Press **Ctrl+C** in the terminal to stop the server. If port 8765 is already in
use, add `--port 8766` to the start command and open http://127.0.0.1:8766.

## 2. Tutorial video

[**Watch the tutorial: Risk modelling of bridge networks**](https://jieunbyun.github.io/research/bridge-networks/)

Five short clips cover the system and Bayesian network, clique-tree inference,
decision tasks and evidence, reading results, and customising `network.json`.
