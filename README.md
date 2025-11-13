# AI Corpus Pipeline & MiniGPT Dashboard

This repository collects everything needed to build a French AI/tech corpus, train a byte-level Transformer on it, and monitor the run through an interactive dashboard.

The workflow has three pillars:

1. **Data preparation** – convert raw exports (YouTube subtitles, LinkedIn posts, Instagram conversations, Wikipedia articles) into plain text under `data_clean/`.
2. **Tokenizer training** – learn a BPE tokenizer on the unified cleaned corpus so the upstream model vocabulary matches real usage (précis accents, emoji, etc.).
3. **Transformer training** – fine-tune a configurable MiniGPT-like model with live metrics, embedding visualisations and an in-browser chat.

---

## Quick Start Commands

### 1. Backend API (Python)

```bash
# Install dependencies (first run only)
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Start the Flask server with the most recent run (adjust paths if neede

This launches the REST API on http://127.0.0.1:8000 with endpoints such as `/metadata`, `/metrics`, `/train`, `/tokenizer/**` and `/chat`.

### 2. Frontend Dashboard (React + Vite)

```bash
cd dashboard/web
npm install        # first run only
npm run dev        # http://127.0.0.1:5173 (proxy vers le backend)
```

For a production-like build use `npm run build && npm run preview`.

### 3. Optional Training Commands

Need to refresh data or re-train?

```bash
# Préparer les sous-titres YouTube nettoyés
python scripts/clean_vtt.py --source-dir data/youtube-subtitile --target-dir data_clean/youtube

# Entraîner le tokenizer BPE sur l'ensemble des données nettoyées
python scripts/train_tokenizer.py --name wiki-bpe --input-dir data_clean --vocab-size 32000

# Lancer un entraînement MiniGPT (adapter les chemins/paramètres)
python scripts/train_subtitles_transformer.py \
  --arch-preset mini-gpt \
  --data-dir data_clean/youtube \
  --extra-data-dir data_clean/fineweb_fr \
  --max-steps 200 \
  --log-dir trained_models/runs

# Reprendre un run existant en augmentant `max_steps`
python scripts/train_subtitles_transformer.py \
  --resume-from trained_models/runs/<run>/checkpoint.pt \
  --resume-run-dir trained_models/runs/<run> \
  --max-steps 12000
```

Les samples générés pendant l'entraînement sont disponibles dans `trained_models/runs/<run>/samples.txt` et reflétés en direct par la console/back-end.

---

## Repository Layout

```
data/                      # raw exports (YouTube .vtt, LinkedIn CSV, Instagram JSON…)
data_clean/                # cleaned text (one .txt per source) – tokenizer + model read from here
scripts/
  clean_vtt.py             # YouTube VTT → clean text pipeline
  prepare_instagram.py     # normalise Instagram inbox & comments exports
  prepare_linkedin.py      # normalise LinkedIn comments / shares / messages
  prepare_wikipedia.py     # fetch AI/ML Wikipedia articles (once)
  clean_wikipedia.py       # strip LaTeX/templates from downloaded Wikipedia text
  train_tokenizer.py       # train a byte-level BPE tokenizer on data_clean/
  download_fineweb.py      # grab & filter a French slice of FineWeb
  download_hf_datasets.py  # snapshot curated Hugging Face corpora (ouvrages, QA…)
  train_subtitles_transformer.py  # end-to-end training loop for the MiniGPT
  run_subtitle_pipeline.sh # shell helper to chain prep + training
dashboard/
  server.py                # Flask backend serving metrics + control endpoints
  web/                     # React frontend (Vite) for live monitoring & chat
trained_models/
  tiny_subtitles_transformer.pt   # latest consolidated checkpoint
  runs/run-*/              # per-run metadata, metrics.jsonl, checkpoints
```

---

## Data Preparation

| Source       | Raw input                              | Clean output                             | Notes |
|--------------|----------------------------------------|------------------------------------------|-------|
| YouTube (legacy) | `data/youtube-subtitile/*.vtt`         | `data_clean/youtube/*.txt`               | `scripts/clean_vtt.py` strips timestamps, HTML and filler |
| YouTube (chaînes) | `data/youtube_channels/<slug>/subtitles/*.vtt` | manifests par chaîne dans `data/youtube_channels/<slug>/manifest.json` | `scripts/download_youtube_subtitles.py` organise les sous-titres par créateur |
| LinkedIn     | `data/Linkedin-data/` exports          | `data_clean/linkedin/comments.txt` etc.  | `scripts/prepare_linkedin.py` flattens the API dumps |
| Instagram    | `data/Instagram/` export               | `data_clean/instagram/messages.txt` etc. | `scripts/prepare_instagram.py` keeps the speaker layout |
| Wikipedia    | `data_clean/wikipedia/*.txt`           | same directory                           | `scripts/prepare_wikipedia.py` (download once) then `clean_wikipedia.py` to purge math/LaTeX |
| FineWeb 🍷    | HuggingFace `HuggingFaceFW/fineweb`    | `data_clean/fineweb_fr/*.txt`            | `scripts/download_fineweb.py` streams + filters French docs |
| OSCAR         | HuggingFace `oscar-corpus/OSCAR-23.01` | `data_clean/oscar_fr/*.txt`              | `scripts/download_oscar.py` récupère un sous-corpus unique |
| Bloom Library | HuggingFace `sil-ai/bloom-lm`          | `data_clean/bloom_lm/*.txt`              | `scripts/download_bloom_lm.py` exporte les histoires par langue |
| Hugging Face (curated) | `datasets.load_dataset(<name>)`            | `data/hf_datasets/<slug>/`               | `scripts/download_hf_datasets.py` met en cache des corpus francophones ciblés |

> **Tip** – The dashboard “Préparation des données” section runs the cleaning jobs in the background. These jobs never hit the network: Wikipedia articles are only cleaned, not re-downloaded.

### YouTube par chaîne

- **CLI**

  ```bash
  python scripts/download_youtube_subtitles.py \
    --output-root data/youtube_channels \
    --channel micode=https://www.youtube.com/@Mic0de
  ```

  Le script télécharge uniquement les sous-titres (`yt-dlp`, sans flux vidéo) et range chaque créateur dans `data/youtube_channels/<slug>/`. Les sous-dossiers contiennent :

  - `subtitles/*.vtt` – un fichier par langue et par vidéo (`<video_id>.<lang>.vtt`).
  - `metadata/*.info.json` – métadonnées brutes `yt-dlp` (titre, durée, date).
  - `manifest.json` – récapitulatif synthétique (langues disponibles, date de publication, URL).

  Relancer le script ne re-télécharge pas les vidéos déjà présentes (`download_archive.txt`). Utilisez `--force-refresh` pour ignorer cette archive. Ajoutez autant de chaînes que nécessaire via `--channel slug=url` ou limitez l'exécution à quelques slugs avec `--only squeezie --only seb`.

### FineWeb (French slice)

- **CLI**

  ```bash
  python scripts/download_fineweb.py \
    --config sample-10BT \
    --max-docs 2000 \
    --max-mib 200 \
    --lang-detect-max-chars 4000 \
    --output-dir data_clean/fineweb_fr
  ```

  This streams `HuggingFaceFW/fineweb` via `datasets`, keeps documents detected as French (`langdetect` ≥ 0.7), and writes both a `.txt` corpus and a `.meta.json` summary under `data_clean/fineweb_fr/`.

- **Dashboard** – the “FineWeb (FR)” tile in **Préparation des données** prompts for (1) document budget, (2) Hugging Face config (`sample-10BT`, `data/CC-MAIN-2024-10`, …), and (3) an optional MiB cap, then spawns the same pipeline server-side.

> FineWeb is mostly English. Language detection is heuristic—consider additional filtering (e.g. stop-word checks) before mixing with your curated corpus.

### OSCAR (single-language slices)

- **CLI**

  ```bash
  python scripts/download_oscar.py \
    --dataset oscar-corpus/OSCAR-23.01 \
    --language fr \
    --output-dir data_clean/oscar_fr \
    --progress-interval 2000
  ```

  OSCAR est déjà séparé par langue ; aucun filtrage supplémentaire n’est appliqué. Le script réunit les documents dans `data_clean/oscar_fr/` (avec un fichier `.meta.json` récapitulatif). Le bouton “OSCAR (langue seule)” dans le dashboard déclenche la même opération sans passer par le terminal.

### Bloom Library (sil-ai/bloom-lm)

> **Important** – ce jeu de données nécessite d’accepter la licence et de partager vos coordonnées sur Hugging Face avant tout téléchargement (`https://huggingface.co/datasets/sil-ai/bloom-lm`). Pensez également à vous authentifier (`huggingface-cli login`).

- **CLI**

  ```bash
  python scripts/download_bloom_lm.py \
    --dataset sil-ai/bloom-lm \
    --language fra \
    --output-dir data_clean/bloom_lm \
    --progress-interval 200
  ```

  Options utiles :

  - `--language` peut être répété (`--language fra --language eng`).
  - `--all-languages` rapatrie toutes les langues (363+ codes, attention au volume).
  - `--splits train,validation` pour ignorer un split.
  - `--max-docs 50` pour limiter la taille de chaque split.
  - `--list-languages` affiche la liste complète des codes ISO disponibles puis quitte.

  Les textes sont concaténés par langue/split dans `data_clean/bloom_lm/bloom_lm_<lang>_<split>.txt`, accompagnés d’un fichier `.meta.json` contenant les statistiques (documents conservés, taille, temps passé). Intégrez ensuite ce corpus à l’entraînement via `--extra-data-dir data_clean/bloom_lm` ou en cochant manuellement la source dans le dashboard (sources additionnelles).

---

## Tokenizer Training

`scripts/train_tokenizer.py` trains a byte-level BPE tokenizer using [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) on **all** cleaned text (`data_clean/` recursively). Key options:

```bash
# Minimal example – writes to trained_models/tokenizers/wiki-bpe/
python scripts/train_tokenizer.py \
  --name wiki-bpe \
  --input-dir data_clean \
  --vocab-size 32000 \
  --min-frequency 2 \
  --lowercase            # optional

# Limit the corpus (e.g. first 10 MiB)
python scripts/train_tokenizer.py --limit-mb 10 --limit-files 500
```

Output artefacts:

- `tokenizer.json` – the BPE model (used by the dashboard test widget and downstream inference/training).
- `metadata.json` – config trace (input directory, vocab size, case handling, duration, number of files).

The dashboard exposes a dedicated **Tokenizer** panel:

- Launch training, monitor progress (% fichiers traités, taille nettoyée, erreurs).
- Browse previous tokenizers (badges).
- **Test**: type a sentence and inspect both *byte-level tokens* and *original segments* decoded via offsets.

> ByteLevel uses a leading `Ġ` to mark spaces; the “Segments originaux” list is the human-readable view.

---

## MiniGPT Architecture (Layers & Purpose)

The model in `train_subtitles_transformer.py` is intentionally minimal but faithful to GPT-style blocks:

1. **Token Embedding (`tok_embed`)** – maps raw bytes (0–255) to dense vectors and shares weights with the output head.
2. **Positional Encoding (`PositionalEncoding`)** – sinusoidal encodings added to embeddings, injecting order information.
3. **Transformer Blocks (`nn.TransformerEncoderLayer`)** – repeated `num_layers` times:
   - Multi-head self-attention (`num_heads`) with causal masking.
   - Feed-forward MLP (`ff_hidden_dim`) with GELU activations.
   - Residual connections + LayerNorm inside each block.
4. **Projection / Normalisation** – optional linear pre/post projections (`layernorm_dim`, `head_dim`) + final LayerNorm.
5. **Output Head (`head`)** – projects back to byte vocabulary. Weight tying with the embedding saves parameters.

Sampling options (`sample_interval`, `sample_temperature`, `sample_top_k`) allow quick sanity checks during training and power the dashboard chat.

---

## Training Pipeline

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch (choose the URL matching your CUDA/cuDNN setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Project dependencies
pip install -r requirements.txt   # includes tokenizers, Flask, requests…
```

### Clean the Data

You can use the dashboard (preferred) or scripts manually:

```bash
python scripts/clean_vtt.py --source-dir data/youtube-subtitile --target-dir data_clean/youtube
python scripts/prepare_linkedin.py --input data/Linkedin-data --output data_clean/linkedin
python scripts/prepare_instagram.py --input data/Instagram --output data_clean/instagram
python scripts/prepare_wikipedia.py --language fr --min-articles 10000   # one-off download
python scripts/clean_wikipedia.py --input-dir data_clean/wikipedia       # strip LaTeX/templates
```

### Train the Tokenizer

```bash
python scripts/train_tokenizer.py --name tokenizer-bpe --input-dir data_clean --vocab-size 32000
```

The dashboard > “Tokenizer BPE” mirrors these arguments, tracks progress in real time and lets you test the output.

### Train the Transformer

```bash
# Quick CLI example (200 steps, mini preset)
python scripts/train_subtitles_transformer.py \
  --arch-preset mini-gpt \
  --data-dir data_clean/youtube \
  --extra-data-dir data_clean/fineweb_fr \
  --batch-size 16 \
  --max-steps 200 \
  --log-dir trained_models/runs
```

`--extra-data-dir` est répétable : ajoutez autant de répertoires nettoyés que souhaité (LinkedIn, Instagram, FineWeb, …). Chaque corpus est concaténé avec un séparateur `<|doc|>`.

Les fenêtres sont désormais tirées avec des longueurs variables (min = 50 % du `block_size` par défaut), ce qui densifie les exemples vus. Ajuste ce comportement via `--min-seq-len` pour fixer une longueur minimale spécifique.

For reproducible end-to-end runs, see `scripts/run_subtitle_pipeline.sh` (honours env vars such as `RUN_NAME`, `TRAIN_STEPS`, `START_DASHBOARD=1`).

> **Max steps must stay > 0** – the dashboard guards against zero/negative values and falls back to defaults when necessary.

---

## Dashboard (Backend + Frontend)

### Backend

```bash
RUN_DIR=$(ls -td trained_models/runs/run-* | head -n1)
python dashboard/server.py --run-dir "$RUN_DIR" --checkpoint "$RUN_DIR/checkpoint.pt" --port 8000
```

Endpoints include:

- `/metadata`, `/metrics`, `/visuals`, `/runs`, `/select`, `/chat` – model state & chat.
- `/data/prepare`, `/data/status` – launch & follow cleaning jobs (YouTube, LinkedIn, Instagram, Wikipedia).
- `/tokenizer/train`, `/tokenizer/status`, `/tokenizer/test` – tokenizer lifecycle.
- `/train`, `/train/status`, `/train/resume` – start or resume Transformer runs (le `resume` prend un chemin de checkpoint existant et un nouveau `max_steps`).

> Device defaults to CUDA; if unavailable we automatically fall back to CPU.

### Frontend

```bash
cd dashboard/web
npm install
npm run dev          # Vite on http://127.0.0.1:5173 (proxy → backend)
# or npm run build && npm run preview
```

Key panels:

- **Préparation des données** – start/monitor cleaning jobs.
- **Tokenizer BPE** – configure, launch, preview tokenisation (byte tokens vs decoded segments).
- **Architecture du Transformer** – adjust layers, dimensions, presets (mini/small GPT, etc.).
- **Configuration & lancement de l'entraînement** – the revamped input grid mirrors the architecture cards (button « Nouveau run » génère des noms uniques pour tokenizer & modèle, modifiables avant lancement).
- **Continuer l'entraînement** – sélectionnez un run existant (avec checkpoint), puis cliquez sur « Continuer l'entraînement » pour spécifier un nouveau `max_steps` et reprendre directement depuis le checkpoint.
- **Sources additionnelles** – cochez des répertoires (FineWeb FR, LinkedIn…) pour les ajouter au jeu d’entraînement actuel sans perdre les sous-titres historiques.
- **Monitoring** – loss charts, PCA of embeddings, latest logits, run metadata.
- **Chat** – interact with the currently loaded checkpoint (temperature/top-k tunable).

State polling keeps tokenizer & training progress up to date without manual refresh.

---

## Troubleshooting & Tips

- **Tokenizer progress doesn’t move** – ensure `tokenizers` is installed (`pip install tokenizers`) and check the backend logs (progress is emitted every few files).
- **Training finishes instantly** – verify `max_steps > 0`. The dashboard applies defaults if it reads `0` from a previous run.
- **Accents look corrupted** – the tokenizer test panel shows both raw tokens (ByteLevel format) and decoded segments. Use the latter for readability.
- **Reproducing a run** – copy the `metadata.json` + `config` block from `trained_models/runs/run-*/metadata.json` and feed it back through `/train` or the dashboard form.

---

## Contributing

- Default coding style: Python 3.10, type hints, black formatting.
- All scripts log to stdout/stderr for observability.
- New data sources should write to `data_clean/<source>/` as `.txt` to remain tokenizer/model-friendly.
- The dashboard pulls frequently; keep endpoints idempotent and streaming-friendly.

Happy training! 🎯
