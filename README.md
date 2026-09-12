# LLM Chaos

Two runs of the same model, with the same seed, on prompts that differ by **one trailing space**.
They produce identical text for 107 tokens, then tell different stories.

Each generated token becomes a bird. The two flocks are rendered side by side, cameras linked,
so you can watch them fly together and then come apart.

```
Run A   "Describe a bird flying over a city, including what it sees below and where it eventually goes."
Run B   "Describe a bird flying over a city, including what it sees below and where it eventually goes. "
                                                                                                      ^
                                                                                       one extra space
```

| | Run A | Run B |
|---|---|---|
| tokens generated | 209 | 300 (hit the cap) |
| first differing token | 107 | 107 |
| token 107 onward | `…workers toiling away at their desks.` | `…workers pushing heavy boxes up and down stairs.` |
| ending | settles down for the night among the stars | a man speaking animatedly in the distance |

Model is `Qwen/Qwen2.5-1.5B-Instruct`, seed 42, temperature 0.8, top-p 0.9, 300 max new tokens.
The seed is reset before each run, so both draw from an identical RNG stream — the divergence comes
from the prompt edit shifting the next-token distribution, not from different random numbers.

## Quick start — no GPU needed

`run_a.json` and `run_b.json` are committed, so the expensive GPU step is already done for you.
Clone and run these from the repository root:

```bash
python -m pip install -r requirements.txt
python pipeline/project_embeddings.py
python pipeline/simulate_semantic_flock_optimized_v2.py
python pipeline/convert_motion_to_binary_v2.py
cd viewer && npm install && npm run dev
```

Run every Python command **from the repository root** — the scripts use paths relative to the
working directory. Step 3 takes a few minutes and writes about 500 MB; step 4 compresses that into
the ~95 MB of binaries the viewer reads.

Generated trajectory data is not committed — the binaries alone are ~95 MB and the intermediate
JSON is ~500 MB, so both are rebuilt locally and ignored by git.

## The pipeline

| | script | does | committed? |
|---|---|---|---|
| 1 | `modal_qwen.py` | generate both runs, export token embeddings (Modal, T4 GPU) | yes, output in repo |
| 2 | `project_embeddings.py` | 1536-d to 3-d orthonormal projection | rebuilt locally |
| 3 | `simulate_semantic_flock_optimized_v2.py` | semantic flocking simulation | rebuilt locally |
| 4 | `convert_motion_to_binary_v2.py` | JSON to LLMCHS01 binary | rebuilt locally |

### Step 1 — generate both runs and export token embeddings (optional)

Already done; its output is in the repo. Re-run it only if you want to change the prompts, the
model, or the sampling settings — see [Trying your own prompt edits](#trying-your-own-prompt-edits).

Loads the model on a T4, generates both continuations, and records for every generated token its
id, its text, and its 1536-dimensional row from the model's input embedding matrix.

### Step 2 — project 1536-d embeddings down to 3-d

Builds a fixed random orthonormal projection — seeded Gaussian matrix, then QR — and applies it to
both runs. The seed (`20260911`) is hardcoded, so the projection is identical across machines and
the two runs land in a shared space. Writes `run_a_projected.json`, `run_b_projected.json`,
`projection_matrix.npy` and `projection_meta.json`, which records the orthonormality error
(~3.3e-16).

A random projection is deliberate: it preserves distances in expectation without being fitted to
the data, so the flocks are not arranged to make divergence look larger than it is.

### Step 3 — simulate the flocks

The long one — a few minutes, and it writes about 500 MB. Every token is born outside the flock,
positioned relative to its three nearest neighbours in embedding space. Those three then physically
pursue it, their turn rate set by the embedding angle, until they make contact and form persistent
bonds. On top of that sit ordinary boids rules: seven-neighbour alignment, cohesion, separation,
plus rescue cohesion that keeps the flock connected and a deterministic straight-line fallback so a
pursuit can never fail to complete.

Simulated at 1/30 s, recording every 4th step.

### Step 4 — pack the trajectories into a compact binary

Streams that 500 MB of JSON through `ijson`, so it never loads a full file into memory, and writes
a small binary format (`LLMCHS01`: 24-byte header, 24-byte index entry per frame, 6 floats per bird)
into `viewer/public/data/`. Roughly 18x smaller, and the browser reads floats directly instead of
parsing JSON. Produces `run_a_motion_v2.bin` (~32 MB) and `run_b_motion_v2.bin` (~67 MB) plus their
`.meta.json` sidecars.

### Step 5 — run the viewer

Three.js, two linked-camera scenes. Hover a bird for its token; use rewind and play/pause to scrub.
Each panel shows the newest token as it lands and the generated text so far.

The two runs are **synced on token index, not on time**. Run A takes 1514 s of simulated time for
209 tokens and run B takes 2299 s for 300, so playing both against one clock would drift them onto
different tokens within seconds of the fork. Instead each run moves continuously through its own
recorded interval for the current token, and the shorter interval is stretched just enough that both
arrive at the next shared token boundary together — with overshoot carried forward, so there is no
stall at the boundary. Whatever you see on the left and the right is always the same token index.

## Trying your own prompt edits

The trailing space is one perturbation out of many, and the interesting question is which kinds of
edit cause divergence and how fast. Everything you need is at the top of `pipeline/modal_qwen.py`:

```python
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # line 51

SEED = 42                              # line 53
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 300

prompt_a = (...)                       # line 58
prompt_b = (...)                       # line 64  <- the edited one
```

Edit `prompt_a` / `prompt_b`, re-run step 1, then steps 2-4 as usual. Things worth trying:

- **punctuation** — a full stop vs an exclamation mark, or an inserted comma
- **a one-letter typo** — `robot` vs `robt`
- **capitalisation** — `a bird` vs `a Bird`, or lowercasing the first word
- **an article swap** — `the old garden` vs `an old garden`
- **a double space** in the middle of the sentence
- **no edit at all** — identical prompts, as a control; the runs should stay identical forever, which
  is a useful check that nothing in your setup is nondeterministic

Keep `SEED` fixed when comparing edits, so the only thing changing is the prompt. Raising
`TEMPERATURE` makes divergence happen sooner and more often; dropping to greedy decoding
(`do_sample=False`) removes the sampler from the picture entirely, which makes any divergence
attributable purely to the model rather than to a resolved near-tie.

`metadata.json` is rewritten on each run with the token counts and the first divergence index, so it
is the quickest way to see what your edit did.

## Setting up Modal (only for step 1)

Step 1 is the sole GPU step, roughly a minute on a T4.

1. Create an account at [modal.com](https://modal.com) — the free tier covers a run this size.
2. Install and authenticate:
   ```bash
   python -m pip install modal
   python -m modal setup
   ```
   This opens a browser to link the CLI to your account.
3. Run the job from the repository root:
   ```bash
   modal run pipeline/modal_qwen.py
   ```
   Modal builds the remote image on first run (a few minutes; cached afterwards), loads the model,
   generates both runs, and writes the results to a Modal volume named `llm-chaos-output`.
4. Pull the output down to the repository root:
   ```bash
   modal volume get llm-chaos-output run_a.json .
   modal volume get llm-chaos-output run_b.json .
   modal volume get llm-chaos-output metadata.json .
   ```
   Add `--force` to overwrite the committed copies.

Two volumes are created automatically: `llm-chaos-hf-cache` holds the downloaded model weights, so
later runs skip the download, and `llm-chaos-output` holds the exported runs.
`modal volume ls llm-chaos-output` lists what is there.

`torch`, `transformers` and `accelerate` are deliberately absent from `requirements.txt` — they are
installed inside the remote Modal image (see the `modal.Image` definition at the top of
`pipeline/modal_qwen.py`), never on your machine. Locally you only need the Modal client.

## Layout

```
run_a.json  run_b.json  metadata.json     step 1 output, committed so you can skip the GPU
pipeline/
  modal_qwen.py                           1. generate + export token embeddings  (Modal, GPU)
  project_embeddings.py                   2. 1536-d to 3-d orthonormal projection
  simulate_semantic_flock_optimized_v2.py 3. semantic flocking simulation
  convert_motion_to_binary_v2.py          4. JSON to LLMCHS01 binary
viewer/
  index.html
  src/main_v2_boundary_synced.js          Three.js renderer + binary loader (the one index.html loads)
  src/main_v2.js                          earlier renderer, kept for comparison
  src/style.css
  public/                                 built trajectories land in public/data/
```

## Notes

- **Deploying to GitHub Pages needs one change.** `viewer/src/main_v2_boundary_synced.js` fetches
  `/data/run_*_motion_v2.bin` with a leading slash. That resolves correctly on `npm run dev` and at
  a domain root, but a project Pages site is served from `username.github.io/<repo>/`, where those
  paths 404. Set `base` in a `vite.config.js` and make the fetch paths relative, or deploy to a
  user/org root site.
- **Step 3 output is large.** Both motion JSON files together are about 500 MB. They are only an
  intermediate — once step 4 has produced the binaries you can delete them.
- Only steps 3 and 4 are v2-specific. Steps 1 and 2 are shared with v1.

## License

MIT — see [LICENSE](LICENSE).
