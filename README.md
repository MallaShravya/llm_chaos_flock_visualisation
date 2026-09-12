# LLM Chaos — Flock Visualisation

This project turns two almost-identical LLM generations into two 3D flocks and lets their trajectories diverge.

The included experiment uses **Qwen2.5-1.5B-Instruct** twice with the same seed and sampling settings. The prompts differ by only one invisible trailing space:

```text
A: Describe a bird flying over a city, including what it sees below and where it eventually goes.
B: Describe a bird flying over a city, including what it sees below and where it eventually goes. 
```

The generated outputs are identical through token 106 and first diverge at token 107.

## Visualisation rules

### Tokens become birds

- One **generated model token** = one bird.
- Prompt tokens are not birds. The prompt is the initial perturbation; the generated sequence is the evolving system.
- Repeated tokens still create separate birds because they are separate token occurrences.
- Birds remain in the flock after they are created.

### Semantic relationships are computed in the original embedding space

Each generated token uses Qwen's static input embedding.

When a new token appears, its **3 most semantically similar earlier tokens** are found using cosine similarity in the full 1536-dimensional embedding space. The embedding angle between the new token and each of those neighbours controls how strongly that bird responds.

A fixed seeded orthonormal projection maps embeddings to 3D only when a spatial direction is needed. The 3D projection is **not** used to decide semantic similarity.

### A new token begins outside the flock

The new bird is initially stationary and external to the flock. Its spawn direction comes from the projected semantic difference between it and its semantic neighbours.

The three semantic neighbours turn toward the new bird and fly slightly faster than the rest of the flock. This creates a local disturbance that can propagate through the physical flock.

The newborn joins the flock once enough older birds physically reach it. It then inherits the local motion of nearby birds.

### Physical flocking is local

Ordinary flock motion uses the **7 nearest physical neighbours** of each bird.

The physical rules combine:

- heading alignment
- weak speed alignment
- local cohesion
- separation / collision avoidance
- restoration toward a common preferred speed

Semantic neighbours and physical neighbours are deliberately different networks:

```text
semantic event -> 3 semantic responders -> local physical disturbance -> flock response
```

### Semantic structure persists

Once a new bird joins, weak persistent semantic bonds connect it to the semantic neighbours that responded to it. These act like soft springs and let earlier semantic relationships continue to influence the flock's shape.

A separate rescue-cohesion rule activates only if the flock physically breaks into disconnected components. It is a stability constraint, not a semantic force.

### Tokens arrive sequentially

The next token is not introduced immediately.

For each token:

1. the new bird appears;
2. its semantic neighbours respond;
3. the flock moves toward and incorporates it;
4. the disturbance is allowed to settle;
5. only then does the next token appear.

There are deterministic safety limits so a difficult token event cannot stall the entire simulation forever.

### Comparing the two runs

Before the first generated-token divergence, the two simulations are identical.

After divergence, they evolve independently using the same rules and constants.

The viewer keeps the two sides on the same generated-token index so the comparison is always between flocks containing the same number of tokens. Playback is accelerated for viewing, but the trajectories themselves are precomputed offline.

The blue flock is run A; the orange flock is run B. The newest token is shown in black, and the corresponding token is highlighted in the generated text.

> The flock model is a custom, starling-inspired visual/physical system. It is not intended to be an exact biological starling model, and the semantic bonds/newborn mechanics are deliberate visualisation rules.

## Reproduce the included experiment

The repository already contains `run_a.json` and `run_b.json`, including the generated token IDs and embeddings, so you do **not** need a GPU or an LLM API to reproduce the included visualisation.

### Requirements

- Python 3 + pip
- Node.js `20.19+` or `22.12+`

### 1. Install Python dependencies

From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
python -m pip install -r requirements.txt
```

### 2. Project the embeddings to 3D

```bash
python pipeline/project_embeddings.py
```

### 3. Run the flock simulation

```bash
python pipeline/simulate_semantic_flock_optimized_v2.py
```

This creates the two precomputed motion JSON files. They can be large and may take a while to generate.

### 4. Convert the trajectories to compact binary files

```bash
python pipeline/convert_motion_to_binary_v2.py
```

The viewer files are written to:

```text
viewer/public/data/
```

### 5. Start the viewer

```bash
cd viewer
npm install
npm run dev
```

Open the local URL printed by Vite.

That's it.

## Regenerating the LLM outputs

If you want to rerun the experiment from the model instead of using the included `run_a.json` and `run_b.json`, `pipeline/modal_qwen.py` runs Qwen2.5-1.5B-Instruct on a Modal T4 GPU and writes the results to the Modal volume `llm-chaos-output`.

After installing the requirements and configuring Modal:

```bash
python -m modal setup
python -m modal run pipeline/modal_qwen.py
```

Then download the generated files from the volume into the repository root before running the remaining pipeline steps:

```bash
modal volume get llm-chaos-output run_a.json .
modal volume get llm-chaos-output run_b.json .
modal volume get llm-chaos-output metadata.json .
```
