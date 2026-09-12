import json
import numpy as np

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

RUN_A_FILE = "run_a.json"
RUN_B_FILE = "run_b.json"

OUTPUT_A_FILE = "run_a_projected.json"
OUTPUT_B_FILE = "run_b_projected.json"

PROJECTION_FILE = "projection_matrix.npy"
PROJECTION_META_FILE = "projection_meta.json"

# This is now the fixed seed for the 3D projection.
# Keep it unchanged for the experiment.
PROJECTION_SEED = 20260911


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


print("Loading embeddings...")

run_a = load_json(RUN_A_FILE)
run_b = load_json(RUN_B_FILE)

embedding_dim = run_a["embedding_dimension"]

if embedding_dim != run_b["embedding_dimension"]:
    raise ValueError(
        "Run A and Run B have different embedding dimensions."
    )

print("Embedding dimension:", embedding_dim)


# --------------------------------------------------
# CREATE FIXED RANDOM ORTHONORMAL PROJECTION
# --------------------------------------------------

rng = np.random.default_rng(PROJECTION_SEED)

# Random Gaussian matrix:
# shape = embedding_dimension × 3
random_matrix = rng.normal(
    size=(embedding_dim, 3)
)

# QR decomposition gives us orthonormal columns.
#
# Q.T @ Q should be approximately the 3x3 identity matrix.
Q, _ = np.linalg.qr(random_matrix)

projection_matrix = Q[:, :3]

print(
    "Projection matrix shape:",
    projection_matrix.shape,
)


# --------------------------------------------------
# VERIFY ORTHONORMALITY
# --------------------------------------------------

orthogonality_check = (
    projection_matrix.T
    @ projection_matrix
)

print("\nP^T P:")
print(orthogonality_check)

identity = np.eye(3)

max_error = np.max(
    np.abs(
        orthogonality_check - identity
    )
)

print(
    "\nMaximum orthonormality error:",
    max_error,
)

if max_error > 1e-10:
    raise RuntimeError(
        "Projection matrix failed orthonormality check."
    )


# --------------------------------------------------
# PROJECT ONE RUN
# --------------------------------------------------

def project_run(run):
    projected_tokens = []

    for token in run["tokens"]:
        embedding = np.asarray(
            token["embedding"],
            dtype=np.float64,
        )

        if embedding.shape != (embedding_dim,):
            raise ValueError(
                f"Unexpected embedding shape "
                f"at token {token['index']}: "
                f"{embedding.shape}"
            )

        # e ∈ R^d
        # P ∈ R^(d×3)
        #
        # projected = eP ∈ R^3
        projected = (
            embedding
            @ projection_matrix
        )

        projected_tokens.append(
            {
                "index": token["index"],
                "token_id": token["token_id"],
                "token": token["token"],
                "projected_3d": [
                    float(projected[0]),
                    float(projected[1]),
                    float(projected[2]),
                ],
            }
        )

    return {
        "run": run["run"],
        "model": run["model"],
        "prompt": run["prompt"],
        "generated_text": run["generated_text"],
        "token_count": run["token_count"],
        "first_divergence": run["first_divergence"],
        "projection_seed": PROJECTION_SEED,
        "tokens": projected_tokens,
    }


print("\nProjecting run A...")
projected_a = project_run(run_a)

print("Projecting run B...")
projected_b = project_run(run_b)


# --------------------------------------------------
# VERIFY SHARED PREFIX
# --------------------------------------------------

divergence = run_a["first_divergence"]

if divergence is not None:
    print(
        f"\nChecking identical prefix "
        f"through token {divergence - 1}..."
    )

    for i in range(divergence):
        token_a = projected_a["tokens"][i]
        token_b = projected_b["tokens"][i]

        if token_a["token_id"] != token_b["token_id"]:
            raise RuntimeError(
                f"Token IDs unexpectedly differ at {i}."
            )

        a = np.asarray(
            token_a["projected_3d"]
        )

        b = np.asarray(
            token_b["projected_3d"]
        )

        if not np.allclose(
            a,
            b,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Projection differs at shared token {i}."
            )

    print(
        "Shared-prefix projection check passed."
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

np.save(
    PROJECTION_FILE,
    projection_matrix,
)

with open(
    OUTPUT_A_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        projected_a,
        f,
        indent=2,
        ensure_ascii=False,
    )

with open(
    OUTPUT_B_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        projected_b,
        f,
        indent=2,
        ensure_ascii=False,
    )

projection_metadata = {
    "projection_seed": PROJECTION_SEED,
    "input_dimension": embedding_dim,
    "output_dimension": 3,
    "method": "seeded Gaussian matrix followed by QR orthonormalization",
    "matrix_shape": list(
        projection_matrix.shape
    ),
    "max_orthonormality_error": float(
        max_error
    ),
}

with open(
    PROJECTION_META_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        projection_metadata,
        f,
        indent=2,
    )


# --------------------------------------------------
# SHOW A FEW RESULTS
# --------------------------------------------------

print("\nFirst five projected tokens from A:")

for token in projected_a["tokens"][:5]:
    print(
        f"{token['index']:3} "
        f"{token['token']!r:20} "
        f"{token['projected_3d']}"
    )

print("\nDivergence token:")

if divergence is not None:
    a = projected_a["tokens"][divergence]
    b = projected_b["tokens"][divergence]

    print(
        "A:",
        a["token_id"],
        repr(a["token"]),
        a["projected_3d"],
    )

    print(
        "B:",
        b["token_id"],
        repr(b["token"]),
        b["projected_3d"],
    )


print("\nDone.")
print("Created:")
print(" ", PROJECTION_FILE)
print(" ", PROJECTION_META_FILE)
print(" ", OUTPUT_A_FILE)
print(" ", OUTPUT_B_FILE)