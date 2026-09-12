
import json
import numpy as np


# ============================================================
# FILES
# ============================================================

RUN_A_FILE = "run_a.json"
RUN_B_FILE = "run_b.json"

PROJECTED_A_FILE = "run_a_projected.json"
PROJECTED_B_FILE = "run_b_projected.json"

OUTPUT_A_FILE = "run_a_semantic_motion_v2.json"
OUTPUT_B_FILE = "run_b_semantic_motion_v2.json"


# ============================================================
# SIMULATION
# ============================================================

DT = 1.0 / 30.0
RECORD_EVERY = 4

PHYSICAL_NEIGHBOR_COUNT = 7
SEMANTIC_NEIGHBOR_COUNT = 3


# ============================================================
# BASE MOTION
# ============================================================

BASE_SPEED = 0.65

INITIAL_DIRECTION = np.array(
    [1.0, 0.25, 0.1],
    dtype=np.float64,
)
INITIAL_DIRECTION /= np.linalg.norm(INITIAL_DIRECTION)


# ============================================================
# LOCAL FLOCKING
# ============================================================

ALIGNMENT_STRENGTH = 1.4
SPEED_ALIGNMENT_STRENGTH = 0.30
COHESION_STRENGTH = 0.70
SEPARATION_STRENGTH = 2.8
SPEED_RESTORE_STRENGTH = 1.15

SEPARATION_RADIUS = 0.70
HARD_COLLISION_RADIUS = 0.24

MAX_ACCELERATION = 3.0
MAX_TURN_RATE = np.deg2rad(155.0)


# ============================================================
# SPEED LIMITS
# ============================================================

# Ordinary birds can inherit a small part of the speed wave.
GENERAL_MAX_SPEED = BASE_SPEED * 1.06

# The three semantic responders remain visibly at the front,
# but only modestly faster than the rest of the flock.
RESPONDER_MIN_SPEED_FACTOR = 1.08
RESPONDER_MAX_SPEED_FACTOR = 1.14
RESPONDER_SPEED_RESPONSE = 4.0

MIN_SPEED = BASE_SPEED * 0.60


# ============================================================
# TOKEN BIRTH DISTANCE
# ============================================================

MIN_BIRTH_DISTANCE = 0.45
MAX_BIRTH_DISTANCE = 0.65
BIRTH_DISTANCE_FACTOR = 0.10

# Use the actual outer radius so the newborn is guaranteed to
# start outside every existing bird, not merely outside 95%.
FLOCK_RADIUS_PERCENTILE = 92


# ============================================================
# EXTERNAL TOKEN / JOIN RULE
# ============================================================

JOIN_NEIGHBOR_COUNT = 3
JOIN_DISTANCE = 0.40


# ============================================================
# SUSTAINED SEMANTIC PURSUIT
# ============================================================

# There is no time decay while the newborn is external.
# The event ends because the newborn is physically reached.

SEMANTIC_TURN_STRENGTH = 2.4

# Embedding angle still modulates the response.
PURSUIT_MIN_TURN_RATE = np.deg2rad(55.0)
PURSUIT_MAX_TURN_RATE = np.deg2rad(120.0)

# If ordinary flock forces somehow prevent contact for too long,
# switch to a deterministic straight pursuit instead of aborting.
EMERGENCY_PURSUIT_AFTER_STEPS = 180


# ============================================================
# PERSISTENT SEMANTIC BONDS
# ============================================================

SEMANTIC_BOND_STRENGTH = 0.22
SEMANTIC_BOND_REST_DISTANCE = 0.80
SEMANTIC_BOND_MAX_FORCE = 0.35


# ============================================================
# RESCUE COHESION
# ============================================================

RESCUE_COHESION_STRENGTH = 0.45
CONNECTIVITY_DISTANCE_FACTOR = 2.5
MIN_CONNECTIVITY_DISTANCE = 0.85


# ============================================================
# EVENT RESOLUTION
# ============================================================

MIN_SETTLE_STEPS = 15
STABLE_WINDOW = 6

SPEED_EPSILON = 0.055

# Once a newborn has joined, give the speed pulse at most this
# many steps to damp. If the continuous flock never crosses the
# numerical threshold, continue rather than aborting the run.
MAX_POST_JOIN_SETTLE_STEPS = 90


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude < 1e-12:
        return np.zeros_like(vector)
    return vector / magnitude


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def safe_direction(vector, fallback=None):
    direction = normalize(vector)

    if np.linalg.norm(direction) > 1e-12:
        return direction

    if fallback is not None:
        direction = normalize(fallback)
        if np.linalg.norm(direction) > 1e-12:
            return direction

    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-12:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b) / denominator,
            -1.0,
            1.0,
        )
    )


def embedding_angle(a, b):
    return float(np.arccos(cosine_similarity(a, b)))


def deterministic_perpendicular(vector):
    direction = safe_direction(vector, INITIAL_DIRECTION)

    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    axis = min(axes, key=lambda x: abs(float(np.dot(direction, x))))
    perpendicular = np.cross(direction, axis)

    return safe_direction(
        perpendicular,
        np.array([0.0, 0.0, 1.0]),
    )


def rotate_towards(current, target, max_angle):
    current = safe_direction(current, target)
    target = safe_direction(target, current)

    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    angle = float(np.arccos(dot))

    if angle <= max_angle or angle < 1e-10:
        return target

    axis = np.cross(current, target)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        axis = deterministic_perpendicular(current)
    else:
        axis = axis / axis_norm

    c = np.cos(max_angle)
    s = np.sin(max_angle)

    rotated = (
        current * c
        + np.cross(axis, current) * s
        + axis * np.dot(axis, current) * (1.0 - c)
    )

    return safe_direction(rotated, target)



# ============================================================
# OPTIMIZED VECTOR HELPERS
# ============================================================

def safe_direction_rows(vectors, fallback):
    """Vectorized equivalent of safe_direction for an (N, 3) array."""
    vectors = np.asarray(vectors, dtype=np.float64)

    if len(vectors) == 0:
        return vectors.copy()

    norms = np.linalg.norm(vectors, axis=1)
    result = np.empty_like(vectors)

    valid = norms >= 1e-12

    if np.any(valid):
        result[valid] = (
            vectors[valid]
            / norms[valid, None]
        )

    if np.any(~valid):
        fallback_array = np.asarray(
            fallback,
            dtype=np.float64,
        )

        if fallback_array.ndim == 1:
            fallback_array = np.broadcast_to(
                fallback_array,
                vectors.shape,
            )

        bad_indices = np.flatnonzero(~valid)
        bad_fallback = fallback_array[bad_indices]
        fallback_norms = np.linalg.norm(
            bad_fallback,
            axis=1,
        )

        fallback_valid = fallback_norms >= 1e-12

        if np.any(fallback_valid):
            selected = bad_indices[fallback_valid]
            result[selected] = (
                fallback_array[selected]
                / fallback_norms[fallback_valid, None]
            )

        if np.any(~fallback_valid):
            selected = bad_indices[~fallback_valid]
            result[selected] = np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float64,
            )

    return result


def rotate_towards_rows(
    current,
    target,
    max_angle,
):
    """
    Vectorized equivalent of rotate_towards.

    The ordinary per-bird turn law is unchanged; this only removes the
    Python loop around it. The rare nearly-antiparallel fallback still uses
    the exact scalar deterministic_perpendicular helper.
    """
    current = safe_direction_rows(
        current,
        target,
    )

    target = safe_direction_rows(
        target,
        current,
    )

    dots = np.clip(
        np.sum(current * target, axis=1),
        -1.0,
        1.0,
    )

    angles = np.arccos(dots)
    result = target.copy()

    rotating = (
        (angles > max_angle)
        & (angles >= 1e-10)
    )

    if not np.any(rotating):
        return result

    current_rotating = current[rotating]
    target_rotating = target[rotating]

    axes = np.cross(
        current_rotating,
        target_rotating,
    )

    axis_norms = np.linalg.norm(
        axes,
        axis=1,
    )

    normal_axes = axis_norms >= 1e-10

    if np.any(normal_axes):
        axes[normal_axes] /= (
            axis_norms[normal_axes, None]
        )

    if np.any(~normal_axes):
        for local_index in np.flatnonzero(
            ~normal_axes
        ):
            axes[local_index] = (
                deterministic_perpendicular(
                    current_rotating[local_index]
                )
            )

    c = np.cos(max_angle)
    s = np.sin(max_angle)

    rotated = (
        current_rotating * c
        + np.cross(
            axes,
            current_rotating,
        ) * s
        + axes
        * np.sum(
            axes * current_rotating,
            axis=1,
        )[:, None]
        * (1.0 - c)
    )

    result[rotating] = safe_direction_rows(
        rotated,
        target_rotating,
    )

    return result


# ============================================================
# SEMANTIC NEIGHBOURS
# ============================================================

def semantic_neighbors(embeddings, new_index):
    if new_index == 0:
        return []

    count = min(SEMANTIC_NEIGHBOR_COUNT, new_index)

    new_embedding = embeddings[new_index]
    previous = embeddings[:new_index]

    previous_norms = np.linalg.norm(previous, axis=1)
    new_norm = np.linalg.norm(new_embedding)

    denominator = np.maximum(
        previous_norms * new_norm,
        1e-12,
    )

    similarities = (previous @ new_embedding) / denominator

    indices = np.argsort(similarities)[-count:][::-1]

    return indices.astype(int).tolist()


# ============================================================
# BIRTH POSITION
# ============================================================

def birth_position(
    new_index,
    positions,
    projected,
    neighbors,
):
    if new_index == 0:
        return np.zeros(3, dtype=np.float64)

    centroid = np.mean(positions, axis=0)

    distances = np.linalg.norm(
        positions - centroid,
        axis=1,
    )

    flock_radius = float(
        np.percentile(
            distances,
            FLOCK_RADIUS_PERCENTILE,
        )
    )

    if len(neighbors) > 0:
        semantic_center = np.mean(
            projected[neighbors],
            axis=0,
        )
    else:
        semantic_center = np.zeros(3, dtype=np.float64)

    semantic_difference = (
        projected[new_index] - semantic_center
    )

    direction = safe_direction(
        semantic_difference,
        projected[new_index],
    )

    extra_distance = float(
        np.clip(
            BIRTH_DISTANCE_FACTOR * flock_radius,
            MIN_BIRTH_DISTANCE,
            MAX_BIRTH_DISTANCE,
        )
    )

    spawn_radius = flock_radius + extra_distance

    return centroid + spawn_radius * direction


# ============================================================
# SEMANTIC PURSUIT RECORDS
# ============================================================

def build_semantic_pursuits(
    new_index,
    neighbors,
    embeddings,
):
    pursuits = []

    for bird_index in neighbors:
        angle = embedding_angle(
            embeddings[bird_index],
            embeddings[new_index],
        )

        pursuits.append(
            {
                "bird_index": int(bird_index),
                "target_index": int(new_index),
                "embedding_angle": float(angle),
            }
        )

    return pursuits


# ============================================================
# NEWBORN VELOCITY
# ============================================================

def newborn_velocity(new_index):
    if new_index == 0:
        return INITIAL_DIRECTION * BASE_SPEED

    # Newborns are stationary until physically incorporated.
    return np.zeros(3, dtype=np.float64)


# ============================================================
# EXTERNAL TOKEN RELEASE
# ============================================================

def update_external_membership(
    positions,
    velocities,
    external_tokens,
):
    if not external_tokens:
        return set()

    released = set()

    for bird_index in list(external_tokens):
        if bird_index <= 0:
            released.add(bird_index)
            continue

        older_positions = positions[:bird_index]
        older_count = len(older_positions)

        if older_count == 0:
            continue

        distances = np.linalg.norm(
            older_positions - positions[bird_index],
            axis=1,
        )

        required_neighbors = min(
            JOIN_NEIGHBOR_COUNT,
            older_count,
        )

        nearby_indices = np.where(
            distances <= JOIN_DISTANCE
        )[0]

        if len(nearby_indices) < required_neighbors:
            continue

        released.add(bird_index)

        sorted_nearby = nearby_indices[
            np.argsort(distances[nearby_indices])
        ]

        selected = sorted_nearby[:required_neighbors]

        local_velocity = np.mean(
            velocities[selected],
            axis=0,
        )

        local_direction = safe_direction(
            local_velocity,
            INITIAL_DIRECTION,
        )

        local_speed = float(
            np.mean(
                np.linalg.norm(
                    velocities[selected],
                    axis=1,
                )
            )
        )

        launch_speed = float(
            np.clip(
                local_speed,
                BASE_SPEED * 0.95,
                GENERAL_MAX_SPEED,
            )
        )

        velocities[bird_index] = (
            local_direction * launch_speed
        )

    return released


# ============================================================
# SEMANTIC BONDS
# ============================================================

def semantic_bond_accelerations(
    positions,
    bond_array,
    external_tokens,
):
    """
    Same spring law as before, evaluated in one NumPy batch.

    np.add.at preserves repeated-index accumulation, so a bird connected to
    several semantic bonds receives the sum of all of those forces exactly as
    in the scalar loop.
    """
    accelerations = np.zeros_like(positions)

    if bond_array is None or len(bond_array) == 0:
        return accelerations

    active_bonds = bond_array

    if external_tokens:
        external_mask = np.zeros(
            len(positions),
            dtype=bool,
        )

        external_indices = np.fromiter(
            external_tokens,
            dtype=np.int64,
            count=len(external_tokens),
        )

        external_mask[external_indices] = True

        keep = ~(
            external_mask[active_bonds[:, 0]]
            | external_mask[active_bonds[:, 1]]
        )

        active_bonds = active_bonds[keep]

    if len(active_bonds) == 0:
        return accelerations

    i = active_bonds[:, 0]
    j = active_bonds[:, 1]

    difference = (
        positions[j]
        - positions[i]
    )

    distance = np.linalg.norm(
        difference,
        axis=1,
    )

    valid = distance >= 1e-9

    if not np.any(valid):
        return accelerations

    i = i[valid]
    j = j[valid]
    difference = difference[valid]
    distance = distance[valid]

    direction = (
        difference
        / distance[:, None]
    )

    displacement = (
        distance
        - SEMANTIC_BOND_REST_DISTANCE
    )

    force_magnitude = np.clip(
        SEMANTIC_BOND_STRENGTH
        * displacement,
        -SEMANTIC_BOND_MAX_FORCE,
        SEMANTIC_BOND_MAX_FORCE,
    )

    force = (
        direction
        * force_magnitude[:, None]
    )

    np.add.at(
        accelerations,
        i,
        force,
    )

    np.add.at(
        accelerations,
        j,
        -force,
    )

    return accelerations


# ============================================================
# SHARED PHYSICAL GEOMETRY
# ============================================================

def build_ordinary_geometry(
    positions,
    velocities,
    external_tokens,
):
    """
    Build the pairwise physical geometry ONCE for this timestep.

    The old simulator independently rebuilt the same distance matrix and the
    same seven-neighbour lookup for local flocking and rescue connectivity.
    Both systems now consume this shared object. No physical rule changes.
    """
    n = len(positions)

    ordinary_mask = np.ones(
        n,
        dtype=bool,
    )

    if external_tokens:
        external_indices = np.fromiter(
            external_tokens,
            dtype=np.int64,
            count=len(external_tokens),
        )
        ordinary_mask[external_indices] = False

    ordinary_indices = np.flatnonzero(
        ordinary_mask
    )

    ordinary_positions = positions[
        ordinary_indices
    ]

    ordinary_velocities = velocities[
        ordinary_indices
    ]

    current_speeds = np.linalg.norm(
        ordinary_velocities,
        axis=1,
    )

    # Preserve the original safe_direction behaviour here. This is only O(N)
    # and keeps the force calculation numerically as close as possible to the
    # previous implementation.
    current_headings = np.array(
        [
            safe_direction(
                velocity,
                INITIAL_DIRECTION,
            )
            for velocity in ordinary_velocities
        ],
        dtype=np.float64,
    )

    m = len(ordinary_indices)

    if m <= 1:
        return {
            "indices": ordinary_indices,
            "positions": ordinary_positions,
            "velocities": ordinary_velocities,
            "speeds": current_speeds,
            "headings": current_headings,
            "difference": None,
            "distance_sq": None,
            "distances": None,
            "nearest": None,
        }

    difference = (
        ordinary_positions[:, None, :]
        - ordinary_positions[None, :, :]
    )

    distance_sq = np.sum(
        difference * difference,
        axis=2,
    )

    np.fill_diagonal(
        distance_sq,
        np.inf,
    )

    distances = np.sqrt(
        distance_sq
    )

    k = min(
        PHYSICAL_NEIGHBOR_COUNT,
        m - 1,
    )

    nearest = np.argpartition(
        distance_sq,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return {
        "indices": ordinary_indices,
        "positions": ordinary_positions,
        "velocities": ordinary_velocities,
        "speeds": current_speeds,
        "headings": current_headings,
        "difference": difference,
        "distance_sq": distance_sq,
        "distances": distances,
        "nearest": nearest,
    }


# ============================================================
# CONNECTIVITY / RESCUE COHESION
# ============================================================

def rescue_accelerations_from_geometry(
    positions,
    geometry,
):
    acceleration = np.zeros_like(positions)

    ordinary_indices = geometry["indices"]
    m = len(ordinary_indices)

    if m < 3:
        return acceleration

    distances = geometry["distances"]
    nearest = geometry["nearest"]

    nearest_distance = np.min(
        distances,
        axis=1,
    )

    finite_nearest = nearest_distance[
        np.isfinite(nearest_distance)
    ]

    if len(finite_nearest) == 0:
        median_nearest = (
            MIN_CONNECTIVITY_DISTANCE
        )
    else:
        median_nearest = float(
            np.median(finite_nearest)
        )

    link_distance = max(
        MIN_CONNECTIVITY_DISTANCE,
        CONNECTIVITY_DISTANCE_FACTOR
        * median_nearest,
    )

    # Preserve the exact graph/component logic from the previous simulator;
    # only the expensive geometry above is reused.
    graph = [
        set()
        for _ in range(len(positions))
    ]

    for local_i in range(m):
        global_i = int(
            ordinary_indices[local_i]
        )

        for local_j in nearest[local_i]:
            local_j = int(local_j)
            global_j = int(
                ordinary_indices[local_j]
            )

            if (
                distances[local_i, local_j]
                <= link_distance
            ):
                graph[global_i].add(
                    global_j
                )
                graph[global_j].add(
                    global_i
                )

    visited = set()
    components = []

    for start in ordinary_indices:
        start = int(start)

        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        component = []

        while stack:
            node = stack.pop()
            component.append(node)

            for neighbour in graph[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        components.append(component)

    if len(components) <= 1:
        return acceleration

    main_component = max(
        components,
        key=len,
    )

    main_center = np.mean(
        positions[main_component],
        axis=0,
    )

    for component in components:
        if component is main_component:
            continue

        component_center = np.mean(
            positions[component],
            axis=0,
        )

        direction = safe_direction(
            main_center
            - component_center
        )

        for bird_index in component:
            acceleration[bird_index] += (
                RESCUE_COHESION_STRENGTH
                * direction
            )

    return acceleration


# ============================================================
# PHYSICAL FLOCKING
# ============================================================

def physical_accelerations_from_geometry(
    positions,
    geometry,
):
    acceleration = np.zeros_like(positions)

    ordinary_indices = geometry["indices"]
    ordinary_positions = geometry["positions"]
    ordinary_velocities = geometry["velocities"]
    current_speeds = geometry["speeds"]
    current_headings = geometry["headings"]

    m = len(ordinary_indices)

    if m == 0:
        return acceleration

    speed_restore = (
        (BASE_SPEED - current_speeds)[:, None]
        * current_headings
    )

    if m == 1:
        acceleration[
            ordinary_indices[0]
        ] = (
            SPEED_RESTORE_STRENGTH
            * speed_restore[0]
        )
        return acceleration

    difference = geometry["difference"]
    distances = geometry["distances"]
    nearest = geometry["nearest"]

    # -------------------------
    # heading alignment
    # -------------------------

    neighbour_velocities = (
        ordinary_velocities[nearest]
    )

    neighbour_speeds = np.linalg.norm(
        neighbour_velocities,
        axis=2,
    )

    neighbour_headings = (
        neighbour_velocities
        / np.maximum(
            neighbour_speeds[:, :, None],
            1e-12,
        )
    )

    average_heading = normalize_rows(
        np.mean(
            neighbour_headings,
            axis=1,
        )
    )

    heading_alignment = (
        average_heading
        - current_headings
    )

    # -------------------------
    # weak speed propagation
    # -------------------------

    average_neighbour_speed = np.mean(
        neighbour_speeds,
        axis=1,
    )

    speed_alignment = (
        (
            average_neighbour_speed
            - current_speeds
        )[:, None]
        * current_headings
    )

    # -------------------------
    # local cohesion
    # -------------------------

    local_centres = np.mean(
        ordinary_positions[nearest],
        axis=1,
    )

    cohesion = normalize_rows(
        local_centres
        - ordinary_positions
    )

    # -------------------------
    # separation
    # -------------------------

    valid_distance = np.where(
        np.isfinite(distances),
        distances,
        1.0,
    )

    separation_mask = (
        distances < SEPARATION_RADIUS
    )

    separation_directions = (
        difference
        / np.maximum(
            valid_distance[:, :, None],
            1e-12,
        )
    )

    separation_strength = (
        np.maximum(
            (
                SEPARATION_RADIUS
                - valid_distance
            )
            / SEPARATION_RADIUS,
            0.0,
        )
        ** 2
    )

    hard_mask = (
        distances
        < HARD_COLLISION_RADIUS
    )

    separation_strength[
        hard_mask
    ] *= 4.0

    separation_strength[
        ~separation_mask
    ] = 0.0

    separation = np.sum(
        separation_directions
        * separation_strength[:, :, None],
        axis=1,
    )

    local_acceleration = (
        ALIGNMENT_STRENGTH
        * heading_alignment

        + SPEED_ALIGNMENT_STRENGTH
        * speed_alignment

        + COHESION_STRENGTH
        * cohesion

        + SEPARATION_STRENGTH
        * separation

        + SPEED_RESTORE_STRENGTH
        * speed_restore
    )

    acceleration[
        ordinary_indices
    ] = local_acceleration

    return acceleration


# ============================================================
# SUSTAINED SEMANTIC PURSUIT FORCE
# ============================================================

def add_semantic_pursuit_acceleration(
    acceleration,
    positions,
    velocities,
    pursuits,
    external_tokens,
):
    active = []

    for pursuit in pursuits:
        bird_index = pursuit["bird_index"]
        target_index = pursuit["target_index"]

        if target_index not in external_tokens:
            continue

        to_target = (
            positions[target_index]
            - positions[bird_index]
        )

        target_direction = safe_direction(
            to_target,
            velocities[bird_index],
        )

        current_heading = safe_direction(
            velocities[bird_index],
            INITIAL_DIRECTION,
        )

        angle = pursuit["embedding_angle"]

        turn_acceleration = (
            target_direction
            - current_heading
        )

        acceleration[bird_index] += (
            SEMANTIC_TURN_STRENGTH
            * angle
            * turn_acceleration
        )

        active.append(pursuit)

    return active


# ============================================================
# RESPONDER PURSUIT KINEMATICS
# ============================================================

def enforce_responder_pursuit(
    velocities,
    positions,
    pursuits,
    external_tokens,
    emergency=False,
):
    for pursuit in pursuits:
        bird_index = pursuit["bird_index"]
        target_index = pursuit["target_index"]

        if target_index not in external_tokens:
            continue

        to_target = (
            positions[target_index]
            - positions[bird_index]
        )

        target_direction = safe_direction(
            to_target,
            INITIAL_DIRECTION,
        )

        angle_fraction = float(
            np.clip(
                pursuit["embedding_angle"]
                / np.pi,
                0.0,
                1.0,
            )
        )

        if emergency:
            # Deterministic fallback: straight at the fixed target.
            new_heading = target_direction
            desired_speed = (
                BASE_SPEED
                * RESPONDER_MAX_SPEED_FACTOR
            )
        else:
            current_heading = safe_direction(
                velocities[bird_index],
                target_direction,
            )

            turn_rate = (
                PURSUIT_MIN_TURN_RATE
                + (
                    PURSUIT_MAX_TURN_RATE
                    - PURSUIT_MIN_TURN_RATE
                )
                * angle_fraction
            )

            new_heading = rotate_towards(
                current_heading,
                target_direction,
                turn_rate * DT,
            )

            desired_factor = (
                RESPONDER_MIN_SPEED_FACTOR
                + (
                    RESPONDER_MAX_SPEED_FACTOR
                    - RESPONDER_MIN_SPEED_FACTOR
                )
                * angle_fraction
            )

            desired_speed = (
                BASE_SPEED
                * desired_factor
            )

        current_speed = np.linalg.norm(
            velocities[bird_index]
        )

        if emergency:
            new_speed = desired_speed
        else:
            blend = min(
                1.0,
                RESPONDER_SPEED_RESPONSE * DT,
            )

            new_speed = (
                current_speed
                + (
                    desired_speed
                    - current_speed
                )
                * blend
            )

            new_speed = float(
                np.clip(
                    new_speed,
                    BASE_SPEED
                    * RESPONDER_MIN_SPEED_FACTOR,
                    BASE_SPEED
                    * RESPONDER_MAX_SPEED_FACTOR,
                )
            )

        velocities[bird_index] = (
            new_heading
            * new_speed
        )


# ============================================================
# PHYSICS STEP
# ============================================================

def apply_step(
    positions,
    velocities,
    pursuits,
    bond_array,
    external_tokens,
    emergency_pursuit=False,
):
    # Only external positions need preserving. The previous implementation
    # copied the entire position and velocity arrays every physics step.
    if external_tokens:
        external_indices = np.fromiter(
            external_tokens,
            dtype=np.int64,
            count=len(external_tokens),
        )
        frozen_external_positions = (
            positions[external_indices].copy()
        )
    else:
        external_indices = np.empty(
            0,
            dtype=np.int64,
        )
        frozen_external_positions = None

    # Build the O(N^2) geometry once and share it between local flocking and
    # rescue connectivity.
    geometry = build_ordinary_geometry(
        positions,
        velocities,
        external_tokens,
    )

    acceleration = (
        physical_accelerations_from_geometry(
            positions,
            geometry,
        )
    )

    acceleration += (
        semantic_bond_accelerations(
            positions,
            bond_array,
            external_tokens,
        )
    )

    acceleration += (
        rescue_accelerations_from_geometry(
            positions,
            geometry,
        )
    )

    active_pursuits = (
        add_semantic_pursuit_acceleration(
            acceleration,
            positions,
            velocities,
            pursuits,
            external_tokens,
        )
    )

    # -------------------------
    # acceleration cap
    # -------------------------

    acceleration_magnitude = np.linalg.norm(
        acceleration,
        axis=1,
    )

    too_large = (
        acceleration_magnitude
        > MAX_ACCELERATION
    )

    if np.any(too_large):
        acceleration[too_large] *= (
            MAX_ACCELERATION
            / acceleration_magnitude[too_large]
        )[:, None]

    # -------------------------
    # ordinary velocity update
    # -------------------------

    proposed_velocity = (
        velocities
        + acceleration * DT
    )

    new_velocities = velocities.copy()

    ordinary_indices = geometry["indices"]

    if len(ordinary_indices) > 0:
        ordinary_old_velocity = velocities[
            ordinary_indices
        ]

        ordinary_proposed_velocity = (
            proposed_velocity[
                ordinary_indices
            ]
        )

        old_heading = safe_direction_rows(
            ordinary_old_velocity,
            INITIAL_DIRECTION,
        )

        proposed_heading = safe_direction_rows(
            ordinary_proposed_velocity,
            old_heading,
        )

        new_heading = rotate_towards_rows(
            old_heading,
            proposed_heading,
            MAX_TURN_RATE * DT,
        )

        proposed_speed = np.linalg.norm(
            ordinary_proposed_velocity,
            axis=1,
        )

        new_speed = np.clip(
            proposed_speed,
            MIN_SPEED,
            GENERAL_MAX_SPEED,
        )

        new_velocities[
            ordinary_indices
        ] = (
            new_heading
            * new_speed[:, None]
        )

    if len(external_indices) > 0:
        new_velocities[
            external_indices
        ] = 0.0

    # Direct responders still use the same exact pursuit rule. There are at
    # most three, so keeping this scalar preserves the original behaviour and
    # costs essentially nothing.
    enforce_responder_pursuit(
        new_velocities,
        positions,
        active_pursuits,
        external_tokens,
        emergency=emergency_pursuit,
    )

    # -------------------------
    # position update
    # -------------------------

    new_positions = (
        positions
        + new_velocities * DT
    )

    if len(external_indices) > 0:
        new_positions[
            external_indices
        ] = frozen_external_positions

        new_velocities[
            external_indices
        ] = 0.0

    # -------------------------
    # contact / incorporation
    # -------------------------

    released = update_external_membership(
        new_positions,
        new_velocities,
        external_tokens,
    )

    # -------------------------
    # event-resolution diagnostic
    # -------------------------

    # Heading change was previously recomputed bird-by-bird every timestep,
    # but it is not used by any current settling condition or output field.
    # Removing that diagnostic changes no physics and no event timing.
    mean_heading_change = 0.0

    if len(ordinary_indices) > 0:
        speeds = np.linalg.norm(
            new_velocities[
                ordinary_indices
            ],
            axis=1,
        )

        speed_error = float(
            np.mean(
                np.abs(
                    speeds
                    - BASE_SPEED
                )
            )
        )
    else:
        speed_error = 0.0

    return (
        new_positions,
        new_velocities,
        mean_heading_change,
        speed_error,
        released,
        len(active_pursuits),
    )


# ============================================================
# FRAME
# ============================================================

def make_frame(
    time_value,
    positions,
    velocities,
    birth_index=None,
):
    return {
        "time": float(time_value),
        "active_count": int(len(positions)),
        "birth_index": (
            None
            if birth_index is None
            else int(birth_index)
        ),
        "positions": (
            positions
            .astype(np.float32)
            .tolist()
        ),
        "velocities": (
            velocities
            .astype(np.float32)
            .tolist()
        ),
    }


# ============================================================
# SIMULATE ONE RUN
# ============================================================

def simulate_run(
    run,
    projected_run,
    start_state=None,
    start_index=0,
    checkpoint_after_index=None,
):
    tokens = run["tokens"]

    embeddings = np.asarray(
        [
            token["embedding"]
            for token in tokens
        ],
        dtype=np.float64,
    )

    projected = np.asarray(
        [
            token["projected_3d"]
            for token
            in projected_run["tokens"]
        ],
        dtype=np.float64,
    )

    if start_state is None:
        positions = np.empty(
            (0, 3),
            dtype=np.float64,
        )

        velocities = np.empty(
            (0, 3),
            dtype=np.float64,
        )

        frames = []
        birth_events = []
        event_diagnostics = []

        bonds = []
        external_tokens = set()

        total_time = 0.0
        global_step = 0

    else:
        # Continue from an already-resolved shared prefix. Every mutable
        # simulation array is copied so the two branches remain independent.
        positions = start_state["positions"].copy()
        velocities = start_state["velocities"].copy()

        frames = list(start_state["frames"])
        birth_events = list(start_state["birth_events"])
        event_diagnostics = list(
            start_state["event_diagnostics"]
        )

        bonds = list(start_state["bonds"])
        external_tokens = set(
            start_state["external_tokens"]
        )

        total_time = float(
            start_state["total_time"]
        )
        global_step = int(
            start_state["global_step"]
        )

    bond_array = (
        np.asarray(
            bonds,
            dtype=np.int64,
        ).reshape(-1, 2)
        if bonds
        else np.empty(
            (0, 2),
            dtype=np.int64,
        )
    )

    checkpoint = None

    for new_index in range(
        start_index,
        len(tokens),
    ):
        neighbours = semantic_neighbors(
            embeddings,
            new_index,
        )

        new_position = birth_position(
            new_index,
            positions,
            projected,
            neighbours,
        )

        pursuits = (
            build_semantic_pursuits(
                new_index,
                neighbours,
                embeddings,
            )
            if new_index > 0
            else []
        )

        new_velocity = newborn_velocity(
            new_index
        )

        positions = np.vstack(
            [
                positions,
                new_position,
            ]
        )

        velocities = np.vstack(
            [
                velocities,
                new_velocity,
            ]
        )

        if new_index > 0:
            external_tokens.add(
                int(new_index)
            )

        # Store bonds immediately, but they are inactive while
        # either endpoint is still external.
        for neighbour in neighbours:
            bonds.append(
                (
                    int(new_index),
                    int(neighbour),
                )
            )

        # Bonds only change once per token, so convert them to a compact
        # numeric array here rather than rebuilding it every physics step.
        if bonds:
            bond_array = np.asarray(
                bonds,
                dtype=np.int64,
            ).reshape(-1, 2)

        frames.append(
            make_frame(
                total_time,
                positions,
                velocities,
                birth_index=new_index,
            )
        )

        birth_events.append(
            {
                "token_index": int(new_index),
                "token_id": int(
                    tokens[new_index]["token_id"]
                ),
                "token": tokens[new_index]["token"],
                "time": float(total_time),
                "semantic_neighbors": [
                    int(x)
                    for x in neighbours
                ],
                "embedding_angles": [
                    float(
                        pursuit["embedding_angle"]
                    )
                    for pursuit in pursuits
                ],
            }
        )

        stable_count = 0
        settle_steps = 0
        joined_step = (
            0
            if new_index == 0
            else None
        )
        emergency_used = False
        cooldown_cap_used = False

        while True:
            newborn_external = (
                new_index in external_tokens
            )

            emergency_pursuit = (
                newborn_external
                and settle_steps
                >= EMERGENCY_PURSUIT_AFTER_STEPS
            )

            if emergency_pursuit:
                emergency_used = True

            (
                positions,
                velocities,
                mean_heading_change,
                mean_speed_error,
                released,
                active_pursuit_count,
            ) = apply_step(
                positions,
                velocities,
                pursuits,
                bond_array,
                external_tokens,
                emergency_pursuit=emergency_pursuit,
            )

            if released:
                external_tokens.difference_update(
                    released
                )

                if (
                    new_index in released
                    and joined_step is None
                ):
                    joined_step = settle_steps + 1
                    stable_count = 0

            total_time += DT
            global_step += 1
            settle_steps += 1

            if global_step % RECORD_EVERY == 0:
                frames.append(
                    make_frame(
                        total_time,
                        positions,
                        velocities,
                    )
                )

            newborn_has_joined = (
                new_index == 0
                or new_index not in external_tokens
            )

            if newborn_has_joined:
                stable_now = (
                    mean_speed_error
                    < SPEED_EPSILON
                )

                if stable_now:
                    stable_count += 1
                else:
                    stable_count = 0

                naturally_resolved = (
                    settle_steps
                    >= MIN_SETTLE_STEPS
                    and stable_count
                    >= STABLE_WINDOW
                )

                if naturally_resolved:
                    break

                if joined_step is None:
                    joined_step = settle_steps

                post_join_steps = (
                    settle_steps - joined_step
                )

                if (
                    post_join_steps
                    >= MAX_POST_JOIN_SETTLE_STEPS
                ):
                    cooldown_cap_used = True
                    break

            else:
                # While still external, never count "stable"
                # frames. The semantic event is still active.
                stable_count = 0

        event_diagnostics.append(
            {
                "token_index": int(new_index),
                "settle_steps": int(settle_steps),
                "joined_step": (
                    None
                    if joined_step is None
                    else int(joined_step)
                ),
                "emergency_pursuit_used": bool(
                    emergency_used
                ),
                "cooldown_cap_used": bool(
                    cooldown_cap_used
                ),
                "final_speed_error": float(
                    mean_speed_error
                ),
            }
        )

        if (
            checkpoint_after_index is not None
            and new_index == checkpoint_after_index
        ):
            checkpoint = {
                "positions": positions.copy(),
                "velocities": velocities.copy(),
                "frames": list(frames),
                "birth_events": list(birth_events),
                "event_diagnostics": list(
                    event_diagnostics
                ),
                "bonds": list(bonds),
                "external_tokens": set(
                    external_tokens
                ),
                "total_time": float(
                    total_time
                ),
                "global_step": int(
                    global_step
                ),
            }

        if (
            new_index % 10 == 0
            or new_index == len(tokens) - 1
        ):
            flags = []

            if emergency_used:
                flags.append("emergency-pursuit")

            if cooldown_cap_used:
                flags.append("cooldown-cap")

            flag_text = (
                ""
                if not flags
                else " | " + ",".join(flags)
            )

            print(
                f"  token {new_index + 1}/{len(tokens)}"
                f" | settle {settle_steps}"
                f" | external {len(external_tokens)}"
                f" | speed_err {mean_speed_error:.4f}"
                f" | t={total_time:.1f}s"
                f"{flag_text}"
            )

    # ========================================================
    # FINAL FREE FLIGHT
    # ========================================================

    final_steps = int(4.0 / DT)

    for _ in range(final_steps):
        (
            positions,
            velocities,
            _,
            _,
            released,
            _,
        ) = apply_step(
            positions,
            velocities,
            [],
            bond_array,
            external_tokens,
            emergency_pursuit=False,
        )

        if released:
            external_tokens.difference_update(
                released
            )

        total_time += DT
        global_step += 1

        if global_step % RECORD_EVERY == 0:
            frames.append(
                make_frame(
                    total_time,
                    positions,
                    velocities,
                )
            )

    motion = {
        "run": run["run"],
        "model": run["model"],
        "prompt": run["prompt"],
        "first_divergence": run["first_divergence"],
        "token_count": len(tokens),
        "duration": float(total_time),

        "settings": {
            "dt": DT,
            "base_speed": BASE_SPEED,
            "physical_neighbors": PHYSICAL_NEIGHBOR_COUNT,
            "semantic_neighbors": SEMANTIC_NEIGHBOR_COUNT,
            "alignment_strength": ALIGNMENT_STRENGTH,
            "speed_alignment_strength": SPEED_ALIGNMENT_STRENGTH,
            "cohesion_strength": COHESION_STRENGTH,
            "separation_strength": SEPARATION_STRENGTH,
            "speed_restore_strength": SPEED_RESTORE_STRENGTH,
            "general_max_speed": GENERAL_MAX_SPEED,
            "responder_min_speed_factor": RESPONDER_MIN_SPEED_FACTOR,
            "responder_max_speed_factor": RESPONDER_MAX_SPEED_FACTOR,
            "min_birth_distance": MIN_BIRTH_DISTANCE,
            "max_birth_distance": MAX_BIRTH_DISTANCE,
            "birth_distance_factor": BIRTH_DISTANCE_FACTOR,
            "join_neighbor_count": JOIN_NEIGHBOR_COUNT,
            "join_distance": JOIN_DISTANCE,
            "semantic_turn_strength": SEMANTIC_TURN_STRENGTH,
            "pursuit_min_turn_rate_deg": float(
                np.rad2deg(PURSUIT_MIN_TURN_RATE)
            ),
            "pursuit_max_turn_rate_deg": float(
                np.rad2deg(PURSUIT_MAX_TURN_RATE)
            ),
            "semantic_bond_strength": SEMANTIC_BOND_STRENGTH,
            "semantic_bond_rest_distance": SEMANTIC_BOND_REST_DISTANCE,
            "rescue_strength": RESCUE_COHESION_STRENGTH,
            "connectivity_factor": CONNECTIVITY_DISTANCE_FACTOR,
            "speed_epsilon": SPEED_EPSILON,
            "emergency_pursuit_after_steps": EMERGENCY_PURSUIT_AFTER_STEPS,
            "max_post_join_settle_steps": MAX_POST_JOIN_SETTLE_STEPS,
            "optimized_kernel": True,
        },

        "birds": [
            {
                "index": int(token["index"]),
                "token_id": int(token["token_id"]),
                "token": token["token"],
            }
            for token in tokens
        ],

        "birth_events": birth_events,
        "event_diagnostics": event_diagnostics,
        "frames": frames,
    }


    return motion, checkpoint


# ============================================================
# MAIN
# ============================================================

def common_prefix_length(run_a, run_b):
    """Number of generated token IDs shared by both runs."""
    tokens_a = run_a["tokens"]
    tokens_b = run_b["tokens"]

    limit = min(
        len(tokens_a),
        len(tokens_b),
    )

    index = 0

    while (
        index < limit
        and int(tokens_a[index]["token_id"])
        == int(tokens_b[index]["token_id"])
    ):
        index += 1

    return index


def main():
    run_a = load_json(RUN_A_FILE)
    run_b = load_json(RUN_B_FILE)

    projected_a = load_json(PROJECTED_A_FILE)
    projected_b = load_json(PROJECTED_B_FILE)

    shared_prefix = common_prefix_length(
        run_a,
        run_b,
    )

    if shared_prefix > 0:
        print(
            f"\nShared generated prefix: "
            f"{shared_prefix} tokens. "
            f"It will be simulated only once."
        )

        print("\nSimulating A...")

        motion_a, checkpoint = simulate_run(
            run_a,
            projected_a,
            checkpoint_after_index=(
                shared_prefix - 1
            ),
        )

        if checkpoint is None:
            raise RuntimeError(
                "Failed to capture the shared-prefix checkpoint."
            )

        print(
            "\nSimulating B from the shared-prefix "
            "checkpoint..."
        )

        motion_b, _ = simulate_run(
            run_b,
            projected_b,
            start_state=checkpoint,
            start_index=shared_prefix,
        )

    else:
        print("\nNo shared generated prefix detected.")

        print("\nSimulating A...")
        motion_a, _ = simulate_run(
            run_a,
            projected_a,
        )

        print("\nSimulating B...")
        motion_b, _ = simulate_run(
            run_b,
            projected_b,
        )

    print("\nSaving...")

    with open(
        OUTPUT_A_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            motion_a,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    with open(
        OUTPUT_B_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            motion_b,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    print("\nDone.")
    print(OUTPUT_A_FILE)
    print(OUTPUT_B_FILE)


if __name__ == "__main__":
    main()
