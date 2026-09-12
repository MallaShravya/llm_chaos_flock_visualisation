import json
import shutil
import struct
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np

try:
    import ijson
except ImportError:
    print(
        "This converter uses ijson so it can process the large motion JSON files "
        "without loading them fully into RAM.\n\n"
        "Install it with:\n"
        "    python -m pip install ijson\n"
    )
    raise SystemExit(1)


MAGIC = b"LLMCHS01"
VERSION = 1

# 8s magic, uint32 version, uint32 frame_count, uint32 max_birds, uint32 reserved
HEADER_STRUCT = struct.Struct("<8sIIII")

# float64 time, uint32 active_count, int32 birth_index,
# uint32 float_offset, uint32 reserved
INDEX_STRUCT = struct.Struct("<dIiII")

OUTPUT_DIR = Path("viewer") / "public" / "data"

RUNS = {
    "a": {
        "input": "run_a_semantic_motion_v2.json",
        "meta": "run_a_motion_v2.meta.json",
        "bin": "run_a_motion_v2.bin",
    },
    "b": {
        "input": "run_b_semantic_motion_v2.json",
        "meta": "run_b_motion_v2.meta.json",
        "bin": "run_b_motion_v2.bin",
    },
}


def scalar_value(value):
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return value


def find_input(filename: str) -> Path:
    candidates = [
        Path(filename),
        OUTPUT_DIR / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename}. Looked in:\n"
        f"  {Path(filename).resolve()}\n"
        f"  {(OUTPUT_DIR / filename).resolve()}"
    )


def read_metadata_prefix(path: Path) -> dict:
    """Read viewer metadata and stop before the huge frames array."""
    meta = {}
    settings = {}
    birds = []
    birth_events = []

    current_bird = None
    current_birth = None
    scalar_events = {"string", "number", "boolean", "null"}

    with path.open("rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "frames" and event == "start_array":
                break

            if (
                prefix in {
                    "run",
                    "model",
                    "prompt",
                    "first_divergence",
                    "token_count",
                    "duration",
                }
                and event in scalar_events
            ):
                meta[prefix] = scalar_value(value)
                continue

            if prefix.startswith("settings.") and event in scalar_events:
                key = prefix.split(".", 1)[1]
                settings[key] = scalar_value(value)
                continue

            if prefix == "birds.item" and event == "start_map":
                current_bird = {}
                continue

            if (
                current_bird is not None
                and prefix.startswith("birds.item.")
                and event in scalar_events
            ):
                key = prefix.split(".", 2)[2]
                current_bird[key] = scalar_value(value)
                continue

            if prefix == "birds.item" and event == "end_map":
                birds.append(current_bird)
                current_bird = None
                continue

            if prefix == "birth_events.item" and event == "start_map":
                current_birth = {}
                continue

            if (
                current_birth is not None
                and prefix.startswith("birth_events.item.")
                and event in scalar_events
            ):
                key = prefix.split(".", 2)[2]
                if key in {"token_index", "token_id", "token", "time"}:
                    current_birth[key] = scalar_value(value)
                continue

            if prefix == "birth_events.item" and event == "end_map":
                birth_events.append(current_birth)
                current_birth = None
                continue

    meta["settings"] = settings
    meta["birds"] = birds
    meta["birth_events"] = birth_events
    return meta


def convert_one(label: str, config: dict):
    input_path = find_input(config["input"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = OUTPUT_DIR / config["meta"]
    bin_path = OUTPUT_DIR / config["bin"]

    print(f"\n[{label.upper()}] Reading compact metadata...")
    meta = read_metadata_prefix(input_path)

    index_tmp = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f"llmchaos_{label}_index_",
        suffix=".tmp",
        delete=False,
    )
    payload_tmp = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f"llmchaos_{label}_payload_",
        suffix=".tmp",
        delete=False,
    )

    index_tmp_path = Path(index_tmp.name)
    payload_tmp_path = Path(payload_tmp.name)

    frame_count = 0
    max_birds = 0
    float_offset = 0
    last_time = 0.0

    try:
        print(f"[{label.upper()}] Streaming frames into Float32 binary...")

        with input_path.open("rb") as f:
            for frame in ijson.items(f, "frames.item"):
                time_value = float(frame["time"])
                active_count = int(frame["active_count"])
                birth_index_raw = frame.get("birth_index")
                birth_index = -1 if birth_index_raw is None else int(birth_index_raw)

                positions = np.asarray(frame["positions"], dtype=np.float32)
                velocities = np.asarray(frame["velocities"], dtype=np.float32)

                if positions.shape != (active_count, 3):
                    raise ValueError(
                        f"Frame {frame_count}: positions shape {positions.shape} "
                        f"does not match active_count={active_count}"
                    )

                if velocities.shape != (active_count, 3):
                    raise ValueError(
                        f"Frame {frame_count}: velocities shape {velocities.shape} "
                        f"does not match active_count={active_count}"
                    )

                # Interleave per bird: px, py, pz, vx, vy, vz
                interleaved = np.empty((active_count, 6), dtype=np.float32)
                interleaved[:, 0:3] = positions
                interleaved[:, 3:6] = velocities

                index_tmp.write(
                    INDEX_STRUCT.pack(
                        time_value,
                        active_count,
                        birth_index,
                        float_offset,
                        0,
                    )
                )

                payload_tmp.write(interleaved.tobytes(order="C"))

                float_offset += active_count * 6
                frame_count += 1
                max_birds = max(max_birds, active_count)
                last_time = time_value

                if frame_count % 5000 == 0:
                    print(
                        f"  {frame_count:,} frames converted "
                        f"(latest active birds: {active_count})"
                    )

        index_tmp.flush()
        payload_tmp.flush()
        index_tmp.close()
        payload_tmp.close()

        print(f"[{label.upper()}] Writing final binary...")

        with bin_path.open("wb") as out:
            out.write(
                HEADER_STRUCT.pack(
                    MAGIC,
                    VERSION,
                    frame_count,
                    max_birds,
                    0,
                )
            )

            with index_tmp_path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)

            with payload_tmp_path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)

        meta["duration"] = float(last_time)
        meta["frame_count"] = frame_count
        meta["max_birds"] = max_birds
        meta["binary_file"] = config["bin"]
        meta["binary_format"] = {
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "endianness": "little",
            "header_bytes": HEADER_STRUCT.size,
            "index_entry_bytes": INDEX_STRUCT.size,
            "floats_per_bird": 6,
            "bird_layout": ["px", "py", "pz", "vx", "vy", "vz"],
            "float_type": "float32",
        }

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

        json_mb = input_path.stat().st_size / (1024 * 1024)
        bin_mb = bin_path.stat().st_size / (1024 * 1024)
        meta_mb = meta_path.stat().st_size / (1024 * 1024)

        print(
            f"[{label.upper()}] Done\n"
            f"  source JSON : {json_mb:,.1f} MB\n"
            f"  binary      : {bin_mb:,.1f} MB\n"
            f"  metadata    : {meta_mb:,.2f} MB\n"
            f"  frames      : {frame_count:,}\n"
            f"  max birds   : {max_birds}"
        )

    finally:
        try:
            index_tmp.close()
        except Exception:
            pass

        try:
            payload_tmp.close()
        except Exception:
            pass

        for temp_path in (index_tmp_path, payload_tmp_path):
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def main():
    print(
        "LLM Chaos motion JSON -> compact binary\n"
        "This does NOT alter any trajectory values.\n"
        "The simulator already stored position/velocity as float32; "
        "this writes those same float32 values directly.\n"
    )

    for label, config in RUNS.items():
        convert_one(label, config)

    print(
        "\nConversion complete.\n"
        f"Files are in: {OUTPUT_DIR.resolve()}\n\n"
        "Now replace viewer/src/main.js with the binary viewer version "
        "and start Vite normally."
    )


if __name__ == "__main__":
    main()
