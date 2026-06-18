"""Plot the latest matched EKF and non-EKF obstacle-avoidance runs.

The script scans Webots run directories under ``logs`` and selects:

1. The latest run with obstacle EKF prediction enabled.
2. The latest compatible run with obstacle EKF prediction disabled.

Either run may be successful or failed. A run fails only when the own-ship
body overlaps a clustered obstacle:

``centre distance < own-ship radius + clustered-obstacle radius``.

The configured safety distance does not determine the outcome.

Historical logs do not contain the Webots world filename. Two runs are
therefore treated as the same environment only when their initial own-ship
positions, APF settings, and nearest-obstacle world trajectories match.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_LAPTOP_PATH = PROJECT_ROOT / "src" / "laptop.py"
DEFAULT_OUTPUT_DIR = DEFAULT_LOGS_DIR / "generated_figures"

TIME_COLUMNS = ("TimeFromStart(s)", "TimeFromStart", "elapsed [s]")
OWN_NORTH_COLUMNS = ("North(m)", "North", "x [m]")
OWN_EAST_COLUMNS = ("East(m)", "East", "y [m]")
OBSTACLE_NORTH_COLUMNS = (
    "NearestObstacleNorth(m)",
    "NearestObstacleNorth",
)
OBSTACLE_EAST_COLUMNS = (
    "NearestObstacleEast(m)",
    "NearestObstacleEast",
)

MATCHED_SETTING_KEYS = (
    "own_equivalent_radius_m",
    "safety_domain_m",
    "collision_horizon_s",
    "prediction_dt_s",
    "constant_descent_speed_m_s",
    "obstacle_prediction_horizon_s",
    "obstacle_prediction_step_s",
    "dynamic_speed_enter_m_s",
    "dynamic_speed_exit_m_s",
)


@dataclass
class RunRecord:
    run_dir: Path
    log_path: Path
    run_time: datetime
    ekf_enabled: bool
    safe_distance_m: float
    time_s: np.ndarray
    distance_m: np.ndarray
    obstacle_position_ne_m: np.ndarray
    collision_time_s: np.ndarray
    collision_distance_m: np.ndarray
    collision_boundary_m: np.ndarray
    initial_own_position_ne_m: np.ndarray
    matched_settings: dict[str, object]

    @property
    def minimum_distance_m(self) -> float:
        return float(np.min(self.distance_m))

    @property
    def collision_clearance_m(self) -> np.ndarray:
        return self.collision_distance_m - self.collision_boundary_m

    @property
    def minimum_collision_clearance_m(self) -> float:
        return float(np.min(self.collision_clearance_m))

    @property
    def collision_boundary_at_closest_clearance_m(self) -> float:
        index = int(np.argmin(self.collision_clearance_m))
        return float(self.collision_boundary_m[index])

    def succeeded(self) -> bool:
        return self.minimum_collision_clearance_m >= 0.0


@dataclass
class EnvironmentMatch:
    median_error_m: float
    percentile_90_error_m: float
    overlap_duration_s: float


def parse_float(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return np.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def first_existing(row: dict[str, str], names: tuple[str, ...]) -> object:
    for name in names:
        if name in row:
            return row[name]
    return None


def values_equal(left: object, right: object) -> bool:
    left_number = parse_float(left)
    right_number = parse_float(right)
    if np.isfinite(left_number) and np.isfinite(right_number):
        return bool(np.isclose(left_number, right_number))
    return left == right


def find_csv_log(run_dir: Path) -> Path:
    candidates = [
        path
        for path in run_dir.glob("log_*.csv")
        if not path.name.endswith("_pseudo_aruco.csv")
    ]
    if not candidates:
        raise FileNotFoundError("No primary log_*.csv file")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def is_webots_run(run_dir: Path) -> bool:
    return any(run_dir.glob("log_*_pseudo_aruco.csv"))


def parse_run_time(run_dir: Path) -> datetime:
    try:
        return datetime.strptime(run_dir.name, "run_%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.fromtimestamp(run_dir.stat().st_mtime)


def read_csv_series(
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    times = []
    distances = []
    obstacle_positions = []
    first_log_time_s = np.nan
    initial_own_position = None

    with log_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")

        required_groups = (
            TIME_COLUMNS,
            OWN_NORTH_COLUMNS,
            OWN_EAST_COLUMNS,
            OBSTACLE_NORTH_COLUMNS,
            OBSTACLE_EAST_COLUMNS,
        )
        missing = [
            names[0]
            for names in required_groups
            if not any(name in reader.fieldnames for name in names)
        ]
        if missing:
            raise ValueError(
                "Missing columns required for distance calculation: "
                + ", ".join(missing)
            )

        for row in reader:
            time_s = parse_float(first_existing(row, TIME_COLUMNS))
            own_north = parse_float(first_existing(row, OWN_NORTH_COLUMNS))
            own_east = parse_float(first_existing(row, OWN_EAST_COLUMNS))

            if np.isfinite(time_s) and not np.isfinite(first_log_time_s):
                first_log_time_s = time_s
            if (
                initial_own_position is None
                and np.isfinite(own_north)
                and np.isfinite(own_east)
            ):
                initial_own_position = np.array([own_north, own_east], dtype=float)

            obstacle_north = parse_float(
                first_existing(row, OBSTACLE_NORTH_COLUMNS)
            )
            obstacle_east = parse_float(first_existing(row, OBSTACLE_EAST_COLUMNS))
            values = (
                time_s,
                own_north,
                own_east,
                obstacle_north,
                obstacle_east,
            )
            if not all(np.isfinite(value) for value in values):
                continue

            times.append(float(time_s))
            obstacle_positions.append([obstacle_north, obstacle_east])
            distances.append(
                float(
                    np.hypot(
                        own_north - obstacle_north,
                        own_east - obstacle_east,
                    )
                )
            )

    if not times:
        raise ValueError("No valid own-ship and nearest-obstacle position samples")
    if initial_own_position is None:
        raise ValueError("No valid initial own-ship position")

    time_array = np.asarray(times, dtype=float)
    distance_array = np.asarray(distances, dtype=float)
    obstacle_array = np.asarray(obstacle_positions, dtype=float)
    order = np.argsort(time_array)
    time_array = time_array[order]
    distance_array = distance_array[order]
    obstacle_array = obstacle_array[order]
    time_array -= (
        float(first_log_time_s)
        if np.isfinite(first_log_time_s)
        else time_array[0]
    )
    return (
        time_array,
        distance_array,
        obstacle_array,
        initial_own_position,
        float(first_log_time_s),
    )


def read_snapshot_collision_series(
    run_dir: Path,
    first_log_time_s: float,
    own_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = []
    distances = []
    boundaries = []

    for snapshot_path in sorted(run_dir.glob("obstacle_*.json")):
        with snapshot_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)

        time_s = parse_float(payload.get("t"))
        robot_position = np.asarray(
            payload.get("robot_pos", [np.nan, np.nan]),
            dtype=float,
        ).reshape(2)
        clusters = payload.get("clusters", [])
        if (
            not np.isfinite(time_s)
            or not np.isfinite(robot_position).all()
            or not isinstance(clusters, list)
        ):
            continue

        candidates = []
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            centre_ne = np.asarray(
                cluster.get("centre_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            obstacle_radius_m = parse_float(
                cluster.get("equivalent_radius_m")
            )
            if (
                not np.isfinite(centre_ne).all()
                or not np.isfinite(obstacle_radius_m)
                or obstacle_radius_m <= 0.0
            ):
                continue

            centre_distance_m = float(
                np.linalg.norm(robot_position - centre_ne)
            )
            collision_boundary_m = own_radius_m + obstacle_radius_m
            candidates.append(
                (
                    centre_distance_m - collision_boundary_m,
                    centre_distance_m,
                    collision_boundary_m,
                )
            )

        if not candidates:
            continue

        _, centre_distance_m, collision_boundary_m = min(
            candidates,
            key=lambda candidate: candidate[0],
        )
        times.append(float(time_s) - first_log_time_s)
        distances.append(centre_distance_m)
        boundaries.append(collision_boundary_m)

    if not times:
        raise ValueError("No valid clustered-obstacle collision samples")

    time_array = np.asarray(times, dtype=float)
    distance_array = np.asarray(distances, dtype=float)
    boundary_array = np.asarray(boundaries, dtype=float)
    order = np.argsort(time_array)
    return (
        time_array[order],
        distance_array[order],
        boundary_array[order],
    )


def read_snapshot_metadata(
    run_dir: Path,
) -> tuple[bool, float, float, dict[str, object]]:
    snapshot_paths = sorted(run_dir.glob("obstacle_*.json"))
    if not snapshot_paths:
        raise ValueError("No obstacle_*.json snapshots")

    sample_indices = sorted({0, len(snapshot_paths) // 2, len(snapshot_paths) - 1})
    sampled_settings = []
    sampled_dbscan = []
    for index in sample_indices:
        with snapshot_paths[index].open(encoding="utf-8") as stream:
            payload = json.load(stream)
        settings = payload.get("apf_settings", {})
        if not isinstance(settings, dict):
            settings = {}
        sampled_settings.append(settings)
        dbscan = payload.get("dbscan", {})
        sampled_dbscan.append(dbscan if isinstance(dbscan, dict) else {})

    ekf_values = [
        settings.get("obstacle_ekf_prediction_enabled")
        for settings in sampled_settings
        if "obstacle_ekf_prediction_enabled" in settings
    ]
    if not ekf_values:
        raise ValueError(
            "EKF state is not recorded; old logs cannot be classified safely"
        )
    if any(bool(value) != bool(ekf_values[0]) for value in ekf_values[1:]):
        raise ValueError("EKF state changes within the run")

    safety_values = [
        parse_float(settings.get("safety_domain_m"))
        for settings in sampled_settings
    ]
    if not all(np.isfinite(value) and value > 0.0 for value in safety_values):
        raise ValueError("No valid safety_domain_m value")
    if any(
        not np.isclose(value, safety_values[0]) for value in safety_values[1:]
    ):
        raise ValueError("Safety distance changes within the run")

    own_radius_values = [
        parse_float(settings.get("own_equivalent_radius_m"))
        for settings in sampled_settings
    ]
    if not all(
        np.isfinite(value) and value > 0.0 for value in own_radius_values
    ):
        raise ValueError("No valid own_equivalent_radius_m value")
    if any(
        not np.isclose(value, own_radius_values[0])
        for value in own_radius_values[1:]
    ):
        raise ValueError("Own-ship radius changes within the run")

    matched_settings = {
        key: sampled_settings[0].get(key) for key in MATCHED_SETTING_KEYS
    }
    matched_settings["dbscan_eps_m"] = sampled_dbscan[0].get("eps_m")
    matched_settings["dbscan_min_samples"] = sampled_dbscan[0].get("min_samples")
    return (
        bool(ekf_values[0]),
        float(safety_values[0]),
        float(own_radius_values[0]),
        matched_settings,
    )


def load_run(run_dir: Path) -> RunRecord:
    if not is_webots_run(run_dir):
        raise ValueError("Not a Webots run")

    log_path = find_csv_log(run_dir)
    (
        ekf_enabled,
        safe_distance_m,
        own_radius_m,
        matched_settings,
    ) = read_snapshot_metadata(run_dir)
    (
        time_s,
        distance_m,
        obstacle_position_ne_m,
        initial_own_position_ne_m,
        first_log_time_s,
    ) = read_csv_series(log_path)
    (
        collision_time_s,
        collision_distance_m,
        collision_boundary_m,
    ) = read_snapshot_collision_series(
        run_dir=run_dir,
        first_log_time_s=first_log_time_s,
        own_radius_m=own_radius_m,
    )
    return RunRecord(
        run_dir=run_dir.resolve(),
        log_path=log_path.resolve(),
        run_time=parse_run_time(run_dir),
        ekf_enabled=ekf_enabled,
        safe_distance_m=safe_distance_m,
        time_s=time_s,
        distance_m=distance_m,
        obstacle_position_ne_m=obstacle_position_ne_m,
        collision_time_s=collision_time_s,
        collision_distance_m=collision_distance_m,
        collision_boundary_m=collision_boundary_m,
        initial_own_position_ne_m=initial_own_position_ne_m,
        matched_settings=matched_settings,
    )


def scan_runs(logs_dir: Path) -> tuple[list[RunRecord], list[tuple[Path, str]]]:
    records = []
    skipped = []
    run_dirs = sorted(
        (path for path in logs_dir.glob("run_*") if path.is_dir()),
        key=parse_run_time,
        reverse=True,
    )
    for run_dir in run_dirs:
        try:
            records.append(load_run(run_dir))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            skipped.append((run_dir, str(exc)))
    return records, skipped


def settings_match(left: RunRecord, right: RunRecord) -> bool:
    keys = set(left.matched_settings) | set(right.matched_settings)
    return all(
        values_equal(
            left.matched_settings.get(key),
            right.matched_settings.get(key),
        )
        for key in keys
    )


def compare_environments(
    left: RunRecord,
    right: RunRecord,
    minimum_overlap_s: float,
) -> EnvironmentMatch | None:
    if not settings_match(left, right):
        return None
    if not np.isclose(left.safe_distance_m, right.safe_distance_m):
        return None
    if (
        np.linalg.norm(
            left.initial_own_position_ne_m - right.initial_own_position_ne_m
        )
        > 0.5
    ):
        return None

    overlap_start_s = max(float(left.time_s[0]), float(right.time_s[0]))
    overlap_end_s = min(float(left.time_s[-1]), float(right.time_s[-1]))
    overlap_duration_s = overlap_end_s - overlap_start_s
    if overlap_duration_s < minimum_overlap_s:
        return None

    sample_count = max(50, min(300, len(left.time_s), len(right.time_s)))
    common_time_s = np.linspace(
        overlap_start_s,
        overlap_end_s,
        sample_count,
    )
    left_positions = np.column_stack(
        [
            np.interp(
                common_time_s,
                left.time_s,
                left.obstacle_position_ne_m[:, axis],
            )
            for axis in range(2)
        ]
    )
    right_positions = np.column_stack(
        [
            np.interp(
                common_time_s,
                right.time_s,
                right.obstacle_position_ne_m[:, axis],
            )
            for axis in range(2)
        ]
    )
    errors_m = np.linalg.norm(left_positions - right_positions, axis=1)
    return EnvironmentMatch(
        median_error_m=float(np.median(errors_m)),
        percentile_90_error_m=float(np.percentile(errors_m, 90.0)),
        overlap_duration_s=float(overlap_duration_s),
    )


def select_latest_pair(
    records: list[RunRecord],
    minimum_overlap_s: float,
    median_tolerance_m: float,
    percentile_90_tolerance_m: float,
) -> tuple[RunRecord, RunRecord, EnvironmentMatch]:
    candidates = []
    ekf_records = [record for record in records if record.ekf_enabled]
    no_ekf_records = [record for record in records if not record.ekf_enabled]

    for ekf_record in ekf_records:
        for no_ekf_record in no_ekf_records:
            match = compare_environments(
                ekf_record,
                no_ekf_record,
                minimum_overlap_s=minimum_overlap_s,
            )
            if match is None:
                continue
            if match.median_error_m > median_tolerance_m:
                continue
            if match.percentile_90_error_m > percentile_90_tolerance_m:
                continue
            candidates.append((ekf_record, no_ekf_record, match))

    if not candidates:
        raise ValueError(
            "No matched pair was found. "
            f"Scanned {len(records)} classified Webots runs. Run the same "
            "Webots world once with EKF enabled and once with EKF disabled, "
            "keeping the APF and safety settings unchanged."
        )

    candidates.sort(
        key=lambda item: (
            min(item[0].run_time, item[1].run_time),
            max(item[0].run_time, item[1].run_time),
            -item[2].median_error_m,
        ),
        reverse=True,
    )
    return candidates[0]


def read_laptop_ekf_switch(laptop_path: Path) -> bool:
    """Read the module-level switch without importing laptop.py."""
    tree = ast.parse(
        laptop_path.read_text(encoding="utf-8"),
        filename=str(laptop_path),
    )
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name)
            and target.id == "ENABLE_OBSTACLE_EKF_PREDICTION"
            for target in targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, bool):
            return value
        break
    raise ValueError(
        f"Could not read ENABLE_OBSTACLE_EKF_PREDICTION from {laptop_path}"
    )


def outcome_text(record: RunRecord) -> str:
    return "succeeded" if record.succeeded() else "failed"


def default_output_path(
    ekf_record: RunRecord,
    no_ekf_record: RunRecord,
) -> Path:
    filename = (
        f"{ekf_record.log_path.stem}__{no_ekf_record.log_path.stem}.png"
    )
    return DEFAULT_OUTPUT_DIR / filename


def plot_pair(
    ekf_record: RunRecord,
    no_ekf_record: RunRecord,
    output_path: Path,
) -> None:
    ekf_succeeded = ekf_record.succeeded()
    no_ekf_succeeded = no_ekf_record.succeeded()
    ekf_linestyle = "-" if ekf_succeeded else "--"
    no_ekf_color = "black" if no_ekf_succeeded else "#d62728"
    no_ekf_linestyle = "-." if no_ekf_succeeded else "--"

    fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=180)
    ax.plot(
        ekf_record.time_s,
        ekf_record.distance_m,
        color="#1f77b4",
        linestyle=ekf_linestyle,
        linewidth=2.3,
        label=(
            "EKF prediction avoidance "
            + outcome_text(ekf_record)
        ),
    )
    ax.plot(
        no_ekf_record.time_s,
        no_ekf_record.distance_m,
        color=no_ekf_color,
        linestyle=no_ekf_linestyle,
        linewidth=2.0,
        label=(
            "No EKF prediction avoidance "
            + outcome_text(no_ekf_record)
        ),
    )
    ax.plot(
        ekf_record.collision_time_s,
        ekf_record.collision_boundary_m,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="EKF clustered-obstacle boundary",
    )
    ax.plot(
        no_ekf_record.collision_time_s,
        no_ekf_record.collision_boundary_m,
        color=no_ekf_color,
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="No-EKF clustered-obstacle boundary",
    )
    ax.set_title("Distance to the Nearest Obstacle Ship")
    ax.set_xlabel("Motion time (s)")
    ax.set_ylabel("Distance between own ship and obstacle ship (m)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_run_summary(records: list[RunRecord]) -> None:
    print("Classified Webots runs, newest first:")
    for record in records:
        print(
            f"  {record.run_dir.name}: EKF={record.ekf_enabled}, "
            f"minimum={record.minimum_distance_m:.3f} m, "
            f"minimum_cluster_clearance="
            f"{record.minimum_collision_clearance_m:.3f} m, "
            f"outcome={outcome_text(record)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Webots logs and plot the latest EKF run against a matching "
            "non-EKF run from the same environment."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory containing run_* log directories",
    )
    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=10.0,
        help="Minimum obstacle-trajectory overlap in seconds",
    )
    parser.add_argument(
        "--environment-median-tolerance",
        type=float,
        default=0.75,
        help="Maximum median obstacle-trajectory difference in metres",
    )
    parser.add_argument(
        "--environment-p90-tolerance",
        type=float,
        default=1.5,
        help="Maximum 90th-percentile trajectory difference in metres",
    )
    parser.add_argument(
        "--laptop",
        type=Path,
        default=DEFAULT_LAPTOP_PATH,
        help="laptop.py containing ENABLE_OBSTACLE_EKF_PREDICTION",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path; by default, the filename is built from the "
            "two selected log filenames"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.minimum_overlap <= 0.0:
        raise ValueError("The minimum overlap must be greater than zero.")

    laptop_ekf_enabled = read_laptop_ekf_switch(args.laptop)
    print(
        "Current laptop.py setting: ENABLE_OBSTACLE_EKF_PREDICTION="
        f"{laptop_ekf_enabled}"
    )

    records, skipped = scan_runs(args.logs_dir)
    print_run_summary(records)
    print(f"Skipped {len(skipped)} unclassifiable or non-Webots runs.")

    try:
        ekf_record, no_ekf_record, environment_match = select_latest_pair(
            records=records,
            minimum_overlap_s=args.minimum_overlap,
            median_tolerance_m=args.environment_median_tolerance,
            percentile_90_tolerance_m=args.environment_p90_tolerance,
        )
    except ValueError as exc:
        raise SystemExit(f"Selection stopped: {exc}") from None

    output_path = (
        default_output_path(ekf_record, no_ekf_record)
        if args.output is None
        else args.output
    )

    print(f"Selected EKF run: {ekf_record.run_dir}")
    print(f"Selected non-EKF run: {no_ekf_record.run_dir}")
    print(
        "Environment trajectory match: "
        f"median={environment_match.median_error_m:.3f} m, "
        f"p90={environment_match.percentile_90_error_m:.3f} m, "
        f"overlap={environment_match.overlap_duration_s:.1f} s"
    )
    print(
        f"EKF outcome: {outcome_text(ekf_record)}, "
        f"minimum={ekf_record.minimum_distance_m:.3f} m, "
        f"minimum cluster clearance="
        f"{ekf_record.minimum_collision_clearance_m:.3f} m"
    )
    print(
        f"Non-EKF outcome: {outcome_text(no_ekf_record)}, "
        f"minimum={no_ekf_record.minimum_distance_m:.3f} m, "
        f"minimum cluster clearance="
        f"{no_ekf_record.minimum_collision_clearance_m:.3f} m"
    )

    plot_pair(
        ekf_record=ekf_record,
        no_ekf_record=no_ekf_record,
        output_path=output_path,
    )
    print(f"Figure saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
