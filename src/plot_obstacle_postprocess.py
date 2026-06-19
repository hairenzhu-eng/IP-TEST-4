import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np


DEFAULT_FIGURE_OUTPUT_DIR = Path("logs/generated_figures")


DEFAULT_DISTANCE_COMPARISON = {
    "prediction_success": Path("logs/prediction_success"),
    "no_prediction_success": Path("logs/no_prediction_success"),
    "no_prediction_failure": Path("logs/no_prediction_failure"),
    "output": None,
    "safe_distance_m": None,
}

DEFAULT_DISTANCE_COMPARISON_FALLBACK = {
    "prediction_success": Path("logs/run_20260617_134958"),
    "no_prediction_success": Path("logs/run_20260617_134809"),
    "no_prediction_failure": Path("logs/run_20260617_135121"),
}


def parse_float(value):
    if value is None:
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    if value == "" or value.lower() in {"none", "nan", "null"}:
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def first_existing(row, names):
    for name in names:
        if name in row:
            return row[name]
    return None


def parse_int(value, default=0):
    value = parse_float(value)
    if np.isfinite(value):
        return int(value)
    return default


def parse_point(value):
    if value is None:
        return None

    if isinstance(value, dict):
        north = parse_float(first_existing(value, ["north", "n", "North(m)", "centre_north_m"]))
        east = parse_float(first_existing(value, ["east", "e", "East(m)", "centre_east_m"]))
        if np.isfinite(north) and np.isfinite(east):
            return (north, east)
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        north = parse_float(value[0])
        east = parse_float(value[1])
        if np.isfinite(north) and np.isfinite(east):
            return (north, east)

    return None


def parse_points(value):
    if not isinstance(value, (list, tuple)):
        return []

    points = []
    for item in value:
        point = parse_point(item)
        if point is not None:
            points.append(point)

    return points


def computed_distance_from_points(own_point, obstacle_point):
    if own_point is None or obstacle_point is None:
        return np.nan

    own_point = np.asarray(own_point, dtype=float)
    obstacle_point = np.asarray(obstacle_point, dtype=float)
    if own_point.shape[0] < 2 or obstacle_point.shape[0] < 2:
        return np.nan

    if not (
        np.isfinite(own_point[0])
        and np.isfinite(own_point[1])
        and np.isfinite(obstacle_point[0])
        and np.isfinite(obstacle_point[1])
    ):
        return np.nan

    return float(np.linalg.norm(own_point[:2] - obstacle_point[:2]))


def read_distance_series_csv(log_path):
    times = []
    distances = []
    start_time_s = np.nan

    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_s = parse_float(first_existing(row, ["TimeFromStart(s)", "TimeFromStart", "elapsed [s]"]))
            if not np.isfinite(start_time_s) and np.isfinite(time_s):
                start_time_s = float(time_s)

            own_point = parse_point(
                {
                    "north": first_existing(row, ["North(m)", "North", "x [m]"]),
                    "east": first_existing(row, ["East(m)", "East", "y [m]"]),
                }
            )
            obstacle_point = parse_point(
                {
                    "north": first_existing(row, ["NearestObstacleNorth(m)", "NearestObstacleNorth"]),
                    "east": first_existing(row, ["NearestObstacleEast(m)", "NearestObstacleEast"]),
                }
            )
            distance = computed_distance_from_points(own_point, obstacle_point)
            if np.isfinite(time_s) and np.isfinite(distance):
                times.append(float(time_s))
                distances.append(distance)

    return normalize_time_series(times, distances, start_time_s=start_time_s)


def nearest_distance_to_obstacles(own_point, obstacles):
    distances = [
        computed_distance_from_points(own_point, obstacle_point)
        for obstacle_point in obstacles
    ]
    distances = [distance for distance in distances if np.isfinite(distance)]
    if not distances:
        return np.nan
    return float(min(distances))


def read_distance_series_obstacle_json(obstacle_dir):
    times = []
    distances = []

    for path in sorted(obstacle_dir.glob("obstacle_*.json")):
        with path.open() as f:
            payload = json.load(f)

        time_s = parse_float(payload.get("t"))
        own_point = parse_point(payload.get("robot_pos"))
        obstacle_points = []

        payload_clusters = payload.get("clusters", [])
        if isinstance(payload_clusters, list):
            for cluster in payload_clusters:
                if not isinstance(cluster, dict):
                    continue
                point = (
                    parse_point(cluster.get("centre_ne_m"))
                    or parse_point(cluster.get("centre_ne"))
                    or parse_point(cluster.get("centroid"))
                )
                if point is not None:
                    obstacle_points.append(point)

        payload_tracks = payload.get("tracks", [])
        if isinstance(payload_tracks, list):
            for track in payload_tracks:
                if not isinstance(track, dict):
                    continue
                point = (
                    parse_point(track.get("position_ne"))
                    or parse_point(track.get("pos_ne"))
                    or parse_point(track.get("state"))
                )
                if point is not None:
                    obstacle_points.append(point)

        distance = nearest_distance_to_obstacles(own_point, obstacle_points)
        if np.isfinite(time_s) and np.isfinite(distance):
            times.append(float(time_s))
            distances.append(distance)

    return normalize_time_series(times, distances)


def normalize_time_series(times, values, start_time_s=np.nan):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if len(times) == 0:
        return times, values

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    if np.isfinite(start_time_s):
        times = times - float(start_time_s)
    else:
        times = times - times[0]
    return times, values


def find_log_in_run_dir(path):
    candidates = [
        candidate
        for candidate in path.rglob("log_*.csv")
        if not candidate.name.endswith("_pseudo_aruco.csv")
    ]
    if not candidates:
        raise FileNotFoundError(f"No log_*.csv files found in {path}")

    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def default_strategy_source(name):
    preferred_path = DEFAULT_DISTANCE_COMPARISON[name]
    if preferred_path.exists():
        return preferred_path

    return DEFAULT_DISTANCE_COMPARISON_FALLBACK[name]


def read_distance_series(source_path):
    source_path = Path(source_path)
    if source_path.is_dir():
        try:
            log_path = find_log_in_run_dir(source_path)
        except FileNotFoundError:
            log_path = None

        if log_path is not None:
            times, distances = read_distance_series_csv(log_path)
            obstacle_dir = matching_obstacle_log_dir(log_path)
            if len(times) > 0 or obstacle_dir is None:
                return times, distances
            return read_distance_series_obstacle_json(obstacle_dir)

        if any(source_path.glob("obstacle_*.json")):
            return read_distance_series_obstacle_json(source_path)

        return np.array([]), np.array([])

    if source_path.suffix.lower() == ".csv":
        times, distances = read_distance_series_csv(source_path)
        if len(times) > 0:
            return times, distances

        obstacle_dir = matching_obstacle_log_dir(source_path)
        if obstacle_dir is not None:
            return read_distance_series_obstacle_json(obstacle_dir)

        return times, distances

    raise ValueError(f"Distance source must be a CSV log or run directory: {source_path}")


def infer_safe_distance(*source_paths, fallback=1.4):
    for source_path in source_paths:
        if source_path is None:
            continue

        source_path = Path(source_path)
        search_dir = source_path if source_path.is_dir() else matching_obstacle_log_dir(source_path)
        if search_dir is None:
            continue

        for path in sorted(search_dir.glob("obstacle_*.json")):
            with path.open() as f:
                payload = json.load(f)
            settings = payload.get("apf_settings", {})
            if isinstance(settings, dict):
                safety_domain_m = parse_float(settings.get("safety_domain_m"))
                if np.isfinite(safety_domain_m) and safety_domain_m > 0.0:
                    return float(safety_domain_m)
                reference_dcpa_m = parse_float(
                    settings.get("reference_dcpa_threshold_m")
                )
                if np.isfinite(reference_dcpa_m) and reference_dcpa_m > 0.0:
                    return float(reference_dcpa_m)
                dcpa_scale = parse_float(settings.get("dcpa_cluster_scale"))
                minimum_radius_m = parse_float(
                    settings.get("obstacle_min_equivalent_radius_m")
                )
                if (
                    np.isfinite(dcpa_scale)
                    and dcpa_scale > 0.0
                    and np.isfinite(minimum_radius_m)
                    and minimum_radius_m > 0.0
                ):
                    return float(dcpa_scale * minimum_radius_m)

    return float(fallback)


def read_csv_log(log_path):
    robot_points = []
    aruco_points = []
    clusters = []
    nearest_only_count = 0

    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            north = parse_float(first_existing(row, ["North(m)", "North"]))
            east = parse_float(first_existing(row, ["East(m)", "East"]))
            if np.isfinite(north) and np.isfinite(east):
                robot_points.append((north, east))

            aruco_north = parse_float(first_existing(row, ["ARUCOSensedNorth(m)", "ARUCOSensedNorth"]))
            aruco_east = parse_float(first_existing(row, ["ARUCOSensedEast(m)", "ARUCOSensedEast"]))
            if np.isfinite(aruco_north) and np.isfinite(aruco_east):
                aruco_points.append((aruco_north, aruco_east))

            cluster_json = first_existing(row, ["ObstacleClusters", "ObstacleClusters(json)"])
            row_clusters = []
            if cluster_json:
                try:
                    decoded = json.loads(cluster_json)
                except json.JSONDecodeError:
                    decoded = []

                if isinstance(decoded, list):
                    row_clusters = decoded

            if row_clusters:
                for cluster in row_clusters:
                    centre = (
                        parse_point(cluster.get("centre_ne_m"))
                        or parse_point(cluster.get("centre_ne"))
                        or parse_point(cluster.get("centroid"))
                    )
                    if centre is None:
                        continue

                    radius = first_finite(
                        cluster.get("equivalent_radius_m"),
                        cluster.get("radius_m"),
                        cluster.get("cluster_radius_m"),
                        half_if_finite(cluster.get("cluster_size_m")),
                        half_if_finite(cluster.get("span_m")),
                    )
                    clusters.append(
                        {
                            "source": "cluster",
                            "row_index": row_index,
                            "centre": centre,
                            "radius": radius,
                            "point_count": parse_int(cluster.get("point_count")),
                            "track_id": cluster.get("track_id"),
                        }
                    )
            else:
                nearest_north = parse_float(first_existing(row, ["NearestObstacleNorth(m)", "NearestObstacleNorth"]))
                nearest_east = parse_float(first_existing(row, ["NearestObstacleEast(m)", "NearestObstacleEast"]))
                if np.isfinite(nearest_north) and np.isfinite(nearest_east):
                    clusters.append(
                        {
                            "source": "nearest",
                            "row_index": row_index,
                            "centre": (nearest_north, nearest_east),
                            "radius": np.nan,
                            "point_count": 0,
                            "track_id": None,
                        }
                    )
                    nearest_only_count += 1

    return robot_points, aruco_points, clusters, nearest_only_count


def first_finite(*values):
    for value in values:
        value = parse_float(value)
        if np.isfinite(value) and value > 0.0:
            return value
    return np.nan


def half_if_finite(value):
    value = parse_float(value)
    if np.isfinite(value) and value > 0.0:
        return 0.5 * value
    return np.nan


def read_obstacle_json_dir(obstacle_dir):
    robot_points = []
    clusters = []
    clouds = []
    track_points = []
    prediction_segments = []

    for snapshot_index, path in enumerate(sorted(obstacle_dir.glob("obstacle_*.json"))):
        with path.open() as f:
            payload = json.load(f)

        robot_pos = parse_point(payload.get("robot_pos"))
        if robot_pos is not None:
            robot_points.append(robot_pos)

        cloud_points = payload.get("cloud", [])
        if not isinstance(cloud_points, list):
            cloud_points = []
        for point in cloud_points:
            cloud_point = parse_point(point)
            if cloud_point is not None:
                clouds.append(cloud_point)

        payload_clusters = payload.get("clusters", [])
        if not isinstance(payload_clusters, list):
            payload_clusters = []
        payload_tracks = payload.get("tracks", [])
        if not isinstance(payload_tracks, list):
            payload_tracks = []

        for cluster in payload_clusters:
            centre = (
                parse_point(cluster.get("centre_ne_m"))
                or parse_point(cluster.get("centre_ne"))
                or parse_point(cluster.get("centroid"))
            )
            if centre is None:
                continue

            radius = first_finite(
                cluster.get("equivalent_radius_m"),
                cluster.get("radius_m"),
                cluster.get("cluster_radius_m"),
                half_if_finite(cluster.get("cluster_size_m")),
                half_if_finite(cluster.get("span_m")),
            )
            clusters.append(
                {
                    "source": "json_cluster",
                    "row_index": len(robot_points),
                    "centre": centre,
                    "radius": radius,
                    "point_count": parse_int(cluster.get("point_count", cluster.get("size"))),
                    "track_id": cluster.get("track_id"),
                }
            )

            predicted_points = parse_points(cluster.get("predicted_trajectory_ne"))
            if len(predicted_points) >= 2 and not payload_tracks:
                prediction_segments.append(
                    {
                        "source": "cluster_prediction",
                        "snapshot_index": snapshot_index,
                        "track_id": cluster.get("track_id"),
                        "points": predicted_points,
                        "radius": radius,
                    }
                )

        for track in payload_tracks:
            position = (
                parse_point(track.get("position_ne"))
                or parse_point(track.get("pos_ne"))
                or parse_point(track.get("state"))
            )
            track_id = track.get("id", track.get("track_id"))
            radius = first_finite(
                track.get("radius_mean_m"),
                track.get("radius_m"),
                track.get("equivalent_radius_m"),
            )
            predicted_points = (
                parse_points(track.get("prediction_ne"))
                or parse_points(track.get("predicted_trajectory_ne"))
            )

            if position is not None:
                track_points.append(
                    {
                        "source": "track",
                        "snapshot_index": snapshot_index,
                        "centre": position,
                        "track_id": track_id,
                        "radius": radius,
                    }
                )

            if len(predicted_points) >= 2:
                prediction_segments.append(
                    {
                        "source": "track_prediction",
                        "snapshot_index": snapshot_index,
                        "track_id": track_id,
                        "points": predicted_points,
                        "radius": radius,
                    }
                )

    return robot_points, clusters, clouds, track_points, prediction_segments


def latest_log_path(log_dir):
    candidates = [
        path
        for path in log_dir.rglob("log_*.csv")
        if not path.name.endswith("_pseudo_aruco.csv")
    ]
    if not candidates:
        raise FileNotFoundError(f"No log_*.csv files found in {log_dir}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def default_output_path(log_path, obstacle_dir, output_dir=DEFAULT_FIGURE_OUTPUT_DIR):
    if log_path is not None:
        return output_dir / f"{log_path.stem}_obstacle_postprocess.png"

    if obstacle_dir is not None:
        return output_dir / f"{obstacle_dir.name}_obstacle_postprocess.png"

    return output_dir / "obstacle_postprocess.png"


def default_log_distance_output_path(log_path, obstacle_dir, output_dir=DEFAULT_FIGURE_OUTPUT_DIR):
    if log_path is not None:
        return output_dir / f"{log_path.stem}_distance.png"

    if obstacle_dir is not None:
        return output_dir / f"{obstacle_dir.name}_distance.png"

    return output_dir / "distance_to_nearest_obstacle.png"


def default_distance_comparison_output_path(output_dir=DEFAULT_FIGURE_OUTPUT_DIR):
    return output_dir / "real_log_distance_comparison.png"


def matching_obstacle_log_dir(log_path):
    if log_path is None:
        return None

    stem = log_path.stem
    if not stem.startswith("log_"):
        return None

    if any(log_path.parent.glob("obstacle_*.json")):
        return log_path.parent

    obstacle_dir = log_path.parent / f"run_{stem[4:]}"
    if obstacle_dir.exists() and any(obstacle_dir.glob("obstacle_*.json")):
        return obstacle_dir

    sibling_obstacle_dir = log_path.parent.parent / f"run_{stem[4:]}"
    if sibling_obstacle_dir.exists() and any(sibling_obstacle_dir.glob("obstacle_*.json")):
        return sibling_obstacle_dir

    return None


def sample_for_circles(clusters, max_circles):
    clusters_with_radius = [
        cluster
        for cluster in clusters
        if np.isfinite(parse_float(cluster.get("radius"))) and parse_float(cluster.get("radius")) > 0.0
    ]
    if len(clusters_with_radius) <= max_circles:
        return clusters_with_radius

    indices = np.linspace(0, len(clusters_with_radius) - 1, max_circles, dtype=int)
    return [clusters_with_radius[index] for index in indices]


def sample_evenly(items, max_items):
    if len(items) <= max_items:
        return items

    indices = np.linspace(0, len(items) - 1, max_items, dtype=int)
    return [items[index] for index in indices]


def append_polyline_segments(northings, eastings, points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return

    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite]
    if len(points) < 2:
        return

    northings.extend(points[:, 0].tolist())
    eastings.extend(points[:, 1].tolist())
    northings.append(np.nan)
    eastings.append(np.nan)


def set_equal_axis_with_padding(ax, padding_ratio=0.08):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_pad = max((x_max - x_min) * padding_ratio, 0.5)
    y_pad = max((y_max - y_min) * padding_ratio, 0.5)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")


def plot_distance_comparison(
    prediction_success_source,
    no_prediction_success_source,
    no_prediction_failure_source,
    output_path,
    safe_distance_m,
):
    scenario_specs = [
        (
            "No prediction avoidance failed",
            no_prediction_failure_source,
            "#d62728",
            "--",
            2.0,
        ),
        (
            "Prediction avoidance succeeded",
            prediction_success_source,
            "#1f77b4",
            "-",
            2.2,
        ),
        (
            "No prediction avoidance succeeded",
            no_prediction_success_source,
            "black",
            "-.",
            2.0,
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    summary_rows = []
    plotted_any = False

    for label, source, color, linestyle, linewidth in scenario_specs:
        times, distances = read_distance_series(source)
        if len(times) == 0:
            raise ValueError(f"No valid distance samples found in {source}")

        ax.plot(
            times,
            distances,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        plotted_any = True
        min_index = int(np.argmin(distances))
        summary_rows.append(
            {
                "label": label,
                "samples": len(times),
                "min_time_s": float(times[min_index]),
                "min_distance_m": float(distances[min_index]),
            }
        )

    if not plotted_any:
        raise ValueError("No distance series were plotted.")

    ax.axhline(
        safe_distance_m,
        color="0.45",
        linestyle="--",
        linewidth=1.5,
        label=f"Safety distance ({safe_distance_m:g} m)",
    )

    ax.set_title("Distance to Nearest Obstacle Ship")
    ax.set_xlabel("Motion time (s)")
    ax.set_ylabel("Distance between own ship and obstacle ship (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.margins(x=0.02)
    y_max = max(ax.get_ylim()[1], safe_distance_m * 1.15)
    ax.set_ylim(bottom=0.0, top=y_max)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return summary_rows


def plot_log_distance(
    source_path,
    output_path,
    safe_distance_m,
    title=None,
):
    times, distances = read_distance_series(source_path)
    if len(times) == 0:
        raise ValueError(f"No valid distance samples found in {source_path}")

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(
        times,
        distances,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        label="Distance to nearest obstacle ship",
    )
    ax.axhline(
        safe_distance_m,
        color="0.45",
        linestyle="--",
        linewidth=1.5,
        label=f"Safety distance ({safe_distance_m:g} m)",
    )

    min_index = int(np.argmin(distances))
    min_time_s = float(times[min_index])
    min_distance_m = float(distances[min_index])
    ax.scatter(
        [min_time_s],
        [min_distance_m],
        s=36,
        color="#d62728",
        zorder=3,
        label=f"Minimum distance ({min_distance_m:.2f} m)",
    )

    ax.set_title(title or "Distance to Nearest Obstacle Ship")
    ax.set_xlabel("Motion time (s)")
    ax.set_ylabel("Distance between own ship and obstacle ship (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.margins(x=0.02)
    y_max = max(ax.get_ylim()[1], safe_distance_m * 1.15, min_distance_m * 1.15)
    ax.set_ylim(bottom=0.0, top=y_max)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return {
        "samples": len(times),
        "min_time_s": min_time_s,
        "min_distance_m": min_distance_m,
    }


def plot_postprocess(
    robot_points,
    aruco_points,
    clusters,
    clouds,
    track_points,
    prediction_segments,
    output_path,
    title,
    max_circles,
    max_predictions,
    show_cloud,
):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)

    if show_cloud and clouds:
        cloud_array = np.asarray(clouds, dtype=float)
        ax.scatter(
            cloud_array[:, 1],
            cloud_array[:, 0],
            s=4,
            color="0.70",
            alpha=0.20,
            linewidths=0,
            label="LiDAR cluster points",
        )

    if robot_points:
        robot_array = np.asarray(robot_points, dtype=float)
        ax.plot(
            robot_array[:, 1],
            robot_array[:, 0],
            color="#1f77b4",
            linewidth=1.8,
            alpha=0.90,
            label="OS trajectory",
        )
        ax.scatter(
            robot_array[:, 1],
            robot_array[:, 0],
            s=10,
            color="#1f77b4",
            alpha=0.55,
            linewidths=0,
            label="OS positions",
        )

    if aruco_points:
        aruco_array = np.asarray(aruco_points, dtype=float)
        ax.scatter(
            aruco_array[:, 1],
            aruco_array[:, 0],
            s=12,
            marker="+",
            color="0.45",
            linewidths=0.8,
            alpha=0.35,
            label="Robot ArUco measurements",
        )

    if clusters:
        cluster_centres = np.asarray([cluster["centre"] for cluster in clusters], dtype=float)
        ax.scatter(
            cluster_centres[:, 1],
            cluster_centres[:, 0],
            s=28,
            marker="x",
            color="#d62728",
            linewidths=1.2,
            alpha=0.85,
            label="LiDAR target positions",
        )

        circle_clusters = sample_for_circles(clusters, max_circles)
        for circle_index, cluster in enumerate(circle_clusters):
            north, east = cluster["centre"]
            radius = parse_float(cluster.get("radius"))
            circle = Circle(
                (east, north),
                radius,
                edgecolor="#d62728",
                facecolor="#d62728",
                alpha=0.08,
                linewidth=0.8,
                label="LiDAR cluster range" if circle_index == 0 else None,
            )
            ax.add_patch(circle)

    if track_points:
        track_centres = np.asarray([track["centre"] for track in track_points], dtype=float)
        ax.scatter(
            track_centres[:, 1],
            track_centres[:, 0],
            s=16,
            marker="o",
            color="#2ca02c",
            edgecolors="none",
            alpha=0.75,
            label="EKF target estimates",
        )

        track_ids = sorted(
            {
                str(track.get("track_id"))
                for track in track_points
                if track.get("track_id") is not None
            }
        )
        for track_id in track_ids:
            points = [
                track["centre"]
                for track in track_points
                if str(track.get("track_id")) == track_id
            ]
            if len(points) < 2:
                continue
            points = np.asarray(points, dtype=float)
            ax.plot(
                points[:, 1],
                points[:, 0],
                color="#2ca02c",
                linewidth=1.0,
                alpha=0.35,
            )

    sampled_predictions = sample_evenly(prediction_segments, max_predictions)
    if sampled_predictions:
        prediction_northings = []
        prediction_eastings = []
        for segment in sampled_predictions:
            append_polyline_segments(
                prediction_northings,
                prediction_eastings,
                segment.get("points", []),
            )

        ax.plot(
            prediction_eastings,
            prediction_northings,
            color="#ff7f0e",
            linewidth=1.0,
            linestyle="--",
            alpha=0.45,
            label="Predicted target trajectories",
        )

    ax.set_title(title)
    ax.set_xlabel("West-East position, East positive (m)")
    ax.set_ylabel("South-North position, North positive (m)")
    ax.grid(True, alpha=0.3)
    set_equal_axis_with_padding(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot postprocessed OS trajectory, LiDAR target detections, and EKF target tracks."
    )
    parser.add_argument("--log", type=Path, help="CSV log path. Defaults to latest logs/log_*.csv.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory used when --log is omitted.")
    parser.add_argument("--obstacle-log-dir", type=Path, help="Optional directory containing obstacle_*.json files.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT_DIR,
        help="Directory for all default generated images.",
    )
    parser.add_argument("--max-circles", type=int, default=250, help="Maximum range circles to draw.")
    parser.add_argument("--max-predictions", type=int, default=250, help="Maximum predicted trajectories to draw; use 0 to hide them.")
    parser.add_argument("--show-cloud", action="store_true", help="Draw logged LiDAR cloud points from obstacle JSON files.")
    parser.add_argument(
        "--prediction-success",
        type=Path,
        help="CSV log or run directory for the prediction-based successful avoidance case.",
    )
    parser.add_argument(
        "--no-prediction-success",
        type=Path,
        help="CSV log or run directory for the no-prediction successful avoidance case.",
    )
    parser.add_argument(
        "--no-prediction-failure",
        type=Path,
        help="CSV log or run directory for the no-prediction failed avoidance case.",
    )
    parser.add_argument("--distance-output", type=Path, help="Output PNG path for the distance comparison plot.")
    parser.add_argument("--safe-distance", type=float, help="Safety distance threshold in metres for the distance plot.")
    parser.add_argument("--distance-only", action="store_true", help="Only draw the three-strategy distance comparison plot.")
    parser.add_argument(
        "--log-distance-output",
        type=Path,
        help="Output PNG path for the single-run distance plot. Defaults to <log_stem>_distance.png.",
    )
    parser.add_argument(
        "--log-distance-only",
        action="store_true",
        help="Only draw the single-run distance plot from --log or --obstacle-log-dir.",
    )
    parser.add_argument(
        "--skip-log-distance",
        action="store_true",
        help="Do not create the single-run distance plot when generating the normal postprocess plot.",
    )
    args = parser.parse_args()
    figure_output_dir = args.output_dir

    using_default_distance_comparison = len(sys.argv) == 1
    if using_default_distance_comparison:
        args.prediction_success = default_strategy_source("prediction_success")
        args.no_prediction_success = default_strategy_source("no_prediction_success")
        args.no_prediction_failure = default_strategy_source("no_prediction_failure")
        args.distance_output = DEFAULT_DISTANCE_COMPARISON["output"]
        if DEFAULT_DISTANCE_COMPARISON["safe_distance_m"] is not None:
            args.safe_distance = DEFAULT_DISTANCE_COMPARISON["safe_distance_m"]
        missing_sources = [
            (name, path)
            for name, path in [
                ("prediction_success", args.prediction_success),
                ("no_prediction_success", args.no_prediction_success),
                ("no_prediction_failure", args.no_prediction_failure),
            ]
            if not Path(path).exists()
        ]
        if missing_sources:
            missing_text = ", ".join(f"{name}={path}" for name, path in missing_sources)
            parser.error(f"Default distance comparison source not found: {missing_text}")
        print("Using default three-strategy distance comparison sources.")

    if args.log_distance_only:
        log_path = args.log
        obstacle_log_dir = args.obstacle_log_dir
        if log_path is None and obstacle_log_dir is None:
            log_path = latest_log_path(args.log_dir)
        if obstacle_log_dir is None:
            obstacle_log_dir = matching_obstacle_log_dir(log_path)

        distance_source = log_path if log_path is not None else obstacle_log_dir
        safe_distance_m = (
            float(args.safe_distance)
            if args.safe_distance is not None
            else infer_safe_distance(distance_source)
        )
        output_path = args.log_distance_output or default_log_distance_output_path(
            log_path,
            obstacle_log_dir,
            figure_output_dir,
        )
        summary = plot_log_distance(
            source_path=distance_source,
            output_path=output_path,
            safe_distance_m=safe_distance_m,
            title=f"Distance to Nearest Obstacle Ship: {Path(distance_source).name}",
        )
        print(f"Saved {output_path}")
        print(
            f"Distance samples: {summary['samples']}; "
            f"minimum={summary['min_distance_m']:.3f} m at {summary['min_time_s']:.2f} s"
        )
        return

    distance_sources = [
        args.prediction_success,
        args.no_prediction_success,
        args.no_prediction_failure,
    ]
    has_distance_sources = any(source is not None for source in distance_sources)
    if args.distance_only or args.distance_output is not None or has_distance_sources:
        if not all(source is not None for source in distance_sources):
            parser.error(
                "--prediction-success, --no-prediction-success, and --no-prediction-failure "
                "must be provided together for the distance comparison plot."
            )

        safe_distance_m = (
            float(args.safe_distance)
            if args.safe_distance is not None
            else infer_safe_distance(*distance_sources)
        )
        distance_output = args.distance_output or default_distance_comparison_output_path(figure_output_dir)
        summary_rows = plot_distance_comparison(
            prediction_success_source=args.prediction_success,
            no_prediction_success_source=args.no_prediction_success,
            no_prediction_failure_source=args.no_prediction_failure,
            output_path=distance_output,
            safe_distance_m=safe_distance_m,
        )
        print(f"Saved {distance_output}")
        for summary in summary_rows:
            print(
                f"{summary['label']}: samples={summary['samples']}, "
                f"minimum={summary['min_distance_m']:.3f} m at {summary['min_time_s']:.2f} s"
            )

        if args.distance_only or (args.log is None and args.obstacle_log_dir is None):
            return

    log_path = args.log
    if log_path is None and args.obstacle_log_dir is None:
        log_path = latest_log_path(args.log_dir)
    if args.obstacle_log_dir is None:
        args.obstacle_log_dir = matching_obstacle_log_dir(log_path)

    robot_points = []
    aruco_points = []
    clusters = []
    clouds = []
    track_points = []
    prediction_segments = []
    nearest_only_count = 0

    if log_path is not None:
        csv_robot_points, aruco_points, csv_clusters, nearest_only_count = read_csv_log(log_path)
        robot_points.extend(csv_robot_points)
        clusters.extend(csv_clusters)

    if args.obstacle_log_dir is not None:
        print(f"Using obstacle JSON dir: {args.obstacle_log_dir}")
        (
            json_robot_points,
            json_clusters,
            json_clouds,
            json_track_points,
            json_prediction_segments,
        ) = read_obstacle_json_dir(args.obstacle_log_dir)
        if not robot_points:
            robot_points.extend(json_robot_points)
        clusters.extend(json_clusters)
        clouds.extend(json_clouds)
        track_points.extend(json_track_points)
        prediction_segments.extend(json_prediction_segments)

    if not robot_points:
        raise ValueError("No robot position points found in the selected logs.")

    if not clusters and not track_points:
        raise ValueError("No obstacle cluster, target track, or nearest-obstacle points found in the selected logs.")

    output_path = args.output or default_output_path(
        log_path,
        args.obstacle_log_dir,
        figure_output_dir,
    )
    title_parts = []
    if log_path is not None:
        title_parts.append(log_path.name)
    if args.obstacle_log_dir is not None:
        title_parts.append(args.obstacle_log_dir.name)
    title = "Postprocessed Trajectory Tracking: " + " + ".join(title_parts)

    plot_postprocess(
        robot_points=robot_points,
        aruco_points=aruco_points,
        clusters=clusters,
        clouds=clouds,
        track_points=track_points,
        prediction_segments=prediction_segments,
        output_path=output_path,
        title=title,
        max_circles=max(args.max_circles, 1),
        max_predictions=max(args.max_predictions, 0),
        show_cloud=args.show_cloud,
    )

    radius_count = sum(
        1
        for cluster in clusters
        if np.isfinite(parse_float(cluster.get("radius"))) and parse_float(cluster.get("radius")) > 0.0
    )
    print(f"Saved {output_path}")
    print(f"Robot points: {len(robot_points)}")
    print(f"Obstacle centre points: {len(clusters)}")
    print(f"Obstacle ranges: {radius_count}")
    print(f"EKF target estimates: {len(track_points)}")
    print(f"Predicted trajectories: {len(prediction_segments)}")
    if nearest_only_count and radius_count == 0:
        print("This CSV is an old format log, so only nearest obstacle centres were available.")

    if not args.skip_log_distance:
        distance_source = log_path if log_path is not None else args.obstacle_log_dir
        safe_distance_m = (
            float(args.safe_distance)
            if args.safe_distance is not None
            else infer_safe_distance(distance_source)
        )
        distance_output_path = args.log_distance_output or default_log_distance_output_path(
            log_path,
            args.obstacle_log_dir,
            figure_output_dir,
        )
        try:
            summary = plot_log_distance(
                source_path=distance_source,
                output_path=distance_output_path,
                safe_distance_m=safe_distance_m,
                title=f"Distance to Nearest Obstacle Ship: {Path(distance_source).name}",
            )
        except ValueError as exc:
            print(f"Skipped distance plot: {exc}")
        else:
            print(f"Saved {distance_output_path}")
            print(
                f"Distance samples: {summary['samples']}; "
                f"minimum={summary['min_distance_m']:.3f} m at {summary['min_time_s']:.2f} s"
            )


if __name__ == "__main__":
    main()
