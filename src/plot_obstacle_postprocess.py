import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np


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

    for path in sorted(obstacle_dir.glob("obstacle_*.json")):
        with path.open() as f:
            payload = json.load(f)

        robot_pos = parse_point(payload.get("robot_pos"))
        if robot_pos is not None:
            robot_points.append(robot_pos)

        for point in payload.get("cloud", []):
            cloud_point = parse_point(point)
            if cloud_point is not None:
                clouds.append(cloud_point)

        for cluster in payload.get("clusters", []):
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

    return robot_points, clusters, clouds


def latest_log_path(log_dir):
    candidates = [
        path
        for path in log_dir.glob("log_*.csv")
        if not path.name.endswith("_pseudo_aruco.csv")
    ]
    if not candidates:
        raise FileNotFoundError(f"No log_*.csv files found in {log_dir}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def default_output_path(log_path, obstacle_dir):
    if log_path is not None:
        return log_path.with_name(f"{log_path.stem}_obstacle_postprocess.png")

    return obstacle_dir / f"{obstacle_dir.name}_obstacle_postprocess.png"


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


def plot_postprocess(
    robot_points,
    aruco_points,
    clusters,
    clouds,
    output_path,
    title,
    max_circles,
    show_cloud,
):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)

    if show_cloud and clouds:
        cloud_array = np.asarray(clouds, dtype=float)
        ax.scatter(
            cloud_array[:, 1],
            cloud_array[:, 0],
            s=4,
            color="0.75",
            alpha=0.25,
            linewidths=0,
            label="LiDAR cloud",
        )

    if robot_points:
        robot_array = np.asarray(robot_points, dtype=float)
        ax.plot(robot_array[:, 1], robot_array[:, 0], color="#1f77b4", linewidth=1.8, label="Robot path")
        ax.scatter(robot_array[:, 1], robot_array[:, 0], s=8, color="#1f77b4", alpha=0.55, label="Robot positions")

    if aruco_points:
        aruco_array = np.asarray(aruco_points, dtype=float)
        ax.scatter(
            aruco_array[:, 1],
            aruco_array[:, 0],
            s=24,
            marker="x",
            color="#2ca02c",
            linewidths=1.0,
            alpha=0.8,
            label="ArUco positions",
        )

    if clusters:
        cluster_centres = np.asarray([cluster["centre"] for cluster in clusters], dtype=float)
        ax.scatter(
            cluster_centres[:, 1],
            cluster_centres[:, 0],
            s=22,
            marker="x",
            color="#d62728",
            linewidths=1.0,
            alpha=0.75,
            label="Obstacle cluster centres",
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
                alpha=0.10,
                linewidth=0.9,
                label="Obstacle cluster range" if circle_index == 0 else None,
            )
            ax.add_patch(circle)

    ax.set_title(title)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot robot positions and DBSCAN obstacle cluster ranges from logs."
    )
    parser.add_argument("--log", type=Path, help="CSV log path. Defaults to latest logs/log_*.csv.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory used when --log is omitted.")
    parser.add_argument("--obstacle-log-dir", type=Path, help="Optional directory containing obstacle_*.json files.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--max-circles", type=int, default=250, help="Maximum range circles to draw.")
    parser.add_argument("--show-cloud", action="store_true", help="Draw logged LiDAR cloud points from obstacle JSON files.")
    args = parser.parse_args()

    log_path = args.log
    if log_path is None and args.obstacle_log_dir is None:
        log_path = latest_log_path(args.log_dir)

    robot_points = []
    aruco_points = []
    clusters = []
    clouds = []
    nearest_only_count = 0

    if log_path is not None:
        csv_robot_points, aruco_points, csv_clusters, nearest_only_count = read_csv_log(log_path)
        robot_points.extend(csv_robot_points)
        clusters.extend(csv_clusters)

    if args.obstacle_log_dir is not None:
        json_robot_points, json_clusters, json_clouds = read_obstacle_json_dir(args.obstacle_log_dir)
        if not robot_points:
            robot_points.extend(json_robot_points)
        clusters.extend(json_clusters)
        clouds.extend(json_clouds)

    if not robot_points:
        raise ValueError("No robot position points found in the selected logs.")

    if not clusters:
        raise ValueError("No obstacle cluster or nearest-obstacle points found in the selected logs.")

    output_path = args.output or default_output_path(log_path, args.obstacle_log_dir)
    title_parts = []
    if log_path is not None:
        title_parts.append(log_path.name)
    if args.obstacle_log_dir is not None:
        title_parts.append(args.obstacle_log_dir.name)
    title = "Obstacle Cluster Postprocess: " + " + ".join(title_parts)

    plot_postprocess(
        robot_points=robot_points,
        aruco_points=aruco_points,
        clusters=clusters,
        clouds=clouds,
        output_path=output_path,
        title=title,
        max_circles=max(args.max_circles, 1),
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
    if nearest_only_count and radius_count == 0:
        print("This CSV is an old format log, so only nearest obstacle centres were available.")


if __name__ == "__main__":
    main()
