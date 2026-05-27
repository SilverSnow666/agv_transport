"""Plot comparison figures for single-AGV offset pushing and three-AGV cooperative pushing.

Outputs:
    1. payload_trajectory_comparison.png
    2. payload_yaw_comparison.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CASES = {
    "Single AGV offset +": Path(
        "logs/case_A2_single_agv_offset_pos/trajectory.csv"
    ),
    "Single AGV offset -": Path(
        "logs/case_A2_single_agv_offset_neg/trajectory.csv"
    ),
    "Three AGV cooperative": Path(
        "logs/three_agv_scripted_eval_contact120/trajectory.csv"
    ),
}

OUTPUT_DIR = Path("logs/three_agv_comparison_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_episode_zero(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found in {csv_path}")

    df = df[df["episode"] == 0].copy()
    df = df.sort_values("step_in_episode")

    return df


def plot_payload_trajectory():
    plt.figure(figsize=(8, 5))

    for label, csv_path in CASES.items():
        df = load_episode_zero(csv_path)

        plt.plot(
            df["payload_x"],
            df["payload_y"],
            label=label,
            linewidth=2,
        )

        # start point
        plt.scatter(
            df["payload_x"].iloc[0],
            df["payload_y"].iloc[0],
            marker="o",
        )

        # end point
        plt.scatter(
            df["payload_x"].iloc[-1],
            df["payload_y"].iloc[-1],
            marker="x",
        )

    # target point from the three-AGV trajectory
    df_ref = load_episode_zero(CASES["Three AGV cooperative"])
    target_x = df_ref["target_x"].iloc[0]
    target_y = df_ref["target_y"].iloc[0]

    plt.scatter(
        target_x,
        target_y,
        marker="*",
        s=160,
        label="Target",
    )

    plt.xlabel("Payload x position / m")
    plt.ylabel("Payload y position / m")
    plt.title("Payload trajectory comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "payload_trajectory_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved: {output_path}")


def plot_payload_yaw():
    plt.figure(figsize=(8, 5))

    for label, csv_path in CASES.items():
        df = load_episode_zero(csv_path)

        time_step = df["step_in_episode"]

        plt.plot(
            time_step,
            df["payload_yaw"],
            label=label,
            linewidth=2,
        )

    plt.xlabel("Step")
    plt.ylabel("Payload yaw / rad")
    plt.title("Payload yaw comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "payload_yaw_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved: {output_path}")


def plot_payload_target_distance():
    plt.figure(figsize=(8, 5))

    for label, csv_path in CASES.items():
        df = load_episode_zero(csv_path)

        plt.plot(
            df["step_in_episode"],
            df["payload_target_dist"],
            label=label,
            linewidth=2,
        )

    plt.xlabel("Step")
    plt.ylabel("Payload-target distance / m")
    plt.title("Payload-target distance comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "payload_target_distance_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved: {output_path}")


def main():
    plot_payload_trajectory()
    plot_payload_yaw()
    plot_payload_target_distance()


if __name__ == "__main__":
    main()