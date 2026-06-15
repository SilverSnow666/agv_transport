import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    ep = df[df["episode"] == args.episode]

    # 根据你的图，env_0 的原点大概在 (2.0, -2.0)
    origin_x = ep["payload_x"].iloc[0]
    origin_y = ep["payload_y"].iloc[0]

    # 当前 V4.1 路径，局部坐标
    waypoints_local = [
        (0.00, 0.00),
        (0.85, 0.00),
        (1.20, 0.12),
        (1.55, 0.20),
        (1.80, 0.08),
        (1.95, 0.00),
    ]

    waypoints_x = [origin_x + p[0] for p in waypoints_local]
    waypoints_y = [origin_y + p[1] for p in waypoints_local]

    plt.figure(figsize=(7, 5))

    plt.plot(
        ep["payload_x"],
        ep["payload_y"],
        label="payload trajectory",
    )

    plt.plot(
        ep["target_x"],
        ep["target_y"],
        "--",
        label="lookahead target",
    )

    plt.plot(
        waypoints_x,
        waypoints_y,
        "-.",
        marker="o",
        label="planned path",
    )

    plt.scatter(
        ep["payload_x"].iloc[0],
        ep["payload_y"].iloc[0],
        label="start",
    )

    plt.scatter(
        ep["payload_x"].iloc[-1],
        ep["payload_y"].iloc[-1],
        label="end",
    )

    plt.axis("equal")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.title(f"Payload trajectory, episode {args.episode}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()