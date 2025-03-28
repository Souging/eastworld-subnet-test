# The MIT License (MIT)
# Copyright © 2025 Eastworld AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import base64
import io
import json
import pickle

from flask import Flask, jsonify, render_template, send_from_directory
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for plotting

import matplotlib.pyplot as plt

from eastworld.miner.slam.grid import ANONYMOUS_NODE_PREFIX, OccupancyGridMap


def visualize_gridmap(map: OccupancyGridMap, x, y: int):
    fig, ax = plt.subplots(figsize=(80, 80))

    # Base map
    prob_map = 1 - 1 / (1 + np.exp(map.grid))
    ax.imshow(prob_map, cmap="Greys", origin="lower")

    # Navigation topological
    for node_id, (_, node_x, node_y) in map.nav_nodes.items():
        grid_x, grid_y = map.world_to_grid(node_x, node_y)
        if node_id.startswith(ANONYMOUS_NODE_PREFIX):
            ax.scatter(grid_x, grid_y, color="blue", s=100, marker="^", alpha=0.5)
        else:
            ax.scatter(grid_x, grid_y, color="orange", s=100, marker="*", alpha=0.5)
    for node_id1, edges in map.nav_edges.items():
        for node_id2 in edges.keys():
            node_1 = map.nav_nodes[node_id1]
            node_2 = map.nav_nodes[node_id2]
            x1, y1 = map.world_to_grid(node_1[1], node_1[2])
            x2, y2 = map.world_to_grid(node_2[1], node_2[2])
            ax.plot([x1, x2], [y1, y2], color="green", alpha=0.5)

    # Current position
    grid_x, grid_y = map.world_to_grid(x, y)
    ax.scatter(grid_x, grid_y, color="red", s=300, marker="o")

    # Save as an image and return base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return image_png


class SLAMConsoleServer:
    def __init__(self, host="0.0.0.0", port=5000, data_dir="slam_data"):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.data_dir = data_dir

        self.setup_routes()

    def setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/api/map/current")
        def get_current_map():
            f = open(f"{self.data_dir}/map.pkl", "rb")
            map = pickle.load(f)
            f.close()

            f = open(f"{self.data_dir}/metadata.json", "r")
            metadata = json.load(f)
            x = metadata["x"]
            y = metadata["y"]
            theta = metadata["theta"]
            f.close()

            image_base64 = base64.b64encode(visualize_gridmap(map, x, y)).decode()

            return jsonify(
                {
                    "image": image_base64,
                    "position": {
                        "x": x,
                        "y": y,
                        "theta": theta,
                    },
                }
            )

        # Static files route
        @self.app.route("/static/<path:path>")
        def send_static(path):
            return send_from_directory("static", path)

    def run(self):
        self.app.run(host=self.host, port=self.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLAM Web Console")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, default="slam_data")

    args = parser.parse_args()

    server = SLAMConsoleServer(port=args.port, data_dir=args.data_dir)
    server.run()
