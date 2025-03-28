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


import concurrent.futures
import gzip
import io
import lzma
import math
import os
import pickle
import random
import time
from pathlib import Path

import bittensor as bt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for plotting

import matplotlib.pyplot as plt

from eastworld.miner.slam.grid import OccupancyGridMap

SENSOR_MAX_RANGE = 50.0


class Particle:
    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=0.0,
        weight=1.0,
        map_width=1000,
        map_height=1000,
        resolution=2.0,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.map = OccupancyGridMap(
            width=map_width, height=map_height, resolution=resolution
        )

    def copy(self):
        p = Particle(self.x, self.y, self.theta, self.weight)
        p.map.grid = self.map.grid.copy()
        return p


class FastSLAM:
    # Direction mapped to angles (radians) - corrected to north up, south down, west left, east right
    direction_to_angle = {
        "north": np.pi / 2,  # 90 degrees - up
        "northeast": np.pi / 4,  # 45 degrees - upper right
        "east": 0,  # 0 degrees - right
        "southeast": -np.pi / 4,  # -45 degrees - lower right
        "south": -np.pi / 2,  # -90 degrees - down
        "southwest": -3 * np.pi / 4,  # -135 degrees - lower left
        "west": np.pi,  # 180 degrees or -180 degrees - left
        "northwest": 3 * np.pi / 4,  # 135 degrees - upper left
    }

    def __init__(self, num_particles=200, load_data=False, data_dir="slam_data"):
        """
        FastSLAM Initialization

        Args:
            num_particles: Number of particles to use
            load_data: Load historical data on initialization
            data_dir: Directory to store SLAM data
        """
        self.num_particles = num_particles
        self.data_dir = data_dir
        self.history_dir = os.path.join(data_dir, "history")
        self.state_dir = os.path.join(data_dir, "states")
        self.history = []
        self.history_base_index = 0
        self.history_metadata = {"total": 0, "timestamps": [], "positions": []}

        self._save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._save_future = None

        self._iteration_count = 0

        # Randomly initialize particles
        self.particles = [
            Particle(
                x=np.random.normal(0, 60),
                y=np.random.normal(0, 60),
                theta=np.random.normal(-np.pi, np.pi),
            )
            for _ in range(num_particles)
        ]

        # Ensure data directories exist
        for directory in [self.data_dir, self.history_dir, self.state_dir]:
            os.makedirs(directory, exist_ok=True)

        # Create history metadata file
        self.metadata_file = os.path.join(self.data_dir, "history_metadata.pkl")

        # Motion model noise parameters
        self.alpha1 = 0.1  # Motion noise ratio
        self.alpha2 = 0.1  # Angle noise ratio

        # Sensor model parameters
        self.z_hit = 0.8  # Hit model weight
        self.z_rand = 0.2  # Random model weight
        self.sigma_hit = 0.4  # Hit model standard deviation

        self.pose_index = 0

        # Load historical data (if any and if needed)
        if load_data:
            self.load()

    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, "_save_executor"):
            self._save_executor.shutdown(wait=True)

    @property
    def grid_map(self):
        return self.get_best_particle().map

    def predict(self, odometry: float, orientation: str):
        """
        Predict the new position of particles based on odometry data
        """
        # Obtain the angle of the direction of movement
        direction_angle = self.direction_to_angle[orientation]

        # Concurrent processing to speed up particle prediction
        def update_particle(p: Particle):
            # Add noise to odometry and direction
            # noisy_odometry = odometry + np.random.normal(0, self.alpha1 * odometry)
            alpha1 = 0.1707
            alpha2 = 0.118

            noisy_odometry = odometry + np.random.normal(0, alpha1)
            noisy_angle = direction_angle + np.random.normal(0, alpha2)

            # Update particle position
            p.x += noisy_odometry * np.cos(noisy_angle)
            p.y += noisy_odometry * np.sin(noisy_angle)
            p.theta = noisy_angle
            return p

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(update_particle, p) for p in self.particles]
            self.particles = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

    def update_weights(self, lidar_data: dict[str, float]):
        """
        Update particle weights based on lidar data

        Args:
            lidar_data: Lidar measurements in different directions
        """
        for p in self.particles:
            # Reset weights
            p.weight = 1.0

            # Process measurements for each direction
            for direction, measured_dist in lidar_data.items():
                angle = self.direction_to_angle[direction]

                # Update the map for the particle based on its position and measurement
                self.update_map_for_particle(p, angle, measured_dist)
                # Calculate likelihood, update particle weight
                p.weight *= self.measurement_likelihood(p, angle, measured_dist)

                # Interpolate angles for more measurement
                interpolate_angles = []
                t = np.arctan(2.0 / measured_dist)
                if t > 0.09:
                    offsets = np.arange(0.09, t, 0.09)
                    interpolate_angles.extend(offsets)
                    interpolate_angles.extend(-offsets)
                for ia in interpolate_angles:
                    offset = angle + ia
                    offset = math.atan2(math.sin(offset), math.cos(offset))

                    confidence_factor = 0.8
                    self.update_map_for_particle(p, offset, measured_dist)
                    likelihood = self.measurement_likelihood(p, offset, measured_dist)
                    p.weight *= likelihood**confidence_factor

        # Normalize particle weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight

    def update_map_for_particle(
        self, particle: Particle, angle: float, distance: float
    ):
        """
        Update particle's map based on particle position and measurement
        """
        start_x, start_y = particle.x, particle.y
        end_x = start_x + distance * np.cos(particle.theta + angle)
        end_y = start_y + distance * np.sin(particle.theta + angle)

        # Bresenham's algorithm for ray tracing
        sx, sy = particle.map.world_to_grid(start_x, start_y)
        ex, ey = particle.map.world_to_grid(end_x, end_y)
        points = self.bresenham(sx, sy, ex, ey)

        # Update the cells along the line (except the last one which is the obstacle)
        for i, (gx, gy) in enumerate(points):
            if i < len(points) - 1:
                particle.map.update_cell(int(gx), int(gy), occupied=False)
            elif distance < SENSOR_MAX_RANGE:
                particle.map.update_cell(int(gx), int(gy), occupied=True)

    def bresenham(self, x0, y0, x1, y1):
        """Bresenham's algorithm implementation for ray tracing"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def measurement_likelihood(
        self, particle: Particle, angle: float, measured_dist: float
    ) -> float:
        """Calculate measurement likelihood"""
        # Gaussian likelihood model
        variance = self.sigma_hit**2
        # Random noise term
        # p_rand = 1.0 / 50.0  # Assume maximum measurement range is 50 meters
        p_rand = 0.1478

        # Find expected measurement in the particle map
        expected_dist = self.raycast(particle, angle)

        # Calculate Gaussian likelihood
        p_hit = 0
        if expected_dist is not None:
            error = measured_dist - expected_dist
            p_hit = np.exp(-0.5 * error**2 / variance) / np.sqrt(2 * np.pi * variance)

        # Combine likelihood
        likelihood = self.z_hit * p_hit + self.z_rand * p_rand
        return max(likelihood, 1e-10)  # Prevent zero weight

    def raycast(self, particle: Particle, angle: float) -> float | None:
        """
        Raycast on the particle map to find the first obstacle
        """
        # Easy implementation: sample at fixed distance
        max_range = SENSOR_MAX_RANGE
        step_size = 2.0

        for dist in np.arange(0, max_range, step_size):
            x = particle.x + dist * np.cos(particle.theta + angle)
            y = particle.y + dist * np.sin(particle.theta + angle)

            gx, gy = particle.map.world_to_grid(x, y)

            if particle.map.is_occupied(gx, gy):
                return dist

        return None

    def resample(self):
        new_particles: list[Particle] = []
        weights = [p.weight for p in self.particles]

        # Systematic Resampling
        cumulative_weights = np.cumsum(weights)
        step = 1.0 / self.num_particles
        u = random.uniform(0, step)

        i = 0
        for _ in range(self.num_particles):
            while u > cumulative_weights[i]:
                i += 1

            # Copy the selected particle
            new_particles.append(self.particles[i].copy())
            u += step

        self.particles = new_particles

        # Reset weights to uniform
        for p in self.particles:
            p.weight = 1.0 / self.num_particles

    def get_best_particle(self) -> Particle:
        return max(self.particles, key=lambda p: p.weight)

    def get_fuse_map(self, top_k: int = 10) -> OccupancyGridMap | None:
        """Pick the top_k particles based on their weights and fuse their maps"""
        top_k = min(top_k, len(self.particles))

        # Sort particles by weight in descending order
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)[
            :top_k
        ]

        if not sorted_particles:
            bt.logging.warning("No particles available for map fusion")
            return None

        # Use the first particle's map as the base
        base_particle = sorted_particles[0]
        map_width = base_particle.map.width
        map_height = base_particle.map.height
        resolution = base_particle.map.resolution

        # Create a new map with the same dimensions as the base particle's map
        fused_map = OccupancyGridMap(
            width=map_width, height=map_height, resolution=resolution
        )
        fused_map.grid = np.zeros((map_height, map_width), dtype=np.float64)
        total_weight = sum(p.weight for p in sorted_particles)

        # If the total weight is 0, use uniform weights
        if total_weight <= 0:
            weights = [1.0 / top_k] * top_k
        else:
            # Normalize weights to sum to 1
            weights = [p.weight / total_weight for p in sorted_particles]

        # Weighted fusion of the maps
        for i, particle in enumerate(sorted_particles):
            # Multiply each particle's map by its normalized weight
            fused_map.grid += particle.map.grid * weights[i]

        bt.logging.info(
            f"Fused map created from top {top_k} particles with total weight {total_weight:.6f}"
        )

        return fused_map

    def mesurement_effectiveness(self) -> float:
        """Calculate the effectiveness and diversity of the current particle set"""
        # 1. Calculate weight distribution effectiveness - using Neff (effective sample size)
        weights = np.array([p.weight for p in self.particles])
        weight_sum = np.sum(weights)

        # Normalize weights
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            return 0.0  # All weights are zero, particle set is completely ineffective

        # Calculate effective sample size Neff = 1 / Σ(w_i^2)
        # When all particles have equal weights, Neff = N (maximum value)
        # When only one particle has weight 1 and others 0, Neff = 1 (minimum value)
        weights_squared_sum = np.sum(np.square(weights))
        if weights_squared_sum > 0:
            neff = 1.0 / weights_squared_sum
        else:
            neff = 0.0

        # Normalize Neff to [0,1] range
        neff_normalized = neff / len(self.particles)

        # 2. Calculate spatial distribution diversity - using variance of particle positions
        positions = np.array([(p.x, p.y) for p in self.particles])

        # Calculate position covariance matrix
        if len(positions) > 1:
            cov_matrix = np.cov(positions.T)
            # Use determinant of covariance matrix as a measure of spatial dispersion
            if cov_matrix.size > 1:  # Ensure it's not a scalar
                det = np.linalg.det(cov_matrix)
                # Use logarithmic scaling to prevent too large values
                spatial_diversity = (
                    np.log(1 + det) / 20.0
                )  # 20 is a scaling factor, can be adjusted based on environment
                spatial_diversity = min(1.0, spatial_diversity)  # Limit to [0,1]
            else:
                spatial_diversity = 0.0
        else:
            spatial_diversity = 0.0

        # 3. Calculate orientation diversity - using variance of particle headings
        thetas = np.array([p.theta for p in self.particles])
        # Since angles are periodic, direct variance calculation is problematic
        # Convert angles to unit vectors, then calculate average vector length
        cos_theta = np.cos(thetas)
        sin_theta = np.sin(thetas)
        mean_vector_length = np.sqrt(np.mean(cos_theta) ** 2 + np.mean(sin_theta) ** 2)
        # mean_vector_length close to 1 indicates concentrated directions, close to 0 indicates dispersed directions
        angle_diversity = 1.0 - mean_vector_length

        # 4. Calculate map diversity - compare map similarity between different particles
        # For efficiency, only sample top_k particles for comparison
        top_k = min(10, len(self.particles))
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)[
            :top_k
        ]

        # Calculate map similarity
        if len(sorted_particles) > 1:
            map_similarities = []
            for i in range(len(sorted_particles)):
                for j in range(i + 1, len(sorted_particles)):
                    # Calculate correlation coefficient or similarity between two map grids
                    map1 = sorted_particles[i].map.grid.flatten()
                    map2 = sorted_particles[j].map.grid.flatten()

                    # Use cosine similarity
                    dot_product = np.dot(map1, map2)
                    norm1 = np.linalg.norm(map1)
                    norm2 = np.linalg.norm(map2)

                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        map_similarities.append(similarity)
                    else:
                        map_similarities.append(0.0)

            if map_similarities:
                avg_similarity = np.mean(map_similarities)
                map_diversity = 1.0 - avg_similarity
            else:
                map_diversity = 0.0
        else:
            map_diversity = 0.0

        # 5. Combine all metrics to calculate final particle set effectiveness
        # Weights: effective sample size(0.4) + spatial diversity(0.3) + orientation diversity(0.1) + map diversity(0.2)
        effectiveness = (
            0.4 * neff_normalized
            + 0.3 * spatial_diversity
            + 0.1 * angle_diversity
            + 0.2 * map_diversity
        )

        bt.logging.info(
            f"Particle effectiveness: {effectiveness:.4f} (Neff: {neff_normalized:.4f}, "
            f"Spatial: {spatial_diversity:.4f}, Angle: {angle_diversity:.4f}, "
            f"Map: {map_diversity:.4f})"
        )

        return effectiveness

    def inject_random_particles(self, ratio=0.1, min_count=5, around_best=True):
        """Inject random particles to improve particle set diversity"""
        # Calculate the number of particles to inject
        inject_count = max(min_count, int(len(self.particles) * ratio))

        # Get the best particle as the injection center
        best_particle = self.get_best_particle()

        # Generate new particles
        new_particles = []
        for _ in range(inject_count):
            if around_best:
                # Add random offsets around the best particle's position
                x = best_particle.x + np.random.normal(
                    0, 20
                )  # 20 is the position offset standard deviation
                y = best_particle.y + np.random.normal(0, 20)
                theta = best_particle.theta + np.random.normal(
                    0, np.pi / 4
                )  # π/4 is the direction offset standard deviation
                theta = math.atan2(math.sin(theta), math.cos(theta))

                # Create new particle with the same map dimensions as the best particle
                p = Particle(
                    x=x,
                    y=y,
                    theta=theta,
                    weight=1.0 / self.num_particles,
                    map_width=best_particle.map.width,
                    map_height=best_particle.map.height,
                    resolution=best_particle.map.resolution,
                )

                # Optional: Copy part of the map information from the best particle
                # Here we only copy cells with high certainty (cells with large absolute values)
                certainty_threshold = 2.0  # logodds threshold
                for i in range(best_particle.map.height):
                    for j in range(best_particle.map.width):
                        cell_value = best_particle.map.grid[i, j]
                        if abs(cell_value) > certainty_threshold:
                            p.map.grid[i, j] = cell_value

                new_particles.append(p)
            else:
                # Generate completely random particles
                p = Particle(
                    x=np.random.normal(
                        0, 100
                    ),  # Larger range for random initial positions
                    y=np.random.normal(0, 100),
                    theta=np.random.uniform(-np.pi, np.pi),
                    weight=1.0 / self.num_particles,
                )
                new_particles.append(p)

        # Remove particles with lowest weights
        self.particles.sort(key=lambda p: p.weight)
        self.particles = self.particles[inject_count:] + new_particles

        # Re-normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight

        bt.logging.info(
            f"Injected {inject_count} new particles "
            f"{'around best particle' if around_best else 'randomly'}"
        )

    def get_fuse_map_with_alignment(self, top_k: int = 10) -> OccupancyGridMap:
        """Advanced map fusion considering particle position differences with alignment"""
        # Limit top_k to not exceed the total number of particles
        top_k = min(top_k, len(self.particles))

        # Sort particles by weight
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)[
            :top_k
        ]

        # If there are no particles, return an empty map
        if not sorted_particles:
            return OccupancyGridMap()

        # Get the highest weight particle as reference
        reference_particle = sorted_particles[0]
        ref_x, ref_y = reference_particle.x, reference_particle.y

        # Create fusion map with the same dimensions as the reference particle
        map_width = reference_particle.map.width
        map_height = reference_particle.map.height
        resolution = reference_particle.map.resolution

        fused_map = OccupancyGridMap(
            width=map_width, height=map_height, resolution=resolution
        )
        fused_map.grid = np.zeros((map_height, map_width), dtype=np.float64)

        # Initialize weight accumulation grid to track how many particles updated each cell
        weight_grid = np.zeros((map_height, map_width), dtype=np.float64)

        # Calculate total weight of selected particles
        total_weight = sum(p.weight for p in sorted_particles)

        # If total weight is 0, use uniform weights
        if total_weight <= 0:
            weights = [1.0 / top_k] * top_k
        else:
            # Normalize weights
            weights = [p.weight / total_weight for p in sorted_particles]

        # Align and fuse each particle
        for i, particle in enumerate(sorted_particles):
            # Calculate position offset relative to reference particle
            dx = ref_x - particle.x
            dy = ref_y - particle.y

            # Convert to grid offset
            grid_dx, grid_dy = int(dx / resolution), int(dy / resolution)

            # Fuse maps after alignment
            for y in range(map_height):
                for x in range(map_width):
                    # Calculate aligned coordinates
                    aligned_x = x - grid_dx
                    aligned_y = y - grid_dy

                    # Check if aligned coordinates are within original map bounds
                    if (0 <= aligned_x < map_width) and (0 <= aligned_y < map_height):
                        # Add to fusion map weighted by particle weight
                        cell_value = (
                            particle.map.grid[aligned_y, aligned_x] * weights[i]
                        )
                        fused_map.grid[y, x] += cell_value
                        weight_grid[y, x] += weights[i]

        # Normalize grid values by accumulated weights
        # Avoid division by zero
        mask = weight_grid > 0
        fused_map.grid[mask] /= weight_grid[mask]

        bt.logging.info(f"Aligned and fused map created from top {top_k} particles")

        return fused_map

    def get_current_pose(self) -> tuple[float, float, float]:
        best_particle = self.get_best_particle()
        return best_particle.x, best_particle.y, best_particle.theta

    def run_iteration(
        self, lidar_data: dict[str, float], odometry: float, odometry_direction: str
    ):
        self.predict(odometry, odometry_direction)
        bt.logging.info("SLAM predict done")
        self.update_weights(lidar_data)
        bt.logging.info("SLAM update weights done")
        self.resample()
        bt.logging.info("SLAM resample done")

        if self._iteration_count % 10 == 0:
            if self._save_future is not None and not self._save_future.done():
                bt.logging.warning("Previous save still running, skipping this save")
                return
            self._save_future = self._save_executor.submit(self.save)
            bt.logging.info("SLAM save state submitted to background")

        self._iteration_count += 1

    def save(self):
        """Save the current SLAM state to history and file"""
        with open(os.path.join(self.data_dir, "map.pkl"), "wb") as f:
            pickle.dump(
                self.get_fuse_map_with_alignment(), f, protocol=pickle.HIGHEST_PROTOCOL
            )

        best_particle = self.get_best_particle()
        bt.logging.debug("SLAM save state get best particle")
        timestamp = time.time()

        # Generate and save the visualization
        image = self.visualize()
        bt.logging.debug("SLAM save state generated visualization")
        image_filename = os.path.join(self.history_dir, f"{timestamp:.6f}.png.gz")
        with gzip.open(image_filename, "wb") as f:
            f.write(image)
        bt.logging.debug("SLAM save state save visualization")

        # Create a history entry (excluding image data)
        entry = {
            "timestamp": timestamp,
            "image_path": image_filename,  # Store the image path instead of image data
            "position": {
                "x": best_particle.x,
                "y": best_particle.y,
                "theta": best_particle.theta,
            },
        }
        self.history.append(entry)

        bt.logging.debug("SLAM save state save serialize begin")
        # Serialize and save the SLAM state
        state_data = {
            "timestamp": timestamp,
            "position": entry["position"],
            "slam_state": self._serialize_efficient(),
        }
        bt.logging.debug("SLAM save state save serialized")
        state_filename = os.path.join(self.state_dir, f"{timestamp:.6f}.pkl.xz")
        with lzma.open(state_filename, "wb") as f:
            pickle.dump(state_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        bt.logging.debug("SLAM save state save state data lzma")

        # Update metadata
        self.history_metadata["total"] = len(self.history)
        self.history_metadata["timestamps"].append(timestamp)
        self.history_metadata["positions"].append(entry["position"])
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.history_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        bt.logging.info(f"SLAM state saved: {state_filename}")
        return entry

    def _serialize_efficient(self):
        """Serialize the SLAM state using a more efficient method, saving only necessary data"""
        # Only save the key attributes of particles to reduce redundant data
        particles_data = []
        for p in self.particles:
            # Compress grid data - only save non-zero values
            grid = p.map.grid
            # Retrieve the indices and values of non-zero elements
            nonzero_indices = np.nonzero(grid)
            # Pack the indices and values into a list of tuples
            sparse_grid = [
                (i, j, grid[i, j])
                for i, j in zip(nonzero_indices[0], nonzero_indices[1])
            ]

            particles_data.append(
                {
                    "x": p.x,
                    "y": p.y,
                    "theta": p.theta,
                    "weight": p.weight,
                    "map_info": {
                        "width": p.map.width,
                        "height": p.map.height,
                        "resolution": p.map.resolution,
                    },
                    "sparse_grid": (
                        sparse_grid
                        if len(sparse_grid) < (p.map.width * p.map.height // 10)
                        else None
                    ),
                    "full_grid": (
                        None
                        if len(sparse_grid) < (p.map.width * p.map.height // 10)
                        else grid.tobytes()
                    ),
                }
            )

        return {
            "num_particles": self.num_particles,
            "particles": particles_data,
        }

    @classmethod
    def _deserialize_efficient(cls, data):
        """Restore the SLAM state from efficient serialization data"""
        slam = cls(num_particles=data["num_particles"], load_history=False)

        new_particles = []
        for p_data in data["particles"]:
            p = Particle(
                x=p_data["x"],
                y=p_data["y"],
                theta=p_data["theta"],
                weight=p_data["weight"],
                map_width=p_data["map_info"]["width"],
                map_height=p_data["map_info"]["height"],
            )

            # Set map resolution
            p.map.resolution = p_data["map_info"]["resolution"]

            # Restore grid data
            if p_data["sparse_grid"] is not None:
                # Restore using sparse representation
                for i, j, value in p_data["sparse_grid"]:
                    p.map.grid[i, j] = value
            elif p_data["full_grid"] is not None:
                # Restore using full binary data - fix read-only array issue
                grid_shape = (p_data["map_info"]["height"], p_data["map_info"]["width"])
                # Create a writable copy of the array using copy=True parameter, or use np.array conversion
                p.map.grid = np.array(
                    np.frombuffer(p_data["full_grid"], dtype=np.float64).reshape(
                        grid_shape
                    ),
                    copy=True,
                )

            new_particles.append(p)

        slam.particles = new_particles
        return slam

    def load(self, max_entries: int = 100, load_states: bool = False):
        """Load historical SLAM data, limiting the maximum number of entries for performance"""
        # Ensure data directories exist
        if not os.path.exists(self.data_dir):
            for directory in [self.data_dir, self.history_dir, self.state_dir]:
                os.makedirs(directory, exist_ok=True)
            return

        # Try to load metadata first
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "rb") as f:
                    self.history_metadata = pickle.load(f)
        except Exception as e:
            bt.logging.error(f"Error loading history metadata: {str(e)}")
            self.history_metadata = {"total": 0, "timestamps": [], "positions": []}

        # If the metadata is empty, scan the directories and sort by timestamp
        if not self.history_metadata["timestamps"]:
            # Find all state files
            state_files = list(Path(self.state_dir).glob("*.pkl.xz"))
            if not state_files:
                return

            # Extract timestamps from filenames and sort
            timestamps = [float(f.name.split(".")[0]) for f in state_files]
            timestamps.sort()
            self.history_metadata["timestamps"] = timestamps
            self.history_metadata["positions"] = [{} for _ in range(len(timestamps))]
            self.history_metadata["total"] = len(timestamps)

            # Update metadata file
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.history_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Limit loading to the latest max_entries entries
        timestamps = self.history_metadata["timestamps"]
        if not timestamps:
            return
        recent_timestamps = timestamps[-max_entries:]

        if load_states:
            self.history = []
            # Load history entries
            for timestamp in recent_timestamps:
                state_filename = os.path.join(self.state_dir, f"{timestamp:.6f}.pkl.xz")
                image_filename = os.path.join(
                    self.history_dir, f"{timestamp:.6f}.png.gz"
                )

                print(f"Loading {state_filename} {time.time()}")
                try:
                    entry = {
                        "timestamp": timestamp,
                        "image_path": image_filename,  # Store image path instead of data
                        "position": None,
                    }

                    if os.path.exists(state_filename):
                        with lzma.open(state_filename, "rb") as f:
                            state_data = pickle.load(f)
                            entry["position"] = state_data["position"]

                    self.history.append(entry)
                except Exception as e:
                    bt.logging.error(
                        f"Error loading history entry {state_filename}: {str(e)}"
                    )

        # Load the latest state as the current state
        try:
            latest_timestamp = timestamps[-1]
            latest_state_file = os.path.join(
                self.state_dir, f"{latest_timestamp:.6f}.pkl.xz"
            )

            if os.path.exists(latest_state_file):
                with lzma.open(latest_state_file, "rb") as f:
                    state_data = pickle.load(f)
                    loaded_slam = self._deserialize_efficient(state_data["slam_state"])
                    self.particles = loaded_slam.particles
                    bt.logging.info(
                        f"Loaded latest SLAM state from {latest_state_file}"
                    )
        except Exception as e:
            bt.logging.error(f"Error loading latest SLAM state: {str(e)}")

    def visualize(self) -> bytes:
        """
        Visualize the current state of the SLAM system
        """
        fig, ax = plt.subplots(figsize=(40, 40))

        # Plot the best particle's map
        best_particle = self.get_best_particle()

        prob_map = 1 - 1 / (1 + np.exp(best_particle.map.grid))
        ax.imshow(prob_map, cmap="Greys", origin="lower")
        ax.set_title("Occupancy Grid Map")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        bt.logging.debug("SLAM visualization base map")

        # Add scale bar
        total_width_meters = best_particle.map.width * best_particle.map.resolution
        total_height_meters = best_particle.map.height * best_particle.map.resolution
        ax.set_title(
            f"FastSLAM with Occupancy Grid Map\nMap size: {total_width_meters:.1f}m x {total_height_meters:.1f}m"
        )
        ax.set_xlabel(f"X (Resolution: {best_particle.map.resolution}m/cell)")
        ax.set_ylabel(f"Y (Resolution: {best_particle.map.resolution}m/cell)")

        # Draw all particles
        for p in self.particles:
            grid_x, grid_y = p.map.world_to_grid(p.x, p.y)
            size = 10 * p.weight * self.num_particles
            ax.scatter(grid_x, grid_y, color="red", s=size, alpha=0.5)

            length = 3
            dx = length * np.cos(p.theta)
            dy = length * np.sin(p.theta)
            ax.arrow(
                grid_x,
                grid_y,
                dx,
                dy,
                head_width=1,
                head_length=1,
                fc="red",
                ec="red",
                alpha=0.5,
            )

        # Highlight the best particle
        grid_x, grid_y = best_particle.map.world_to_grid(
            best_particle.x, best_particle.y
        )
        ax.scatter(grid_x, grid_y, color="green", s=100, marker="*")

        plt.tight_layout()  # Ensure title and axis labels are fully displayed
        bt.logging.debug("SLAM visualization particles draw " + plt.get_backend())

        # Convert the figure to a base64 string
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            transparent=False,
            dpi=80,
        )  # Add bbox_inches parameter to ensure complete saving
        plt.close(fig)
        bt.logging.debug("SLAM visualization plt saved to buffer")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        return image_png
