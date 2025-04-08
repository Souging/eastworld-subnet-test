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


import random
import unittest

import numpy as np

from eastworld.miner.slam.grid import OccupancyGridMap


class TestOccupancyGridMap(unittest.TestCase):
    def setUp(self):
        self.grid_map = OccupancyGridMap(width=200, height=200, resolution=5.0)

    def test_initialization(self):
        self.assertEqual(self.grid_map.width, 200)
        self.assertEqual(self.grid_map.height, 200)
        self.assertEqual(self.grid_map.resolution, 5.0)
        self.assertEqual(self.grid_map.grid.shape, (200, 200))

        self.assertTrue(np.all(self.grid_map.grid == 0))

    def test_conversion_world_to_grid(self):
        cases = [
            ((12.5, -12.5), (102, 97)),
            ((7.5, 7.5), (101, 101)),
            ((5, 5), (101, 101)),
            ((4, 4), (100, 100)),  # positive direction
            ((0, 0), (100, 100)),  # center
            ((-4, -4), (99, 99)),  # negative direction
            ((-5, -5), (99, 99)),
            ((-7.5, -7.5), (98, 98)),
            ((-12.5, 12.5), (97, 102)),
            # boundary
            ((600, 600), (199, 199)),
            ((-600, -600), (0, 0)),
        ]
        for world, grid in cases:
            grid_x, grid_y = self.grid_map.world_to_grid(*world)
            self.assertEqual((grid_x, grid_y), grid)

    def test_conversion_grid_to_world(self):
        cases = [
            ((12.5, -12.5), (102, 97)),
            ((7.5, 7.5), (101, 101)),
            ((5, 5), (101, 101)),
            ((4, 4), (100, 100)),  # positive direction
            ((0, 0), (100, 100)),  # center
            ((-4, -4), (99, 99)),  # negative direction
            ((-5, -5), (99, 99)),
            ((-7.5, -7.5), (98, 98)),
            ((-12.5, 12.5), (97, 102)),  # updated case
            # boundary
            ((495, 495), (199, 199)),
            ((-500, -500), (0, 0)),
        ]
        for world, grid in cases:
            world_x, world_y = self.grid_map.grid_to_world(*grid)
            self.assertLessEqual(abs(world_x - world[0]), self.grid_map.resolution)
            self.assertLessEqual(abs(world_y - world[1]), self.grid_map.resolution)

    def test_conversion_roundtrip(self):
        cases = [
            ((12.5, -12.5), (102, 97)),
            ((7.5, 7.5), (101, 101)),
            ((5, 5), (101, 101)),
            ((4, 4), (100, 100)),  # positive direction
            ((0, 0), (100, 100)),  # center
            ((-4, -4), (99, 99)),  # negative direction
            ((-5, -5), (99, 99)),
            ((-7.5, -7.5), (98, 98)),
            ((-12.5, 12.5), (97, 102)),
        ]

        for world, _ in cases:
            grid_x, grid_y = self.grid_map.world_to_grid(*world)
            world_x, world_y = self.grid_map.grid_to_world(grid_x, grid_y)

            self.assertLessEqual(abs(world[0] - world_x), self.grid_map.resolution)
            self.assertLessEqual(abs(world[1] - world_y), self.grid_map.resolution)

    def test_conversion_roundtrip_with_offset(self):
        cases = [
            ((12.5, -12.5), (112, 87)),
            ((7.5, 7.5), (111, 91)),
            ((5, 5), (111, 91)),
            ((4, 4), (110, 90)),  # positive direction
            ((0, 0), (110, 90)),  # center
            ((-4, -4), (99, 89)),  # negative direction
            ((-5, -5), (99, 89)),
            ((-7.5, -7.5), (98, 88)),
            ((-12.5, 12.5), (97, 92)),
        ]

        map_with_offset = OccupancyGridMap(width=200, height=200, resolution=5.0)
        map_with_offset.base_offset_x = 10
        map_with_offset.base_offset_y = -10

        map_with_offset.update_cell(112, 87, True)
        map_with_offset.update_cell(97, 102, True)
        map_with_offset.justify_map(10)

        for world, _ in cases:
            grid_x, grid_y = map_with_offset.world_to_grid(*world)
            world_x, world_y = map_with_offset.grid_to_world(grid_x, grid_y)

            self.assertLessEqual(abs(world[0] - world_x), map_with_offset.resolution)
            self.assertLessEqual(abs(world[1] - world_y), map_with_offset.resolution)

    def test_update_cell(self):
        # Update cell to occupied
        self.grid_map.update_cell(10, 10, True)
        self.assertEqual(
            self.grid_map.grid[10, 10], self.grid_map.log_odds_occupied, "Occupied"
        )

        # Update cell to unoccupied
        self.grid_map.update_cell(10, 10, False)
        self.assertEqual(
            self.grid_map.grid[10, 10],
            self.grid_map.log_odds_occupied + self.grid_map.log_odds_free,
            "Unoccupied",
        )

        # Test that updates outside the boundary have no effect
        original_grid = self.grid_map.grid.copy()
        self.grid_map.update_cell(-1, -1, True)
        self.grid_map.update_cell(
            self.grid_map.width + 1, self.grid_map.height + 1, True
        )
        np.testing.assert_array_equal(self.grid_map.grid, original_grid)

    def test_is_occupied(self):
        # Initial state: all cells are unoccupied
        for _ in range(10):
            x = random.randint(0, self.grid_map.width - 1)
            y = random.randint(0, self.grid_map.height - 1)
            self.assertFalse(self.grid_map.is_occupied(x, y))

        # Update enough times for the cell to be considered occupied
        for _ in range(10):  # Multiple updates to accumulate log-odds value
            self.grid_map.update_cell(10, 10, True)

        # Confirm that the cell is now considered occupied
        self.assertTrue(self.grid_map.is_occupied(10, 10))

        # Cells outside the boundary should be considered unknown
        self.assertIsNone(self.grid_map.is_occupied(-1, -1))
        self.assertIsNone(
            self.grid_map.is_occupied(self.grid_map.width, self.grid_map.height)
        )

    def test_expand_map(self):
        # Mark some cells near the center of the map
        marked_cells = []
        for y in range(90, 110):
            for x in range(90, 110):
                # Mark cells as occupied and unoccupied alternately
                is_occupied = (x + y) % 2 == 0
                self.grid_map.update_cell(x, y, is_occupied)
                marked_cells.append((x, y, self.grid_map.grid[y, x]))

        original_width = self.grid_map.width
        original_height = self.grid_map.height
        x_offset, y_offset = self.grid_map.expand_map()

        # Verify that the map size has increased
        self.assertGreater(self.grid_map.width, original_width)
        self.assertGreater(self.grid_map.height, original_height)

        # Verify the offset calculation
        expected_x_offset = (self.grid_map.width - original_width) // 2
        expected_y_offset = (self.grid_map.height - original_height) // 2
        self.assertEqual(x_offset, expected_x_offset)
        self.assertEqual(y_offset, expected_y_offset)

        # Verify that the original data has been correctly copied to the new map
        for x, y, value in marked_cells:
            new_x = x + x_offset
            new_y = y + y_offset
            self.assertEqual(self.grid_map.grid[new_y, new_x], value)

        expanded_width = self.grid_map.width
        expanded_height = self.grid_map.height

        # Try expanding with a smaller size (should be ignored)
        small_width = self.grid_map.width // 2
        small_height = self.grid_map.height // 2

        x_offset, y_offset = self.grid_map.expand_map(
            new_width=small_width, new_height=small_height
        )
        # Assert that the map size has not changed
        self.assertEqual(self.grid_map.width, expanded_width)
        self.assertEqual(self.grid_map.height, expanded_height)
        self.assertEqual(x_offset, 0)
        self.assertEqual(y_offset, 0)

    def test_justify_map(self):
        """Test if the justify_map method can correctly adjust the map size and maintain coordinate consistency"""
        # Prepare test data: create a content area in part of the map
        content_region = [(50, 50), (100, 50), (50, 100), (100, 100)]

        # Mark these positions with different values for later checking
        for i, (x, y) in enumerate(content_region):
            value = (i + 1) * 1.0  # Use different values
            self.grid_map.grid[y, x] = value

        # Save world coordinates in the original coordinate system
        original_world_coords = [
            self.grid_map.grid_to_world(x, y) for x, y in content_region
        ]

        # Execute map adjustment
        self.grid_map.justify_map(factor=1.5)

        # Verify the map size has been adjusted
        self.assertEqual(
            self.grid_map.width, 76
        )  # (100-50+1)*1.5 = 76.5, rounded to 76
        self.assertEqual(self.grid_map.height, 76)

        # Find the content after adjustment
        found_cells = []
        for y in range(self.grid_map.height):
            for x in range(self.grid_map.width):
                if self.grid_map.grid[y, x] > 0:
                    found_cells.append((x, y, self.grid_map.grid[y, x]))

        # Verify all content is preserved
        self.assertEqual(len(found_cells), len(content_region))

        # Verify if the relative position of the content is correct
        # Content should be centered, with the first point at (x_offset, y_offset)
        x_offset = (76 - (100 - 50 + 1)) // 2
        y_offset = (76 - (100 - 50 + 1)) // 2

        # Check if the first point is at the correct position
        first_point = next(fc for fc in found_cells if fc[2] == 1.0)
        self.assertEqual((first_point[0], first_point[1]), (x_offset, y_offset))

        # Verify coordinate consistency: find the same content through world coordinates
        for i, world_coord in enumerate(original_world_coords):
            grid_x, grid_y = self.grid_map.world_to_grid(*world_coord)
            # Find the grid cell with the corresponding value
            expected_value = (i + 1) * 1.0
            self.assertAlmostEqual(
                self.grid_map.grid[grid_y, grid_x],
                expected_value,
                msg=f"Coordinate inconsistency: world coordinate {world_coord} maps to {grid_x},{grid_y}",
            )

    def test_justify_map_empty(self):
        """Test the case when the map is empty"""
        # Create a completely empty map
        empty_map = OccupancyGridMap(width=100, height=100)

        # Save original dimensions
        original_width = empty_map.width
        original_height = empty_map.height

        # Execute map adjustment
        empty_map.justify_map()

        # Verify map dimensions are unchanged
        self.assertEqual(empty_map.width, original_width)
        self.assertEqual(empty_map.height, original_height)

    def test_justify_map_coordinate_consistency(self):
        """Test the consistency of the coordinate system after map adjustment"""
        # Create some structures on the map
        for y in range(40, 140):
            for x in range(60, 160):
                occupied = (x + y) % 5 == 0  # Create some pattern
                self.grid_map.update_cell(x, y, occupied)

        # Select some world coordinate points for testing
        test_points = [
            (0, 0),  # Origin
            (50, 50),  # Inside the content area
            (-50, -50),  # Map edge
            (100, -100),  # Another side of the map
        ]

        # Record grid coordinates of these points before adjustment
        original_grid_points = [self.grid_map.world_to_grid(*p) for p in test_points]

        # Get the values at these grid coordinates
        original_values = [self.grid_map.grid[y, x] for x, y in original_grid_points]

        # Execute map adjustment
        self.grid_map.justify_map(factor=1.2)

        # Verify that grid cells mapped from world coordinates contain the same values after adjustment
        for i, point in enumerate(test_points):
            new_grid_x, new_grid_y = self.grid_map.world_to_grid(*point)
            # Check boundary cases
            if (
                0 <= new_grid_x < self.grid_map.width
                and 0 <= new_grid_y < self.grid_map.height
            ):
                # For areas with content, should maintain the same value
                if abs(original_values[i]) > 0.1:  # Points with actual content
                    self.assertAlmostEqual(
                        self.grid_map.grid[new_grid_y, new_grid_x],
                        original_values[i],
                        msg=f"Value inconsistency for point {point}: expected {original_values[i]}, actual {new_grid_x} {new_grid_y} {self.grid_map.grid[new_grid_y, new_grid_x]}",
                    )

    def test_find_frontier_cells(self):
        # Create a simple environment: the center area is known and free, surrounded by unknown areas
        for y in range(80, 130):
            for x in range(80, 130):
                self.grid_map.update_cell(x, y, False)  # Mark as free

        # Find frontier cells
        frontiers = self.grid_map._find_frontier_cells(min_frontier_size=1)

        # There should be frontier cells
        self.assertTrue(len(frontiers) > 0)

        # Verify that frontier cells are indeed at the boundary of known and unknown areas
        for frontier in frontiers:
            for x, y in frontier:
                # Check if this cell is free
                self.assertTrue(self.grid_map.grid[y, x] < -0.1)

                # Check if this cell has unknown neighbors
                has_unknown_neighbor = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_map.width and 0 <= ny < self.grid_map.height:
                        if abs(self.grid_map.grid[ny, nx]) < 0.1:  # Unknown area
                            has_unknown_neighbor = True
                            break
                self.assertTrue(has_unknown_neighbor)

    def test_get_best_exploration_target(self):
        # Setup a similar environment as the find_frontier_cells test
        for y in range(80, 130):
            for x in range(80, 130):
                self.grid_map.update_cell(x, y, False)  # Mark as free

        # Get the best exploration target from the center of the map
        target = self.grid_map.get_nearest_exploration_target(100, 100)

        # The target should be found
        self.assertIsNotNone(target)

        # The target should be a coordinate point
        self.assertEqual(len(target), 2)

        # The target should be within the map boundaries
        target_x, target_y = target
        self.assertTrue(0 <= target_x < self.grid_map.width)
        self.assertTrue(0 <= target_y < self.grid_map.height)

    def test_astar_path(self):
        """Test if A* path algorithm can correctly find a path from start to goal"""
        # Scenario 1: Pathfinding in an open area
        # Reset the map
        self.grid_map.reset()

        # No obstacles between start and goal
        start = (10, 10)
        goal = (50, 50)

        # Call A* algorithm to find path
        path = self.grid_map._astar_path(start, goal)

        # Verify path exists
        self.assertTrue(len(path) > 0, "Should find a path in an open area")
        # Verify start and end points of the path are correct
        self.assertEqual(path[0], start, "Path start point incorrect")
        self.assertEqual(path[-1], goal, "Path end point incorrect")
        # Verify path continuity
        self.verify_path_continuity(path)

        # Scenario 2: Pathfinding with obstacles
        # Reset the map
        self.grid_map.reset()

        # Create a horizontal wall with a passage
        for x in range(20, 40):
            for y in range(29, 31):
                if not (x == 30):  # Leave a passage at x=30
                    self.grid_map.update_cell(x, y, True)  # Mark as obstacle

        start = (25, 25)
        goal = (25, 35)

        # Call A* algorithm to find path
        path = self.grid_map._astar_path(start, goal)

        # Verify path exists and goes through the passage
        self.assertTrue(len(path) > 0, "Should find a path around obstacles")
        # Verify start and end points of the path are correct
        self.assertEqual(path[0], start, "Path start point incorrect")
        self.assertEqual(path[-1], goal, "Path end point incorrect")
        # Verify path continuity
        self.verify_path_continuity(path)
        # Check if path goes through the designated passage (x=30)
        passage_used = False
        for x, y in path:
            if x == 30 and 29 <= y <= 31:
                passage_used = True
                break
        self.assertTrue(passage_used, "Path should go through the designated passage")

        # Scenario 3: Unreachable goal
        # Reset the map
        self.grid_map.reset()

        # Create a completely enclosed area
        for x in range(60, 80):
            for y in range(60, 80):
                if x == 60 or x == 79 or y == 60 or y == 79:
                    self.grid_map.update_cell(x, y, True)  # Create a wall

        start = (65, 65)  # Inside point
        goal = (90, 90)  # Outside point

        # Call A* algorithm to find path
        path = self.grid_map._astar_path(start, goal)

        # Verify that no path can be found
        self.assertEqual(
            len(path), 0, "Points inside an enclosed area cannot reach outside"
        )

        # Scenario 4: Maximum iteration limit
        # Reset the map
        self.grid_map.reset()

        # Create a very complex maze (simplified as a large grid here)
        for x in range(100, 150):
            for y in range(100, 150):
                if (x + y) % 2 == 0:  # Create "checkerboard" obstacles
                    self.grid_map.update_cell(x, y, True)

        start = (100, 100)
        goal = (149, 149)

        # Call A* algorithm with a very small maximum iteration count
        path = self.grid_map._astar_path(start, goal, max_iterations=10)

        # Verify that with limited iterations, cannot find complete path
        self.assertEqual(
            len(path), 0, "Limited iterations should result in no path found"
        )

    def test_dstar_lite_path(self):
        """Test if D* Lite path algorithm can correctly find a path from start to goal and handle dynamic environments"""
        # Scenario 1: Pathfinding in an open area
        # Reset the map
        self.grid_map.reset()

        # No obstacles between start and goal
        start = (10, 10)
        goal = (50, 50)

        # Call D* Lite algorithm to find path
        path = self.grid_map._dstar_lite_path(start, goal)

        # Verify path exists
        self.assertTrue(len(path) > 0, "Should find a path in an open area")
        # Verify start and end points of the path are correct
        self.assertEqual(path[0], start, "Path start point incorrect")
        self.assertEqual(path[-1], goal, "Path end point incorrect")
        # Verify path continuity
        self.verify_path_continuity(path)

        # Scenario 2: Pathfinding with obstacles
        # Reset the map
        self.grid_map.reset()

        # Create a horizontal wall with a passage
        for x in range(20, 40):
            for y in range(29, 31):
                if not (x == 30):  # Leave a passage at x=30
                    self.grid_map.update_cell(x, y, True)  # Mark as obstacle

        start = (25, 25)
        goal = (25, 35)

        # Call D* Lite algorithm to find path
        path = self.grid_map._dstar_lite_path(start, goal)

        # Verify path exists and goes through the passage
        self.assertTrue(len(path) > 0, "Should find a path around obstacles")
        # Verify start and end points of the path are correct
        self.assertEqual(path[0], start, "Path start point incorrect")
        self.assertEqual(path[-1], goal, "Path end point incorrect")
        # Verify path continuity
        self.verify_path_continuity(path)
        # Check if path goes through the designated passage (x=30)
        passage_used = False
        for x, y in path:
            if x == 30 and 29 <= y <= 31:
                passage_used = True
                break
        self.assertTrue(passage_used, "Path should go through the designated passage")

        # Scenario 3: Unreachable goal
        # Reset the map
        self.grid_map.reset()

        # Create a completely enclosed area
        for x in range(60, 80):
            for y in range(60, 80):
                if x == 60 or x == 79 or y == 60 or y == 79:
                    self.grid_map.update_cell(x, y, True)  # Create a wall

        start = (65, 65)  # Inside point
        goal = (90, 90)  # Outside point

        # Call D* Lite algorithm to find path
        path = self.grid_map._dstar_lite_path(start, goal)

        # Verify that no path can be found
        self.assertEqual(
            len(path), 0, "Points inside an enclosed area cannot reach outside"
        )

        # Scenario 4: Dynamic environment pathfinding (advantage of D* Lite)
        # Reset the map
        self.grid_map.reset()

        start = (20, 20)
        goal = (70, 70)

        # Get initial path
        initial_path = self.grid_map._dstar_lite_path(start, goal)
        self.assertTrue(
            len(initial_path) > 0, "Should find a path in an obstacle-free environment"
        )

        # Add new obstacles along the path
        mid_point = initial_path[len(initial_path) // 2]

        # Add obstacle wall around midpoint
        wall_x = mid_point[0]
        for y in range(mid_point[1] - 10, mid_point[1] + 10):
            self.grid_map.update_cell(wall_x, y, True)

        # Use the same start and goal, replan the path
        new_path = self.grid_map._dstar_lite_path(start, goal)

        # Verify new path exists
        self.assertTrue(
            len(new_path) > 0,
            "Should be able to replan path after new obstacles appear",
        )
        # Verify new path avoids obstacles
        no_collision = True
        for x, y in new_path:
            if self.grid_map.is_occupied(x, y):
                no_collision = False
                break
        self.assertTrue(no_collision, "New path should avoid all obstacles")

        # Verify new path is different from the original (should go around new obstacles)
        if len(initial_path) == len(new_path):
            paths_different = False
            for i in range(len(initial_path)):
                if initial_path[i] != new_path[i]:
                    paths_different = True
                    break
            self.assertTrue(
                paths_different, "New path should be different from the original path"
            )
        else:
            # If path lengths differ, the path has changed
            pass

    def verify_path_continuity(self, path):
        """Verify path continuity, ensuring the distance between adjacent path points is reasonable"""
        for i in range(1, len(path)):
            prev_x, prev_y = path[i - 1]
            curr_x, curr_y = path[i]

            # Calculate distance (Chebyshev or Manhattan distance)
            distance = max(abs(curr_x - prev_x), abs(curr_y - prev_y))

            # Ensure distance between adjacent points doesn't exceed 1 (diagonal distance is √2, but max should not exceed 1.5)
            self.assertLessEqual(
                distance,
                1.5,
                f"Distance between path points {path[i-1]} and {path[i]} is unreasonable",
            )

    def test_node_navigation(self):
        """Test the navigation between nodes in the navigation graph"""
        # Reset the map
        self.grid_map.reset()

        # Create a simple navigation graph
        # Add nodes
        self.grid_map.nav_nodes = {
            "node_0": (0, 0, 0, "node_0"),  # (pose_index, x, y)
            "node_1": (1, 50, 0, "node_1"),  # right
            "node_2": (2, 0, 50, "node_2"),  # down
            "node_3": (3, 50, 50, "node_3"),  # right-down
            "node_4": (4, 100, 100, "node_4"),  # further away
            "node_5": (5, 150, 50, "node_5"),  # further right
            "node_6": (6, 100, 0, "node_6"),  # right of node_1
        }

        # Create a more complex topology with different edge costs
        self.grid_map._add_nav_edge("node_0", "node_1", 50)  # Direct path
        self.grid_map._add_nav_edge("node_0", "node_2", 50)  # Direct path
        self.grid_map._add_nav_edge("node_1", "node_3", 50)  # Direct path
        self.grid_map._add_nav_edge("node_2", "node_3", 50)  # Direct path
        self.grid_map._add_nav_edge("node_3", "node_4", 70.7)  # Direct diagonal path
        self.grid_map._add_nav_edge("node_3", "node_5", 100)  # Direct path
        self.grid_map._add_nav_edge("node_1", "node_6", 50)  # Direct path
        self.grid_map._add_nav_edge("node_6", "node_5", 70.7)  # Direct diagonal path
        self.grid_map._add_nav_edge("node_4", "node_5", 70.7)  # Direct diagonal path

        # Test 1: Navigate between directly connected nodes
        path = self.grid_map.node_navigation("node_0", "node_1")
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], (0, 0))  # Start at node_0
        self.assertEqual(path[-1], (50, 0))  # End at node_1

        # Test 2: Navigate to the same node
        path = self.grid_map.node_navigation("node_0", "node_0")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], (0, 0))  # Only node_0

        # Test 3: Navigate through multiple nodes - verify shortest path
        # Path: node_0 -> node_1 -> node_3 -> node_4 (total cost: 50 + 50 + 70.7 = 170.7)
        # vs node_0 -> node_2 -> node_3 -> node_4 (total cost: 50 + 50 + 70.7 = 170.7)
        # Both paths have the same cost
        path = self.grid_map.node_navigation("node_0", "node_4")
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], (0, 0))  # Start at node_0
        self.assertEqual(path[-1], (100, 100))  # End at node_4

        # Get the nodes in the path
        path_nodes = self._extract_nodes_from_path(path)

        # Either path is valid since both have the same cost
        valid_path1 = ["node_0", "node_1", "node_3", "node_4"]
        valid_path2 = ["node_0", "node_2", "node_3", "node_4"]

        self.assertTrue(
            path_nodes == valid_path1 or path_nodes == valid_path2,
            f"Path should be one of the shortest paths, got: {path_nodes}",
        )

        # Test 4: Navigate through nodes with multiple possible paths - verify shortest path
        # Path: node_0 -> node_1 -> node_6 -> node_5 (total cost: 50 + 50 + 70.7 = 170.7)
        # vs node_0 -> node_1 -> node_3 -> node_5 (total cost: 50 + 50 + 100 = 200)
        # vs node_0 -> node_2 -> node_3 -> node_5 (total cost: 50 + 50 + 100 = 200)

        path = self.grid_map.node_navigation("node_0", "node_5")
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], (0, 0))  # Start at node_0
        self.assertEqual(path[-1], (150, 50))  # End at node_5

        # Get the nodes in the path
        path_nodes = self._extract_nodes_from_path(path)

        # Check that the algorithm found the shortest path
        expected_path = ["node_0", "node_1", "node_6", "node_5"]
        self.assertEqual(
            path_nodes,
            expected_path,
            f"A* should find the shortest path. Expected {expected_path}, got {path_nodes}",
        )

        # Test 5: Test with non-existent nodes
        with self.assertRaises(ValueError):
            self.grid_map.node_navigation("node_0", "non_existent_node")

        with self.assertRaises(ValueError):
            self.grid_map.node_navigation("non_existent_node", "node_0")

        # Test 6: Test with disconnected nodes
        # Create an isolated node
        self.grid_map.nav_nodes["isolated_node"] = (7, 200, 200, "isolated_node")

        # Try to navigate to the isolated node
        path = self.grid_map.node_navigation("node_0", "isolated_node")
        self.assertEqual(len(path), 0)  # No path should be found

        # Test 7: Create a more complex graph with cycles
        self.grid_map._add_nav_edge("node_6", "node_4", 50)  # Add a shortcut

        # Path: node_0 -> node_1 -> node_6 -> node_4 (total cost: 50 + 50 + 50 = 150)
        # vs previous paths to node_4 (total cost: 170.7)

        path = self.grid_map.node_navigation("node_0", "node_4")
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)

        path_nodes = self._extract_nodes_from_path(path)

        # Manually calculate costs to verify
        total_cost = 0
        for i in range(len(path_nodes) - 1):
            node1 = path_nodes[i]
            node2 = path_nodes[i + 1]
            if node2 in self.grid_map.nav_edges[node1]:
                edge_cost = self.grid_map.nav_edges[node1][node2]
                total_cost += edge_cost

        # Expected path node_0 -> node_1 -> node_6 -> node_4 (should be 150)
        # vs node_0 -> node_1/node_2 -> node_3 -> node_4 (would be 170.7)
        expected_path = ["node_0", "node_1", "node_6", "node_4"]

        # Check that the algorithm found the new shortest path
        self.assertEqual(
            path_nodes,
            expected_path,
            f"Should find new shortest path after adding shortcut. Expected {expected_path}, got {path_nodes}",
        )

        # Test 8: Test pose_navigation functionality
        # Test navigating from arbitrary world coordinates
        path = self.grid_map.pose_navigation(10, 10, 90, 90)
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertAlmostEqual(path[0][0], 10, delta=1)  # Start x close to 10
        self.assertAlmostEqual(path[0][1], 10, delta=1)  # Start y close to 10
        self.assertAlmostEqual(path[-1][0], 90, delta=1)  # End x close to 90
        self.assertAlmostEqual(path[-1][1], 90, delta=1)  # End y close to 90

    def _extract_nodes_from_path(self, path):
        """Helper method to extract node IDs from a path of coordinates"""
        node_path = []

        # Create a mapping from coordinates to node IDs for easy lookup
        coord_to_node = {}
        for node_id, (_, x, y, _) in self.grid_map.nav_nodes.items():
            coord_to_node[(x, y)] = node_id

        # Extract nodes from the path
        for point in path:
            if point in coord_to_node:
                node_path.append(coord_to_node[point])

        return node_path


if __name__ == "__main__":
    unittest.main()
