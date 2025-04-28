# The MIT License (MIT)
# Copyright © 2025 Eastworld AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import re
import asyncio
import json
import math
import os
import random
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Annotated, Dict, List, Optional, Set, Tuple, TypedDict, Union

import bittensor as bt
import openai
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from eastworld.base.miner import BaseMinerNeuron
from eastworld.miner.slam.grid import ANONYMOUS_NODE_PREFIX
from eastworld.miner.slam.isam import ISAM2
from eastworld.protocol import Observation

SENSOR_MAX_RANGE = 50.0


class EnhancedJSONFileMemory:
    """Enhanced memory system with semantic search capabilities"""
    
    memory: dict
    embedding_cache: dict  # Cache for embeddings to avoid recomputation
    memory_queue: deque  # Queue for memory management
    max_memories: int = 100  # Maximum number of memories to keep per category
    
    def __init__(self, file_path: str, llm_client=None):
        self.file_path = file_path
        self.llm_client = llm_client
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.memory = None
        self.embedding_cache = {}
        self.memory_queue = deque(maxlen=200)
        self.last_save_time = time.time()
        self.save_interval = 30  # Save every 30 seconds
        
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.memory = json.load(f)
            except json.JSONDecodeError:
                bt.logging.error("Memory file corrupted, creating new memory")
        
        if self.memory is None:
            self.memory = {
                "goals": [],
                "plans": [],
                "reflections": [],
                "logs": [],
                "semantic_memories": [],
                "risk_assessments": [],
                "landmark_metadata": {},
            }
        
        # Ensure all required keys exist
        required_keys = ["goals", "plans", "reflections", "logs", "semantic_memories", 
                         "risk_assessments", "landmark_metadata"]
        for key in required_keys:
            if key not in self.memory:
                self.memory[key] = [] if key != "landmark_metadata" else {}
    
    def save(self, force=False):
        """Save memory to file with throttling to prevent excessive writes"""
        current_time = time.time()
        if force or (current_time - self.last_save_time) > self.save_interval:
            try:
                with open(self.file_path, "w") as f:
                    json.dump(self.memory, f, indent=2)
                self.last_save_time = current_time
            except Exception as e:
                bt.logging.error(f"Failed to save memory: {e}")
    
    async def get_embedding(self, text: str) -> list:
        """Get embedding for text using LLM API"""
        if not text:
            return [0] * 768  # Default embedding size
        
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            if self.llm_client:
                response = await self.llm_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                )
                embedding = response.data[0].embedding
                self.embedding_cache[text] = embedding
                return embedding
            else:
                # Fallback to simple hash-based embedding if no LLM client
                return self._simple_embedding(text)
        except Exception as e:
            bt.logging.error(f"Failed to get embedding: {e}")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> list:
        """Simple hash-based embedding as fallback"""
        # Use hash of words as a simple embedding
        words = text.lower().split()
        hash_values = [hash(word) % 768 for word in words]
        # Pad or truncate to 768 dimensions
        while len(hash_values) < 768:
            hash_values.append(0)
        return hash_values[:768]
    
    def semantic_search(self, query: str, category: str = "semantic_memories", top_k: int = 3) -> list:
        """Search for similar memories based on semantic similarity"""
        if not self.memory[category]:
            return []
        
        # Use executor for synchronous execution in async context
        future = self.executor.submit(self._semantic_search_sync, query, category, top_k)
        return future.result()
    
    def _semantic_search_sync(self, query: str, category: str, top_k: int) -> list:
        """Synchronous version of semantic search"""
        query_embedding = self._simple_embedding(query)
        
        # Get embeddings for all memories in the category
        memory_embeddings = []
        for idx, memory in enumerate(self.memory[category]):
            if "embedding" in memory:
                memory_embeddings.append((idx, memory["embedding"]))
            else:
                # Generate embedding if not already cached
                embedding = self._simple_embedding(memory["content"])
                memory["embedding"] = embedding
                memory_embeddings.append((idx, embedding))
        
        if not memory_embeddings:
            return []
        
        # Calculate similarity scores
        similarities = []
        for idx, embedding in memory_embeddings:
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((idx, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K results
        return [self.memory[category][idx] for idx, score in similarities[:top_k]]
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        try:
            # Ensure vectors are the same length
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(x * x for x in vec1))
            magnitude2 = math.sqrt(sum(x * x for x in vec2))
            
            if magnitude1 * magnitude2 == 0:
                return 0.0
                
            return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            bt.logging.error(f"Cosine similarity calculation error: {e}")
            return 0.0
    
    def add_semantic_memory(self, content: str, metadata: dict = None):
        """Add a new semantic memory with metadata"""
        if not content:
            return
            
        memory_entry = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "embedding": self._simple_embedding(content)
        }
        
        self.memory["semantic_memories"].append(memory_entry)
        self.memory_queue.append(("semantic_memories", len(self.memory["semantic_memories"]) - 1))
        
        # Prune if needed
        self._prune_memories("semantic_memories")
    
    def add_risk_assessment(self, location: tuple, risk_level: float, description: str):
        """Add a risk assessment for a specific location"""
        assessment = {
            "location": location,
            "risk_level": risk_level,
            "description": description,
            "timestamp": time.time()
        }
        
        self.memory["risk_assessments"].append(assessment)
        self._prune_memories("risk_assessments")
    
    def get_nearest_risk_assessment(self, location: tuple, radius: float = 10.0) -> dict:
        """Get the nearest risk assessment within a radius"""
        # Ensure risk_assessments key exists
        if "risk_assessments" not in self.memory:
            self.memory["risk_assessments"] = []
            return None
            
        if not self.memory["risk_assessments"]:
            return None
            
        nearest = None
        min_distance = float('inf')
        
        try:
            for assessment in self.memory["risk_assessments"]:
                distance = math.sqrt(
                    (location[0] - assessment["location"][0]) ** 2 + 
                    (location[1] - assessment["location"][1]) ** 2
                )
                
                if distance <= radius and distance < min_distance:
                    min_distance = distance
                    nearest = assessment
        except Exception as e:
            bt.logging.error(f"Error in get_nearest_risk_assessment: {e}")
            return None
                
        return nearest
    
    def push_reflection(self, reflection: str):
        """Add a new reflection with semantic indexing"""
        self.memory["reflections"].append(reflection)
        if len(self.memory["reflections"]) > 20:
            self.memory["reflections"] = self.memory["reflections"][-10:]
            
        # Also add as semantic memory for better retrieval
        self.add_semantic_memory(reflection, {"type": "reflection"})

    def push_log(self, action: str):
        """Add a new action log entry"""
        log = {
            "action": action.strip(),
            "feedback": "",
            "repeat_times": 1,
        }
        self.memory["logs"].append(log)
        if len(self.memory["logs"]) > 100:
            self.memory["logs"] = self.memory["logs"][-60:]

    def update_log(self, feedback: str):
        """Update the latest log with feedback"""
        if not self.memory["logs"]:
            # Miner may have restarted and the last action is lost
            return

        last_log = self.memory["logs"][-1]
        if last_log["feedback"]:
            # The last log already has feedback, unexpected behavior
            return
        last_log["feedback"] = feedback.strip()

        # Try to merge the last two logs if they are the same
        if len(self.memory["logs"]) < 2:
            return
        
        previous_log = self.memory["logs"][-2]
        if (
            previous_log["action"] == last_log["action"]
            and previous_log["feedback"] == last_log["feedback"]
        ):
            # Merge the two logs with the same action and feedback
            previous_log["repeat_times"] += 1
            self.memory["logs"].pop()
            
        # Add to semantic memories if contains important information
        if feedback and len(feedback) > 10:
            self.add_semantic_memory(
                f"Action: {last_log['action']} - Result: {feedback}", 
                {"type": "action_result"}
            )
    
    def update_landmark_metadata(self, landmark_id: str, metadata: dict):
        """Update metadata for a specific landmark"""
        if landmark_id not in self.memory["landmark_metadata"]:
            self.memory["landmark_metadata"][landmark_id] = {}
            
        self.memory["landmark_metadata"][landmark_id].update(metadata)
    
    def get_landmark_metadata(self, landmark_id: str) -> dict:
        """Get metadata for a specific landmark"""
        return self.memory["landmark_metadata"].get(landmark_id, {})
    
    def _prune_memories(self, category: str):
        """Prune memories if they exceed the maximum limit"""
        if len(self.memory[category]) > self.max_memories:
            # Keep the most recent memories
            self.memory[category] = self.memory[category][-self.max_memories:]


class ActionCache:
    """Action caching system to optimize repetitive actions and reduce LLM calls"""
    
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.cache = {}  # {state_hash: action}
        self.timestamps = {}  # {state_hash: last_access_time}
        self.usage_counts = {}  # {state_hash: usage_count}
    
    def _compute_state_hash(self, state: dict) -> str:
        """Compute a hash for the current state"""
        # Extract relevant features from the state for hashing
        relevant_features = {
            "environment": state.get("environment", ""),
            "lidar": tuple(sorted([f"{d}:{dist}" for d, dist, *_ in state.get("lidar", [])]))
        }
        return hash(str(relevant_features))
    
    def get_action(self, state: dict) -> dict:
        """Get cached action for a state if it exists"""
        state_hash = self._compute_state_hash(state)
        if state_hash in self.cache:
            self.timestamps[state_hash] = time.time()
            self.usage_counts[state_hash] = self.usage_counts.get(state_hash, 0) + 1
            return self.cache[state_hash]
        return None
    
    def cache_action(self, state: dict, action: dict):
        """Cache an action for a state"""
        state_hash = self._compute_state_hash(state)
        
        # If cache is full, evict least recently used item
        if len(self.cache) >= self.capacity:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.cache.pop(oldest_key)
            self.timestamps.pop(oldest_key)
            self.usage_counts.pop(oldest_key, None)
        
        self.cache[state_hash] = action
        self.timestamps[state_hash] = time.time()
        self.usage_counts[state_hash] = 1
    
    def is_reliable_action(self, state: dict) -> bool:
        """Check if the cached action for a state is reliable (used multiple times)"""
        state_hash = self._compute_state_hash(state)
        return state_hash in self.usage_counts and self.usage_counts[state_hash] >= 3


class ParallelProcessor:
    """Handles parallel processing of tasks to optimize performance"""
    
    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.async_tasks = {}
        self.results = {}
    
    async def run_in_parallel(self, tasks: dict):
        """Run multiple tasks in parallel
        
        Args:
            tasks: Dictionary of {task_name: task_coroutine}
        """
        # Convert all tasks to asyncio.Task objects
        async_tasks = {}
        for name, task in tasks.items():
            if asyncio.iscoroutine(task):
                async_tasks[name] = asyncio.create_task(task)
            else:
                # For non-coroutines, run in ThreadPoolExecutor
                loop = asyncio.get_event_loop()
                async_tasks[name] = loop.run_in_executor(self.executor, task)
        
        # Wait for all tasks to complete
        self.async_tasks = async_tasks
        self.results = {}
        
        # Process results as they become available
        for name, task in async_tasks.items():
            try:
                result = await task
                self.results[name] = {"success": True, "data": result}
            except Exception as e:
                bt.logging.error(f"Task {name} failed: {e}")
                self.results[name] = {"success": False, "error": str(e)}
        
        return self.results
    
    def cancel_all_tasks(self):
        """Cancel all running tasks"""
        for task in self.async_tasks.values():
            if not task.done():
                task.cancel()


class AgentState(TypedDict):
    """Type definition for the agent state graph"""
    observation: Annotated[Observation, lambda a, b: b or a]
    navigation_locations: Annotated[list[str], lambda a, b: b or a]
    perception_state: Annotated[dict, lambda a, b: b or a]
    perception_entities: Annotated[list, lambda a, b: b or a]
    reflection: Annotated[str, lambda a, b: b or a]
    action: Annotated[dict, lambda a, b: b or a]
    cached_action: Annotated[bool, lambda a, b: a or b]
    risks: Annotated[list, lambda a, b: a + b]
    errors: Annotated[set[str], lambda a, b: a.union(b)]


class SemanticSLAM:
    """Semantic SLAM system that integrates both geometric and semantic information"""
    
    def __init__(self, isam: ISAM2, memory: EnhancedJSONFileMemory):
        self.isam = isam
        self.memory = memory
        self.landmark_confidence = {}  # Confidence scores for landmarks
        self.semantic_features = {}  # Semantic features associated with landmarks
    
    def update(self, lidar_data: dict, odometry: float, odometry_direction: str, 
               semantic_info: dict = None) -> tuple:
        """Update SLAM with new sensor data and semantic information"""
        try:
            if odometry > 0:
                self.isam.run_iteration(lidar_data, odometry, odometry_direction)
                
                # Get current pose
                x, y, theta = self.isam.get_current_pose()
                
                # Update navigation topology with current pose
                self.isam.grid_map.update_nav_topo(
                    self.isam.pose_index, x, y, allow_isolated=True
                )
                
                # Update semantic information if available
                if semantic_info:
                    self._update_semantic_features(x, y, semantic_info)
                
                return x, y, theta
        except Exception as e:
            bt.logging.error(f"SemanticSLAM update error: {e}")
            traceback.print_exc()
        
        # Return last known pose if update failed
        return self.isam.get_current_pose()
    
    def _update_semantic_features(self, x: float, y: float, semantic_info: dict):
        """Update semantic features associated with the current position"""
        try:
            # Find nearest landmarks
            nearby_nodes = self.isam.grid_map.get_nav_nodes(x, y, 20.0)
            
            for node_id in nearby_nodes:
                if node_id not in self.semantic_features:
                    self.semantic_features[node_id] = []
                
                # Add semantic information to node
                if "objects" in semantic_info:
                    features = [obj.strip() for obj in semantic_info["objects"].split(",")]
                    self.semantic_features[node_id].extend(features)
                    
                    # Limit number of features per node
                    if len(self.semantic_features[node_id]) > 10:
                        self.semantic_features[node_id] = self.semantic_features[node_id][-10:]
                
                # Update landmark metadata
                if not node_id.startswith(ANONYMOUS_NODE_PREFIX):
                    self.memory.update_landmark_metadata(node_id, {
                        "last_visit": time.time(),
                        "semantic_features": self.semantic_features[node_id],
                        "position": (x, y)
                    })
        except Exception as e:
            bt.logging.error(f"Semantic feature update error: {e}")
    
    def get_semantic_features(self, node_id: str) -> list:
        """Get semantic features associated with a landmark"""
        return self.semantic_features.get(node_id, [])
    
    def find_landmarks_by_features(self, features: list) -> list:
        """Find landmarks that contain specific semantic features"""
        matching_landmarks = []
        
        for node_id, node_features in self.semantic_features.items():
            if any(feature in node_features for feature in features):
                matching_landmarks.append(node_id)
        
        return matching_landmarks


class RiskAwareNavigator:
    """Navigation system with risk awareness and path optimization"""
    
    def __init__(self, slam: SemanticSLAM, memory: EnhancedJSONFileMemory):
        self.slam = slam
        self.memory = memory
        self.risk_map = {}  # {(x, y): risk_level}
        self.path_history = deque(maxlen=20)  # Store recent paths for learning
        self.directions = [
            "north", "northeast", "east", "southeast", 
            "south", "southwest", "west", "northwest"
        ]
    
    def update_risk(self, x: float, y: float, risk_level: float, description: str = ""):
        """Update risk level for a location"""
        grid_x, grid_y = int(x // 5) * 5, int(y // 5) * 5  # Quantize to 5x5 grid
        self.risk_map[(grid_x, grid_y)] = risk_level
        
        # Also store in memory for persistence
        self.memory.add_risk_assessment((grid_x, grid_y), risk_level, description)
    
    def get_risk(self, x: float, y: float) -> float:
        """Get risk level for a location"""
        # Check exact match first
        grid_x, grid_y = int(x // 5) * 5, int(y // 5) * 5  # Quantize to 5x5 grid
        if (grid_x, grid_y) in self.risk_map:
            return self.risk_map[(grid_x, grid_y)]
        
        # Check nearby cells with decay
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_x, check_y = grid_x + dx*5, grid_y + dy*5
                if (check_x, check_y) in self.risk_map:
                    # Risk decays with distance
                    distance = math.sqrt(dx**2 + dy**2)
                    decay_factor = max(0, 1 - (distance / 3))
                    return self.risk_map[(check_x, check_y)] * decay_factor
        
        return 0.0  # No risk information
    
    def navigate_to(self, current_x: float, current_y: float, target_node: str) -> tuple:
        """Navigate to a target node with risk awareness"""
        # Get target node data
        node = self.slam.isam.grid_map.nav_nodes.get(target_node)
        if node is None:
            bt.logging.error(f"Navigation target node {target_node} not found")
            return None, 0
        
        # Find path
        path = self.slam.isam.grid_map.pose_navigation(
            current_x, current_y, node[0], node[1]
        )
        
        if not path or len(path) < 2:
            bt.logging.error(f"No path found to target node {target_node}")
            return None, 0
        
        # Check for risks along the path
        risky_path = False
        for point in path[1:3]:  # Check next 2 points
            risk = self.get_risk(point[0], point[1])
            if risk > 0.7:  # High risk threshold
                risky_path = True
                break
        
        if risky_path:
            bt.logging.warning(f"Risky path detected to {target_node}, finding alternative")
            # TODO: Implement alternative path finding with risk avoidance
        
        # Get next waypoint
        next_node = path[1]
        
        # Calculate direction and distance
        direction = self._relative_direction(
            current_x, current_y, next_node[0], next_node[1]
        )
        distance = self._relative_distance(
            current_x, current_y, next_node[0], next_node[1]
        )
        
        # Remember this path for learning
        self.path_history.append({
            "start": (current_x, current_y),
            "target": (node[0], node[1]),
            "path": path,
            "timestamp": time.time()
        })
        
        return direction, distance
    
    def _relative_direction(
        self, origin_x: float, origin_y: float, target_x: float, target_y: float
    ) -> str:
        """Calculate the relative direction from origin to target"""
        dx = target_x - origin_x
        dy = target_y - origin_y
        angle = (180 + (180 / 3.14) * math.atan2(dy, dx)) % 360
        angle = (angle + 22.5) % 360
        index = int(angle / 45)
        return self.directions[index]
    
    def _relative_distance(
        self, origin_x: float, origin_y: float, target_x: float, target_y: float
    ) -> float:
        """Calculate the relative distance from origin to target"""
        return math.sqrt((target_x - origin_x) ** 2 + (target_y - origin_y) ** 2)
    
    def get_nearest_landmark(self, x: float, y: float, max_distance: float = 50.0) -> str:
        """Find the nearest named landmark"""
        min_distance = float('inf')
        nearest_landmark = None
        
        for node_id, node_data in self.slam.isam.grid_map.nav_nodes.items():
            if node_id.startswith(ANONYMOUS_NODE_PREFIX):
                continue
                
            distance = math.sqrt((x - node_data[0])**2 + (y - node_data[1])**2)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_landmark = node_id
        
        return nearest_landmark


class MultiModalPerception:
    """Multi-modal perception system that fuses data from different sensors"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.last_perception = {}
        self.entity_memory = {}  # Tracks entities across frames
        self.confidence_threshold = 0.7
        
        # Cache detected entities to avoid redundant processing
        self.entity_cache = {}
        self.cache_ttl = 5  # Cache TTL in steps
        self.current_step = 0
    
    def process_lidar(self, lidar_data: list) -> dict:
        """Process LiDAR data to extract obstacle information"""
        processed_data = {}
        
        for data in lidar_data:
            if data[1][-1] == "+":
                processed_data[data[0]] = SENSOR_MAX_RANGE + 1
            else:
                processed_data[data[0]] = float(data[1].split("m")[0])
        
        # Calculate spatial metrics
        metrics = {
            "open_directions": [d for d, dist in processed_data.items() if dist > 15.0],
            "blocked_directions": [d for d, dist in processed_data.items() if dist < 5.0],
            "average_distance": sum(processed_data.values()) / len(processed_data) if processed_data else 0,
            "min_distance": min(processed_data.values()) if processed_data else 0,
            "max_distance": max(processed_data.values()) if processed_data else 0,
            "is_open_space": all(dist > 10.0 for dist in processed_data.values()) if processed_data else False,
            "is_narrow": any(dist < 5.0 for dist in processed_data.values()) if processed_data else False,
        }
        
        return {
            "raw": processed_data,
            "metrics": metrics
        }
    
    async def process_visual(self, environment_text: str, objects_text: str) -> dict:
        """Process visual information from environment and objects text"""
        # Increment step counter
        self.current_step += 1
        
        # Check cache first
        cache_key = hash(f"{environment_text}|{objects_text}")
        if cache_key in self.entity_cache and (self.current_step - self.entity_cache[cache_key]["step"]) < self.cache_ttl:
            return self.entity_cache[cache_key]["data"]
        
        entities = []
        relations = []
        
        try:
            # Extract entities and their properties
            entity_texts = objects_text.split(",")
            for entity_text in entity_texts:
                entity_text = entity_text.strip()
                if not entity_text:
                    continue
                    
                # Simple parsing for demonstration
                parts = entity_text.split(":")
                if len(parts) >= 2:
                    entity_name = parts[0].strip()
                    entity_desc = parts[1].strip()
                    
                    entity = {
                        "id": f"entity_{hash(entity_name) % 10000}",
                        "name": entity_name,
                        "description": entity_desc,
                        "confidence": 0.9,
                    }
                    entities.append(entity)
            
            # Extract environment context
            context = {
                "general_description": environment_text,
                "lighting": "unknown",
                "weather": "unknown",
                "terrain": "unknown",
            }
            
            # Extract keywords from environment description
            keywords = []
            for keyword in ["dark", "light", "rain", "fog", "rocky", "sandy", "metal", "wooden"]:
                if keyword in environment_text.lower():
                    keywords.append(keyword)
            
            result = {
                "entities": entities,
                "relations": relations,
                "context": context,
                "keywords": keywords,
            }
            
            # Cache the result
            self.entity_cache[cache_key] = {
                "data": result,
                "step": self.current_step
            }
            
            return result
            
        except Exception as e:
            bt.logging.error(f"Visual processing error: {e}")
            return {
                "entities": [],
                "relations": [],
                "context": {"general_description": environment_text},
                "keywords": [],
                "error": str(e)
            }
    
    async def fuse_perceptions(self, lidar_data: list, environment_text: str, 
                              objects_text: str, interactions: list) -> dict:
        """Fuse data from multiple perception sources"""
        lidar_results = self.process_lidar(lidar_data)
        visual_results = await self.process_visual(environment_text, objects_text)
        
        # Process interactions
        interaction_entities = []
        for interaction in interactions:
            if len(interaction) >= 2:
                interaction_entities.append({
                    "id": f"interaction_{hash(interaction[0]) % 10000}",
                    "type": interaction[0],
                    "details": interaction[1] if len(interaction) > 1 else ""
                })
        
        # Fuse spatial awareness from LiDAR with entity recognition from vision
        entities_with_positions = []
        for entity in visual_results["entities"]:
            # Try to associate entities with directions from LiDAR
            # This is a simple heuristic that could be improved
            entity_with_pos = entity.copy()
            entity_with_pos["position"] = "unknown"
            
            # Simple keyword matching to assign directions
            for direction in lidar_results["raw"].keys():
                if direction.lower() in entity["description"].lower():
                    entity_with_pos["position"] = direction
                    entity_with_pos["distance"] = lidar_results["raw"][direction]
                    break
            
            entities_with_positions.append(entity_with_pos)
        
        # Identify risks from perception
        risks = []
        for entity in entities_with_positions:
            # Identify potential threats or obstacles
            threat_keywords = ["danger", "hostile", "weapon", "broken", "unstable", "sharp"]
            is_threat = any(keyword in entity["description"].lower() for keyword in threat_keywords)
            
            if is_threat:
                risk = {
                    "source": entity["name"],
                    "level": 0.8,  # High risk
                    "description": f"Potential threat detected: {entity['description']}",
                    "position": entity.get("position", "unknown")
                }
                risks.append(risk)
            
        # Update entity memory for tracking across frames
        for entity in entities_with_positions:
            entity_id = entity["id"]
            if entity_id in self.entity_memory:
                # Update existing entity
                self.entity_memory[entity_id]["last_seen"] = self.current_step
                self.entity_memory[entity_id]["confidence"] = min(
                    1.0, self.entity_memory[entity_id]["confidence"] + 0.1
                )
            else:
                # New entity
                self.entity_memory[entity_id] = {
                    "first_seen": self.current_step,
                    "last_seen": self.current_step,
                    "confidence": 0.6,  # Initial confidence
                    "data": entity
                }
        
        # Clean up entity memory (remove old entities)
        for entity_id in list(self.entity_memory.keys()):
            if self.current_step - self.entity_memory[entity_id]["last_seen"] > 20:
                del self.entity_memory[entity_id]
        
        # Persistent entities (seen multiple times)
        persistent_entities = [
            mem["data"] for entity_id, mem in self.entity_memory.items()
            if mem["confidence"] > self.confidence_threshold
        ]
        
        result = {
            "spatial": lidar_results["metrics"],
            "entities": entities_with_positions,
            "persistent_entities": persistent_entities,
            "interactions": interaction_entities,
            "context": visual_results["context"],
            "keywords": visual_results["keywords"],
            "risks": risks
        }
        
        self.last_perception = result
        return result


class SeniorAgent(BaseMinerNeuron):
    """Enhanced Senior Agent with layered architecture and multi-modal perception"""
    
    def __init__(
        self, config=None, slam_data: str = None, memory_file_path: str = "memory.json"
    ):
        super(SeniorAgent, self).__init__(config=config)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.step = 0
        
        # Initialize LLM client
        self.llm = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1",api_key="sk-or-v1-888888888888888888")
        #
        self.llm2 = openai.AsyncOpenAI(base_url="https://open.888888888888.cn/api/paas/v4",api_key="88888888888888888.8888")
        self.model_small = "google/gemini-2.5-flash-preview"
        self.model_medium = "glm-z1-airx"
        self.model_large = "glm-z1-airx"
        
        # Initialize memory system
        self.memory = EnhancedJSONFileMemory(memory_file_path, self.llm)
        if not self.memory.memory["goals"]:
            self._init_memory()
        
        # Initialize SLAM system
        if slam_data is None:
            isam = ISAM2(load_data=False, data_dir="slam_data")
        else:
            isam = ISAM2(load_data=True, data_dir=slam_data)
        
        # Initialize perception, planning and execution systems
        self.semantic_slam = SemanticSLAM(isam, self.memory)
        self.perception = MultiModalPerception(self.llm)
        self.navigator = RiskAwareNavigator(self.semantic_slam, self.memory)
        self.action_cache = ActionCache(capacity=50)
        self.parallel_processor = ParallelProcessor(max_workers=3)
        
        # Build the processing graph
        self.graph = self._build_graph()
        
        # Load prompts
        self._load_prompts()
        
        # Load action space
        self._load_action_space()
        
        # State for fault tolerance and recovery
        self.last_successful_state = None
        self.failure_count = 0
        self.recovery_mode = False
        
        # Variables for `maze_run`
        self.maze_run_explore_direction = "north"
        self.maze_run_counter = 0
        self.landmark_annotation_step = 0
    
    def _init_memory(self):
        """Initialize memory with default goals and plans"""
        self.memory.memory["goals"] = [
            "Work alongside your team to accomplish critical objectives",
            "Venture deep into the uncharted canyon to scavenge vital components for your mothership's repairs",
        ]
        self.memory.memory["plans"] = [
            "Talk to your team to understand the current situation and the objectives",
            "Explore unknown areas to supplement data for navigation systems",
        ]
    
    def _load_prompts(self):
        """Load all prompt templates"""
        prompt_dir = "eastworld/miner/prompts"
        
        self.landmark_annotation_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_landmark_annotation.txt")
        )
        self.after_action_review_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_after_action_review.txt")
        )
        self.grounding_learning_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_grounding_learning.txt")
        )
        self.objective_reevaluation_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_objective_reevaluation.txt")
        )
        self.action_selection_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_action_selection.txt")
        )
    
    def _load_action_space(self):
        """Load available actions from local_actions.json"""
        try:
            with open("eastworld/miner/local_actions.json", "r") as f:
                self.local_action_space = json.load(f)
        except Exception as e:
            bt.logging.error(f"Failed to load local actions: {e}")
            self.local_action_space = []
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the agent's processing graph"""
        graph_builder = StateGraph(AgentState)
        
        # Perception layer
        graph_builder.add_node("Localization & Mapping", self.localization_mapping)
        graph_builder.add_node("Multi-Modal Perception", self.multi_modal_perception)
        graph_builder.add_node("Landmark Annotation", self.landmark_annotation)
        graph_builder.add_node("Risk Assessment", self.risk_assessment)
        
        # Planning layer
        graph_builder.add_node("After-Action Review", self.after_action_review)
        graph_builder.add_node("Grounding & Learning", self.grounding_learning)
        graph_builder.add_node("Objective Reevaluation", self.objective_reevaluation)
        graph_builder.add_node("Action Selection", self.action_selection)
        
        # Execution layer
        graph_builder.add_node("Action Execution", self.action_execution)
        graph_builder.add_node("Execution Monitoring", self.execution_monitoring)
        
        # Fault tolerance layer
        graph_builder.add_node("Error Recovery", self.error_recovery)
        
        # Define our conditional routing function
        def has_errors(state: AgentState):
            if state["errors"]:
                return "Error Recovery"
            else:
                return "Multi-Modal Perception"
        
        def continue_or_error(state: AgentState, destination: str):
            if state["errors"]:
                return "Error Recovery"
            else:
                return destination
        
        # Connect the nodes with simple and conditional edges
        graph_builder.add_edge(START, "Localization & Mapping")
        
        # Add conditional routing from Localization & Mapping
        graph_builder.add_conditional_edges(
            "Localization & Mapping",
            lambda state: "Error Recovery" if state["errors"] else "Multi-Modal Perception"
        )
        
        # Add conditional edges for the rest of the nodes
        graph_builder.add_conditional_edges(
            "Multi-Modal Perception",
            lambda state: "Error Recovery" if state["errors"] else "Landmark Annotation"
        )
        
        graph_builder.add_conditional_edges(
            "Landmark Annotation",
            lambda state: "Error Recovery" if state["errors"] else "Risk Assessment"
        )
        
        graph_builder.add_conditional_edges(
            "Risk Assessment",
            lambda state: "Error Recovery" if state["errors"] else "After-Action Review"
        )
        
        graph_builder.add_conditional_edges(
            "After-Action Review",
            lambda state: "Error Recovery" if state["errors"] else "Grounding & Learning"
        )
        
        graph_builder.add_conditional_edges(
            "Grounding & Learning",
            lambda state: "Error Recovery" if state["errors"] else "Objective Reevaluation"
        )
        
        graph_builder.add_conditional_edges(
            "Objective Reevaluation",
            lambda state: "Error Recovery" if state["errors"] else "Action Selection"
        )
        
        graph_builder.add_conditional_edges(
            "Action Selection",
            lambda state: "Error Recovery" if state["errors"] else "Action Execution"
        )
        
        graph_builder.add_conditional_edges(
            "Action Execution",
            lambda state: "Error Recovery" if state["errors"] else "Execution Monitoring"
        )
        
        # Final nodes end the graph
        graph_builder.add_edge("Execution Monitoring", END)
        graph_builder.add_edge("Error Recovery", END)
        
        return graph_builder.compile()
    
    async def forward(self, synapse: Observation) -> Observation:
        """Process an observation and return an action"""
        try:
            self.step += 1
            config = RunnableConfig(
                configurable={"thread_id": f"step_{self.uid}_{self.step}"}
            )
            
            # Create initial state
            initial_state = AgentState(
                observation=synapse,
                navigation_locations=[],
                perception_state={"spatial": {"is_narrow": False}},  # Initialize with default values
                perception_entities=[],
                reflection="",
                action={},
                cached_action=False,
                risks=[],
                errors=set()
            )
            
            # Try cached action first for efficiency
            bt.logging.info(f"last_action: {synapse}")
            cached_state = self._try_cached_action(initial_state)
            if cached_state["cached_action"]:
                bt.logging.info(f"Using cached action: {cached_state['action']}")
                # Only update mem
                if synapse.action_log:
                    self.memory.update_log(synapse.action_log[0])
                self.memory.save()
                synapse.action = [cached_state["action"]]
                return synapse
            
            # Process through the graph
            state = await self.graph.ainvoke(initial_state, config)
            
            # If we got valid action, update the cache
            if state["action"] and "errors" not in state:
                self.action_cache.cache_action(
                    self._extract_state_features(synapse),
                    state["action"]
                )
            
            # If errors occurred, use fallback action
            if state["errors"]:
                bt.logging.error(
                    f"Errors in LLM Graph: {len(state['errors'])}. Fallback to risk-aware navigation"
                )
                action = self._generate_fallback_action(synapse, state["errors"])
            else:
                action = state["action"]
            
            # Update memory and save
            self.memory.save()
            
            # Return action to validator
            bt.logging.info(f">> Agent Action: {action}")
            synapse.action = [action]
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Critical error in forward: {e}")
            traceback.print_exc()
            
            # Emergency fallback
            direction, distance = self.maze_run(synapse)
            action = {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
            synapse.action = [action]
            return synapse
    
    def _try_cached_action(self, state: AgentState) -> AgentState:
        """Try to use a cached action if available and reliable"""
        synapse = state["observation"]
        state_features = self._extract_state_features(synapse)
        
        # Only use cache if the action is reliable (used multiple times successfully)
        if self.action_cache.is_reliable_action(state_features):
            cached_action = self.action_cache.get_action(state_features)
            if cached_action:
                state["action"] = cached_action
                state["cached_action"] = True
        
        return state
    
    def _extract_state_features(self, synapse: Observation) -> dict:
        """Extract key features from observation for action caching"""
        features = {
            "environment": synapse.perception.environment,
            "lidar": [(direction, distance) for direction, distance, *_ in synapse.sensor.lidar],
            "odometry": synapse.sensor.odometry
        }
        return features
    
    def _generate_fallback_action(self, synapse: Observation, errors: set) -> dict:
        """Generate fallback action based on the current state and errors"""
        self.failure_count += 1
        self.recovery_mode = True
        
        # Log errors for analysis
        for error in errors:
            bt.logging.error(f"Error leading to fallback: {error}")
        
        # If we have a critical number of failures, use the simplest navigation
        if self.failure_count > 5:
            bt.logging.warning("Critical failure count reached, using basic maze run")
            direction, distance = self.maze_run(synapse)
            return {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
        
        # Check if we're near a known landmark
        try:
            x, y, _ = self.semantic_slam.isam.get_current_pose()
            nearest_landmark = self.navigator.get_nearest_landmark(x, y, 50.0)
            
            # If we found a landmark, try to move away from it to somewhere new
            if nearest_landmark:
                bt.logging.info(f"In recovery mode, moving away from {nearest_landmark}")
                
                # Get landmark position
                node = self.semantic_slam.isam.grid_map.nav_nodes.get(nearest_landmark)
                if node:
                    # Move in the opposite direction of the landmark
                    direction = self.navigator._relative_direction(
                        node[0], node[1], x, y
                    )
                    return {
                        "name": "move_in_direction",
                        "arguments": {"direction": direction, "distance": 20},
                    }
        except Exception as e:
            bt.logging.error(f"Error in fallback landmark navigation: {e}")
        
        # If all else fails, use maze run
        direction, distance = self.maze_run(synapse)
        return {
            "name": "move_in_direction",
            "arguments": {"direction": direction, "distance": distance},
        }
    
    async def localization_mapping(self, state: AgentState) -> AgentState:
        """Process sensor data for localization and mapping"""
        bt.logging.debug(">> Localization & Mapping")
        try:
            synapse: Observation = state["observation"]
            lidar_data = {}
            odometry = 0
            odometry_direction = ""

            # Process lidar data
            for data in synapse.sensor.lidar:
                if data[1][-1] == "+":
                    lidar_data[data[0]] = max(
                        float(data[1].split("m")[0]), SENSOR_MAX_RANGE + 1
                    )
                else:
                    lidar_data[data[0]] = float(data[1].split("m")[0])
            
            # Process odometry data
            odometry_direction = synapse.sensor.odometry[0]
            odometry = float(synapse.sensor.odometry[1].split("m")[0])

            # Update SLAM with new sensor data
            if odometry > 0:
                try:
                    bt.logging.debug(
                        f"SLAM: {lidar_data} {odometry} {odometry_direction}"
                    )
                    
                    # Update semantic SLAM
                    semantic_info = {
                        "environment": synapse.perception.environment,
                        "objects": synapse.perception.objects
                    }
                    x, y, theta = self.semantic_slam.update(
                        lidar_data, odometry, odometry_direction, semantic_info
                    )
                    
                    bt.logging.debug(f"SLAM: Current pose: {x} {y} {theta}")
                    
                    # Reset recovery mode if successful
                    if self.recovery_mode:
                        self.failure_count = max(0, self.failure_count - 1)
                        if self.failure_count == 0:
                            self.recovery_mode = False
                    
                except Exception as e:
                    bt.logging.error(f"SLAM Error: {e}")
                    traceback.print_exc()
                    state["errors"].add(f"SLAM Error: {e}")

            # Get all labeled navigation nodes
            nav_nodes_labeled_all = [
                f"{node_id} : {node_data[3]}"
                for node_id, node_data in self.semantic_slam.isam.grid_map.nav_nodes.items()
                if not node_id.startswith(ANONYMOUS_NODE_PREFIX)
            ]
            state["navigation_locations"] = nav_nodes_labeled_all
            
            # Save last successful state for recovery
            self.last_successful_state = {
                "x": x if 'x' in locals() else 0,
                "y": y if 'y' in locals() else 0,
                "theta": theta if 'theta' in locals() else 0,
                "nav_nodes": nav_nodes_labeled_all
            }
            
        except Exception as e:
            bt.logging.error(f"Localization & Mapping Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Localization & Mapping Error: {e}")
        finally:
            return state
    
    async def multi_modal_perception(self, state: AgentState) -> AgentState:
        """Process multi-modal perception data"""
        bt.logging.debug(">> Multi-Modal Perception")
        try:
            synapse: Observation = state["observation"]
            
            # Initialize perception state with defaults in case processing fails
            state["perception_state"] = {
                "spatial": {
                    "open_directions": [],
                    "blocked_directions": [],
                    "average_distance": 0,
                    "min_distance": 0,
                    "max_distance": 0,
                    "is_open_space": False,
                    "is_narrow": False
                },
                "context": {"general_description": ""},
                "keywords": []
            }
            
            # Process perception in parallel
            try:
                perception_result = await self.perception.fuse_perceptions(
                    synapse.sensor.lidar,
                    synapse.perception.environment,
                    synapse.perception.objects,
                    synapse.perception.interactions
                )
                
                # Store perception state
                state["perception_state"] = perception_result
                state["perception_entities"] = perception_result.get("entities", [])
                
                # Add risks from perception
                state["risks"] = perception_result.get("risks", [])
            except Exception as e:
                bt.logging.error(f"Perception processing error: {e}")
                # We already have default values, so we can continue
                state["perception_entities"] = []
                state["risks"] = []
            
        except Exception as e:
            bt.logging.error(f"Multi-Modal Perception Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Multi-Modal Perception Error: {e}")
        finally:
            return state
    
    async def landmark_annotation(self, state: AgentState) -> AgentState:
        """Annotate landmarks in the environment"""
        bt.logging.debug(">> Landmark Annotation")
        try:
            synapse: Observation = state["observation"]
            if (
                self.step - self.landmark_annotation_step < 5
                or synapse.sensor.odometry[1] == "0m"
            ):
                # No need to annotate landmarks on every step
                return state

            self.landmark_annotation_step = self.step

            x, y, theta = self.semantic_slam.isam.get_current_pose()
            nav_nodes = self.semantic_slam.isam.grid_map.get_nav_nodes(x, y, 40.0)
            nav_nodes_labeled = [
                node_id
                for node_id in nav_nodes
                if not node_id.startswith(ANONYMOUS_NODE_PREFIX)
            ]

            prompt_context = {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "anonymous_landmark_count": len(nav_nodes) - len(nav_nodes_labeled),
                "labeled_landmark_count": len(nav_nodes_labeled),
                "labeled_landmark_list": ", ".join(nav_nodes_labeled),
                "labeled_landmark_all": "\n".join(
                    [f"  - {k}" for k in state["navigation_locations"]]
                ),
                "sensor_readings": "\n".join(
                    [f"  - {', '.join(items)}" for items in synapse.sensor.lidar]
                ),
                "environment": synapse.perception.environment,
                "objects": synapse.perception.objects,
            }
            prompt = self.landmark_annotation_prompt.format(**prompt_context)
            bt.logging.debug(f"Landmark Annotation Prompt: {prompt}")
            response = await self.llm.chat.completions.create(
                model=self.model_small,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_content = response.choices[0].message.content
            cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            bt.logging.debug(f"Landmark Annotation Response: {cleaned_content}")
            node_data = cleaned_content.splitlines()
            node_id = node_data[0].strip()
            node_desc = node_data[1].strip() if len(node_data) > 1 else ""
            if node_id != "NA":
                self.semantic_slam.isam.grid_map.update_nav_topo(
                    self.semantic_slam.isam.pose_index,
                    x,
                    y,
                    node_id=node_id,
                    node_desc=node_desc,
                    allow_isolated=True,
                )
                
                # Also store landmark metadata
                self.memory.update_landmark_metadata(node_id, {
                    "description": node_desc,
                    "position": (x, y),
                    "created_at": time.time()
                })
                
        except Exception as e:
            bt.logging.error(f"Landmark Annotation Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Landmark Annotation Error: {e}")
        finally:
            return state
    
    async def risk_assessment(self, state: AgentState) -> AgentState:
        """Assess risks in the environment"""
        bt.logging.debug(">> Risk Assessment")
        try:
            synapse: Observation = state["observation"]
            
            # Get current position
            x, y, _ = self.semantic_slam.isam.get_current_pose()
            
            # Process risks from perception
            for risk in state["risks"]:
                try:
                    # Add risk to navigator's risk map
                    risk_level = risk.get("level", 0.5)
                    description = risk.get("description", "")
                    self.navigator.update_risk(x, y, risk_level, description)
                except Exception as e:
                    bt.logging.error(f"Error processing risk: {e}")
                    # Continue with next risk
            
            # Environment-based risk assessment
            try:
                # Fix the type error - checking correctly for is_narrow boolean property
                spatial = state["perception_state"].get("spatial", {})
                if spatial.get("is_narrow", False):
                    self.navigator.update_risk(x, y, 0.6, "Narrow passage detected")
            except Exception as e:
                bt.logging.error(f"Error in environment risk assessment: {e}")
                
            # Get existing risk assessment for this location
            try:
                existing_risk = self.memory.get_nearest_risk_assessment((x, y), 10.0)
                if existing_risk:
                    bt.logging.info(f"Known risk at current location: {existing_risk['description']}")
                    state["risks"].append({
                        "source": "memory",
                        "level": existing_risk["risk_level"],
                        "description": existing_risk["description"],
                        "position": "current"
                    })
            except Exception as e:
                bt.logging.error(f"Error getting existing risk assessment: {e}")
                
        except Exception as e:
            bt.logging.error(f"Risk Assessment Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Risk Assessment Error: {e}")
        finally:
            return state

    async def after_action_review(self, state: AgentState) -> AgentState:
        """Review the outcome of the last action and current situation"""
        bt.logging.debug(">> After-Action Review")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]

            if synapse.action_log:
                self.memory.update_log(synapse.action_log[0])

            # Prepare context for reflection
            recent_reflections = ""
            for idx, r in enumerate(self.memory.memory["reflections"][-1:]):
                recent_reflections += f"  {idx + 1}. {r}\n"
                
            recent_action_log = ""
            for idx, l in enumerate(self.memory.memory["logs"][-6:]):
                repeat_str = (
                    f" (repeated {l['repeat_times']} times)"
                    if l["repeat_times"] > 1
                    else ""
                )
                recent_action_log += f"\n  - Log {idx + 1}\n    Action: {l['action']} {repeat_str}\n    Result: {l['feedback']}"

            # Build action space description
            action_space = ""
            for act in [*synapse.action_space, *self.local_action_space]:
                action_space += (
                    f"  - {act['function']['name']}: {act['function']['description']}\n"
                )

            # Current spatial understanding
            spatial_data = ""
            if "spatial" in state["perception_state"]:
                spatial = state["perception_state"]["spatial"]
                spatial_data = (
                    f"  - Open directions: {', '.join(spatial.get('open_directions', []))}\n"
                    f"  - Blocked directions: {', '.join(spatial.get('blocked_directions', []))}\n"
                    f"  - Average distance to obstacles: {spatial.get('average_distance', 0):.1f}m\n"
                    f"  - Minimum distance to obstacles: {spatial.get('min_distance', 0):.1f}m\n"
                )

            # Include risk assessment
            risk_data = ""
            for idx, risk in enumerate(state["risks"]):
                risk_data += f"  - Risk {idx+1}: {risk.get('description', '')} (level: {risk.get('level', 0):.1f})\n"

            prompt_context = {
                "goals": "\n".join([f"  - {x}" for x in self.memory.memory["goals"]]),
                "plans": "\n".join([f"  - {x}" for x in self.memory.memory["plans"]]),
                "sensor_readings": "\n".join(
                    [f"  - {', '.join(items)}" for items in synapse.sensor.lidar]
                ),
                "odometry_reading": f"  - {', '.join(synapse.sensor.odometry)}",
                "perception": f"{synapse.perception.environment}\n{synapse.perception.objects}",
                "interaction": "\n".join(
                    [f"  - {', '.join(x)}" for x in synapse.perception.interactions]
                ),
                "items": "\n".join(
                    [
                        f"  - {item.name} x{item.count}: {item.description.strip()}"
                        for item in synapse.items
                    ]
                ),
                "navigation_locations": "\n".join(
                    [f"  - {x}" for x in state["navigation_locations"]]
                ),
                "action_space": action_space,
                "recent_reflections": recent_reflections,
                "recent_action_log": recent_action_log,
                "spatial_data": spatial_data,
                "risk_data": risk_data,
            }
            
            # Include additional context from perception
            additional_context = (
                f"\n- **Entities in View**:\n"
                f"  {', '.join([e.get('name', '') for e in state['perception_entities']])}\n"
                f"\n- **Spatial Analysis**:\n{spatial_data}\n"
                f"\n- **Risk Assessment**:\n{risk_data}\n"
            )
            prompt_context["perception"] += additional_context
            
            prompt = self.after_action_review_prompt.format(**prompt_context)
            bt.logging.debug(f"After Action Review Prompt: {prompt}")
            response = await self.llm2.chat.completions.create(
                model=self.model_large,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_content = response.choices[0].message.content
            cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            reflection = cleaned_content
            bt.logging.debug(f"After Action Review Response: {reflection}")
            state["reflection"] = reflection
            self.memory.push_reflection(reflection)
            
        except Exception as e:
            bt.logging.error(f"After Action Review Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"After Action Review Error: {e}")
        finally:
            return state

    async def grounding_learning(self, state: AgentState) -> AgentState:
        """Learn from experiences and adapt to the environment"""
        bt.logging.debug(">> Grounding & Learning")
        if state["errors"]:
            return state

        try:
            # Extract keywords and concepts from the reflection
            if state["reflection"]:
                # Store in semantic memory for retrieval
                self.memory.add_semantic_memory(
                    state["reflection"], 
                    {"type": "learning", "step": self.step}
                )
                
            # Update the action cache reliability based on action outcomes
            synapse = state["observation"]
            if synapse.action_log and len(self.memory.memory["logs"]) > 0:
                last_log = self.memory.memory["logs"][-1]
                
                # Check if the action was successful based on feedback
                success_keywords = ["successful", "found", "reached", "discovered", "acquired"]
                failure_keywords = ["failed", "blocked", "couldn't", "unable", "error"]
                
                was_successful = any(keyword in last_log["feedback"].lower() for keyword in success_keywords)
                was_failure = any(keyword in last_log["feedback"].lower() for keyword in failure_keywords)
                
                if was_successful or was_failure:
                    # Learn from this experience
                    experience = {
                        "action": last_log["action"],
                        "result": last_log["feedback"],
                        "successful": was_successful and not was_failure,
                        "context": {
                            "environment": synapse.perception.environment,
                            "objects": synapse.perception.objects,
                        }
                    }
                    
                    # Store in semantic memory
                    self.memory.add_semantic_memory(
                        f"Action: {experience['action']} - Result: {experience['result']}",
                        {"type": "experience", "successful": experience["successful"]}
                    )
            
            # Learn relationships between objects and landmarks
            if state["perception_entities"] and state["navigation_locations"]:
                x, y, _ = self.semantic_slam.isam.get_current_pose()
                nearest_landmark = self.navigator.get_nearest_landmark(x, y, 30.0)
                
                if nearest_landmark:
                    # Associate entities with this landmark
                    entities = [e.get("name", "") for e in state["perception_entities"] if e.get("name")]
                    if entities:
                        self.memory.update_landmark_metadata(nearest_landmark, {
                            "associated_entities": entities,
                            "last_entities_update": time.time()
                        })
                                                
        except Exception as e:
            bt.logging.error(f"Grounding & Learning Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Grounding & Learning Error: {e}")
        finally:
            return state

    async def objective_reevaluation(self, state: AgentState) -> AgentState:
        """Reevaluate objectives based on new information"""
        bt.logging.debug(">> Objective Reevaluation")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]

            # Retrieve similar past reflections for context
            similar_reflections = self.memory.semantic_search(
                state["reflection"], "semantic_memories", top_k=3
            )
            
            similar_reflection_text = ""
            for idx, ref in enumerate(similar_reflections):
                if "type" in ref.get("metadata", {}) and ref["metadata"]["type"] == "reflection":
                    similar_reflection_text += f"  {idx+1}. {ref['content']}\n"

            prompt_context = {
                "goals": "\n".join(self.memory.memory["goals"]),
                "plans": "\n".join(self.memory.memory["plans"]),
                "reflection": state["reflection"],
                "similar_reflections": similar_reflection_text,
            }
            prompt = self.objective_reevaluation_prompt.format(**prompt_context)
            #bt.logging.debug(f"Objective Reevaluation Prompt: {prompt}")
            response = await self.llm.chat.completions.create(
                model=self.model_small,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_content = response.choices[0].message.content
            cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            content = cleaned_content.split("\n\n")
            bt.logging.debug(f"Objective Reevaluation Response: {content}")
            
            if len(content) >= 2:
                new_goals = content[0].split("\n")
                new_plans = content[1].split("\n")
                
                self.memory.memory["goals"] = [
                    goal.strip()
                    for goal in new_goals
                    if goal.strip() and not goal.strip().startswith("#")
                ]
                self.memory.memory["plans"] = [
                    plan.strip()
                    for plan in new_plans
                    if plan.strip() and not plan.strip().startswith("#")
                ]
        except Exception as e:
            bt.logging.error(f"Objective Reevaluation Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Objective Reevaluation Error: {e}")
        finally:
            return state

    async def action_selection(self, state: AgentState) -> AgentState:
        """Select the most appropriate action based on current state"""
        bt.logging.debug(">> Action Selection")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]
            
            # Check if we should use cached action
            if state.get("cached_action", False):
                bt.logging.info("Using cached action")
                return state
            
            # Include risk assessment in the prompt
            risk_information = ""
            for risk in state["risks"]:
                risk_information += f"  - {risk.get('description', '')} (level: {risk.get('level', 0):.1f})\n"
            
            # Spatial awareness data
            spatial_data = ""
            if "spatial" in state["perception_state"]:
                spatial = state["perception_state"]["spatial"]
                spatial_data = (
                    f"  - Open directions: {', '.join(spatial.get('open_directions', []))}\n"
                    f"  - Blocked directions: {', '.join(spatial.get('blocked_directions', []))}\n"
                    f"  - In open space: {spatial.get('is_open_space', False)}\n"
                    f"  - In narrow passage: {spatial.get('is_narrow', False)}\n"
                )
                            
            # Get previous similar experiences
            prev_experiences = ""
            if len(self.memory.memory["logs"]) > 0:
                last_log = self.memory.memory["logs"][-1]
                similar_experiences = self.memory.semantic_search(
                    f"Action: {last_log['action']}", "semantic_memories", top_k=2
                )
                
                for idx, exp in enumerate(similar_experiences):
                    if "type" in exp.get("metadata", {}) and exp["metadata"]["type"] == "experience":
                        prev_experiences += f"  {idx+1}. {exp['content']}\n"

            prompt_context = {
                "goals": "\n".join([f"  - {x}" for x in self.memory.memory["goals"]]),
                "plans": "\n".join([f"  - {x}" for x in self.memory.memory["plans"]]),
                "reflection": state["reflection"],
                "sensor_readings": "\n".join(
                    [f"  - {', '.join(items)}" for items in synapse.sensor.lidar]
                ),
                "perception": (
                    f"{synapse.perception.environment}\n{synapse.perception.objects}\n"
                    f"\n**Spatial Analysis**:\n{spatial_data}\n"
                    f"\n**Risk Assessment**:\n{risk_information}\n"
                    f"\n**Previous Similar Experiences**:\n{prev_experiences}"
                ),
                "interaction": "\n".join(
                    [f"  - {', '.join(x)}" for x in synapse.perception.interactions]
                ),
                "items": "\n".join(
                    [
                        f"  - {item.name} x{item.count}: {item.description.strip()}"
                        for item in synapse.items
                    ]
                ),
                "navigation_locations": "\n".join(
                    [f"  - {x}" for x in state["navigation_locations"]]
                ),
                "entities": "\n".join([
                    f"  - {e.get('name', '')}: {e.get('description', '')}" 
                    for e in state["perception_entities"]
                ])
            }
            
            prompt = self.action_selection_prompt.format(**prompt_context)
            prompt += "\n\nIMPORTANT: You MUST select one of the provided tools/functions to execute the action. Do not provide code or text responses."
            #bt.logging.debug(f"Action Selection Prompt: {prompt}")
            
            # Try to use a faster model first, fallback to larger if needed
            try:
                response = await self.llm.chat.completions.create(
                    model=self.model_small,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[*synapse.action_space, *self.local_action_space],
                    tool_choice="auto", 
                    timeout=5  # Shorter timeout for faster response
                )
            except Exception as e:
                bt.logging.warning(f"Medium model failed, falling back to large: {e}")
                response = await self.llm.chat.completions.create(
                    model=self.model_small,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[*synapse.action_space, *self.local_action_space],
                    tool_choice="auto",  
                )

            bt.logging.debug(f"Action Selection Response: {response}")

            if response.choices[0].message.tool_calls:
                action = response.choices[0].message.tool_calls[0].function
                if action:
                    parsed_action = {
                        "name": action.name,
                        "arguments": json.loads(action.arguments),
                    }
                    state["action"] = parsed_action
                    self.memory.push_log(
                        f"{parsed_action['name']}, "
                        + ", ".join(
                            [f"{k}: {v}" for k, v in parsed_action["arguments"].items()]
                        )
                    )
            else:
                bt.logging.error("No tool call in response")
                state["errors"].add("No tool call in response")
                
        except Exception as e:
            bt.logging.error(f"Action Selection Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Action Selection Error: {e}")
        finally:
            return state

    def action_execution(self, state: AgentState) -> AgentState:
        """Execute the selected action or prepare it for execution"""
        bt.logging.debug(">> Action Execution")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]
            action = state["action"]
            
            if not action:
                bt.logging.error("No action to execute")
                state["errors"].add("No action to execute")
                return state

            if action["name"] == "explore_wall_following":
                direction, distance = self.maze_run(synapse)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
            elif action["name"] == "navigate_to":
                target = action["arguments"].get("target")
                
                # Validate target exists
                if not target or target not in [
                    node_id.split(" : ")[0] 
                    for node_id in state["navigation_locations"]
                ]:
                    bt.logging.error(f"Invalid navigation target: {target}")
                    state["errors"].add(f"Invalid navigation target: {target}")
                    return state
                
                # Get current position
                x, y, _ = self.semantic_slam.isam.get_current_pose()
                
                # Use the risk-aware navigator
                direction, distance = self.navigator.navigate_to(x, y, target)
                
                if direction:
                    state["action"] = {
                        "name": "move_in_direction",
                        "arguments": {"direction": direction, "distance": distance},
                    }
                else:
                    bt.logging.error(f"Navigation failed to {target}")
                    state["errors"].add(f"Navigation failed to {target}")
        except Exception as e:
            bt.logging.error(f"Action Execution Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Action Execution Error: {e}")
        finally:
            return state
    
    def execution_monitoring(self, state: AgentState) -> AgentState:
        """Monitor action execution and validate the result"""
        bt.logging.debug(">> Execution Monitoring")
        # This is a placeholder for future implementation
        # Could validate that the action makes sense given the current state
        return state
    
    def error_recovery(self, state: AgentState) -> AgentState:
        """Handle errors and recover from failures"""
        bt.logging.debug(">> Error Recovery")
        synapse = state["observation"]
        
        try:
            bt.logging.error(f"Errors detected, initiating recovery: {state['errors']}")
            
            # Count failure and enter recovery mode
            self.failure_count += 1
            self.recovery_mode = True
            
            # Log errors for investigation
            for error in state["errors"]:
                bt.logging.error(f"Recovery from: {error}")
                
            # Generate fallback action
            if "action" not in state or not state["action"]:
                direction, distance = self.maze_run(synapse)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
                
            # Save memory - important to preserve data even in failure cases
            self.memory.save(force=True)
            
        except Exception as e:
            bt.logging.critical(f"Error in recovery handler: {e}")
            traceback.print_exc()
            
            # Last resort emergency action
            direction, distance = "north", 10.0
            state["action"] = {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
        finally:
            return state
    
    def maze_run(self, synapse: Observation) -> tuple[str, float]:
        """Enhanced maze exploration strategy with risk awareness"""
        directions = [
            "north", "northeast", "east", "southeast", 
            "south", "southwest", "west", "northwest"
        ]
        
        # Process lidar readings
        r = [1] * len(directions)
        for data in synapse.sensor.lidar:
            r[directions.index(data[0])] = float(data[1].split("m")[0])

        # Get current position if available
        try:
            x, y, _ = self.semantic_slam.isam.get_current_pose()
            
            # Check for risks in potential directions
            risks = []
            for i, direction in enumerate(directions):
                # Skip if distance is too short
                if r[i] < 5:
                    risks.append(1.0)  # High risk (blocked)
                    continue
                
                # Calculate position in this direction
                angle = i * 45 * (3.14 / 180)  # Convert to radians
                dx = math.cos(angle) * min(r[i], 20)  # Cap at 20m
                dy = math.sin(angle) * min(r[i], 20)
                
                # Check risk at this position
                risk = self.navigator.get_risk(x + dx, y + dy)
                risks.append(risk)
            
            # Adjust readings based on risk
            for i in range(len(r)):
                risk_factor = 1.0 - risks[i]  # Invert risk (higher is better)
                r[i] = r[i] * risk_factor
        except Exception as e:
            bt.logging.debug(f"Error in risk-aware maze run: {e}")
            # Continue with standard maze run
            pass

        l = len(directions)
        cdi = directions.index(self.maze_run_explore_direction)
        ldi = (cdi - 1) % l
        rdi = (cdi + 1) % l
        bt.logging.debug(
            f"Current: {self.maze_run_explore_direction} {cdi}, Readings: {[f'{f:.01f}' for f in r]}"
        )
        
        wall_dist = 12 if sum((r[ldi], r[cdi], r[rdi])) / 3 > 20 else 6
        
        if r[rdi] < wall_dist:
            self.maze_run_counter = 0
            if r[cdi] < wall_dist:
                # Block by front and right: Turn left
                while r[ldi] < wall_dist:
                    ldi = (ldi - 1) % l
                choice = directions[ldi]
                distance = 5
            else:
                # Walls on right: Go straight
                choice = directions[cdi]
                distance = r[cdi] // random.randint(2, 4)
        else:
            self.maze_run_counter += 1
            # No walls on right side: Turn right
            choice = directions[rdi]
            distance = r[rdi] // random.randint(2, 4)

        if self.maze_run_counter > 10:
            # Break cycle with random exploration
            self.maze_run_counter = 0
            self.maze_run_explore_direction = random.choice(directions)
            return self.random_walk(synapse)
            
        # Update the exploration direction
        self.maze_run_explore_direction = choice
        
        # Cap distance for safety
        distance = min(max(distance, 5), 30)
        
        return choice, distance
    
    def random_walk(self, synapse: Observation) -> tuple[str, float]:
        """Random walk with weighted directional choices"""
        directions = [
            "north", "northeast", "east", "southeast", 
            "south", "southwest", "west", "northwest"
        ]
        weights = [1] * len(directions)

        # Update weights based on lidar data
        if synapse.sensor.lidar:
            readings = {}
            for data in synapse.sensor.lidar:
                readings[data[0]] = float(data[1].split("m")[0]) - 5.0

            for i, d in enumerate(directions):
                weights[i] = readings.get(d, 0) / 50.0

        # Avoid moving backwards
        if synapse.sensor.odometry[1] != "0m":
            i = directions.index(synapse.sensor.odometry[0])
            weights[(i + len(weights) // 2) % len(weights)] = 1e-6

        bt.logging.debug(f"Direction Weight: {[f'{f:.04f}' for f in weights]}")

        choice = random.choices(directions, weights=weights, k=1)[0]
        distance = random.randint(5, 30)

        return choice, distance
