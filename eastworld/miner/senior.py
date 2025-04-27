import json
import math
import os
import random
import traceback
from functools import lru_cache
from typing import Any, Annotated, Dict, List, Optional, Set, Tuple, TypedDict, Union
import asyncio
import concurrent.futures
import time

import bittensor as bt
import numpy as np
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
    """Enhanced memory system with semantic retrieval capabilities"""
    
    def __init__(self, file_path: str, embedding_model: str = "text-embedding-3-small"):
        self.file_path = file_path
        self.embedding_model = embedding_model
        self.memory = self._load_memory()
        self.embedding_cache = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file or initialize if not exists"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                bt.logging.error("Memory file corrupted, creating new memory")
        
        return {
            "goals": [],
            "plans": [],
            "reflections": [],
            "logs": [],
            "landmarks": [],
            "entities": [],
            "risk_factors": []
        }
        
    def save(self):
        """Save memory to file"""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            bt.logging.error(f"Failed to save memory: {e}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # Use executor to run sync code asynchronously
            client = openai.OpenAI()
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
            )
            embedding = response.data[0].embedding
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            bt.logging.error(f"Failed to get embedding: {e}")
            # Return a zero embedding as fallback
            return [0.0] * 1536
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0
    
    async def semantic_search(self, query: str, collection: str, limit: int = 5) -> List[Tuple[Any, float]]:
        """Search for semantically similar items in memory"""
        if collection not in self.memory or not self.memory[collection]:
            return []
        
        query_embedding = await self.get_embedding(query)
        results = []
        
        for item in self.memory[collection]:
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
            elif isinstance(item, str):
                text = item
            else:
                continue
            
            item_embedding = await self.get_embedding(text)
            similarity = self.cosine_similarity(query_embedding, item_embedding)
            results.append((item, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]
    
    async def retrieve_relevant_context(self, query: str, limit_per_collection: int = 3) -> Dict[str, List[Any]]:
        """Retrieve relevant context across multiple collections"""
        collections = ["goals", "plans", "reflections", "landmarks", "entities", "risk_factors"]
        results = {}
        
        for collection in collections:
            items = await self.semantic_search(query, collection, limit_per_collection)
            results[collection] = [item[0] for item in items]
            
        return results
    
    def push_reflection(self, reflection: str):
        """Add a reflection to memory"""
        self.memory["reflections"].append({
            "text": reflection.strip(),
            "timestamp": time.time()
        })
        if len(self.memory["reflections"]) > 20:
            self.memory["reflections"] = self.memory["reflections"][-10:]
    
    def push_log(self, action: str):
        """Add action log to memory"""
        log = {
            "action": action.strip(),
            "feedback": "",
            "repeat_times": 1,
            "timestamp": time.time()
        }
        self.memory["logs"].append(log)
        if len(self.memory["logs"]) > 100:
            self.memory["logs"] = self.memory["logs"][-60:]
    
    def update_log(self, feedback: str):
        """Update the most recent log with feedback"""
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
    
    def add_landmark(self, landmark_id: str, description: str, x: float, y: float):
        """Add or update a landmark in memory"""
        for landmark in self.memory["landmarks"]:
            if landmark["id"] == landmark_id:
                landmark["description"] = description
                landmark["x"] = x
                landmark["y"] = y
                return
        
        self.memory["landmarks"].append({
            "id": landmark_id,
            "description": description,
            "x": x,
            "y": y,
            "timestamp": time.time()
        })
    
    def add_entity(self, entity_id: str, entity_type: str, description: str):
        """Add or update an entity in memory"""
        for entity in self.memory["entities"]:
            if entity["id"] == entity_id:
                entity["type"] = entity_type
                entity["description"] = description
                entity["seen_last"] = time.time()
                entity["seen_count"] += 1
                return
        
        self.memory["entities"].append({
            "id": entity_id,
            "type": entity_type,
            "description": description,
            "seen_first": time.time(),
            "seen_last": time.time(),
            "seen_count": 1
        })
    
    def add_risk_factor(self, description: str, severity: float, location: Optional[str] = None):
        """Add a risk factor to memory"""
        self.memory["risk_factors"].append({
            "text": description,
            "severity": severity,
            "location": location,
            "timestamp": time.time()
        })
        if len(self.memory["risk_factors"]) > 50:
            self.memory["risk_factors"] = sorted(
                self.memory["risk_factors"], 
                key=lambda x: x["severity"] * (1.0 / (1.0 + (time.time() - x["timestamp"]) / 3600)),
                reverse=True
            )[:30]


class ActionCache:
    """Cache for action execution to optimize repeated actions"""
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
    
    @lru_cache(maxsize=32)
    def _get_cache_key(self, state_dict: Dict[str, Any]) -> str:
        """Create a cache key from state"""
        # Convert state to hashable form
        try:
            # Filter only elements we want to include in the key
            filtered_state = {
                "perception": state_dict.get("perception", {}),
                "lidar": state_dict.get("lidar", {}),
                "location": state_dict.get("location", (0, 0)),
            }
            return json.dumps(filtered_state, sort_keys=True)
        except:
            return str(hash(str(state_dict)))
    
    def get(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached action for a state"""
        key = self._get_cache_key(state)
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, state: Dict[str, Any], action: Dict[str, Any]) -> None:
        """Cache action for a state"""
        key = self._get_cache_key(state)
        self.cache[key] = action
        self.access_count[key] = 1
        
        if len(self.cache) > self.cache_size:
            # Evict least frequently used entry
            lfu_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[lfu_key]
            del self.access_count[lfu_key]


class AgentState(TypedDict):
    """State representation for the agent graph processing"""
    observation: Annotated[Observation, lambda a, b: b or a]
    navigation_locations: Annotated[list[str], lambda a, b: b or a]
    perception_data: Annotated[Dict[str, Any], lambda a, b: b or a]
    reflection: Annotated[str, lambda a, b: b or a]
    action: Annotated[dict, lambda a, b: b or a]
    errors: Annotated[set[str], lambda a, b: a.union(b)]
    risk_assessment: Annotated[Dict[str, Any], lambda a, b: b or a]
    context: Annotated[Dict[str, Any], lambda a, b: b or a]


class SeniorAgent(BaseMinerNeuron):
    """
    Enhanced Senior Agent implementation with hierarchical architecture
    
    Architecture:
    1. Perception Layer - Processes sensor data and environment
    2. Planning Layer - Strategic planning and decision making
    3. Execution Layer - Action execution and error handling
    """
    
    directions = [
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
    ]

    uid: int
    step: int
    graph: CompiledStateGraph
    slam: ISAM2
    llm: openai.AsyncOpenAI
    memory: EnhancedJSONFileMemory

    local_action_space: list[dict] = []

    def __init__(
        self, 
        config=None, 
        slam_data: str = None, 
        memory_file_path: str = "memory.json"
    ):
        super(SeniorAgent, self).__init__(config=config)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.step = 0
        
        # Initialize main components
        self._init_perception_system()
        self._init_planning_system()
        self._init_execution_system()
        
        # Build the processing graph
        self.graph = self._build_graph()
        
        # Initialize SLAM system
        if slam_data is None:
            self.slam = ISAM2(load_data=False, data_dir="slam_data")
        else:
            self.slam = ISAM2(load_data=True, data_dir=slam_data)
        
        # Initialize LLM client
        self.llm = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1",api_key="sk-or-v1-88888888888888")
        self.model_small = "google/gemini-2.0-flash-lite-001"
        self.model_medium = "google/gemini-2.5-flash-preview"
        self.model_large = "google/gemini-2.5-flash-preview"
        
        # Load prompt templates
        self._load_prompts()
        
        # Initialize enhanced memory
        self.memory = EnhancedJSONFileMemory(memory_file_path)
        if not self.memory.memory["goals"]:
            self._init_memory()
        
        # Load local action space
        with open("eastworld/miner/local_actions.json", "r") as f:
            self.local_action_space = json.load(f)
        
        # Initialize action cache
        self.action_cache = ActionCache()
        
        # Initialize navigation variables
        self.maze_run_explore_direction = "north"
        self.maze_run_counter = 0
        
        # Initialize parallel processor pool
        self.processor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def _init_perception_system(self):
        """Initialize perception system components"""
        self.landmark_annotation_step = 0
        self.entity_extraction_step = 0
        self.risk_assessment_threshold = 0.7
        
        # Visual perception related variables
        self.visual_features_cache = {}
    
    def _init_planning_system(self):
        """Initialize planning system components"""
        self.planning_horizon = 5  # Steps to look ahead
        self.uncertainty_threshold = 0.6  # Threshold for active learning
        self.exploration_rate = 0.2  # Base exploration rate
    
    def _init_execution_system(self):
        """Initialize execution system components"""
        self.fallback_strategies = [
            self.navigate_to_safety,
            self.maze_run,
            self.random_walk
        ]
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
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
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the hierarchical processing graph"""
        graph_builder = StateGraph(AgentState)
        
        # Perception layer
        graph_builder.add_node("Sensor Processing", self.sensor_processing)
        graph_builder.add_node("Semantic SLAM", self.semantic_slam)
        graph_builder.add_node("Environment Perception", self.environment_perception)
        graph_builder.add_node("Entity Extraction", self.entity_extraction)
        graph_builder.add_node("Risk Assessment", self.risk_assessment)
        
        # Planning layer
        graph_builder.add_node("After-Action Review", self.after_action_review)
        graph_builder.add_node("Grounding & Learning", self.grounding_learning)
        graph_builder.add_node("Context Retrieval", self.context_retrieval)
        graph_builder.add_node("Objective Reevaluation", self.objective_reevaluation)
        graph_builder.add_node("Action Selection", self.action_selection)
        
        # Execution layer
        graph_builder.add_node("Action Validation", self.action_validation)
        graph_builder.add_node("Action Execution", self.action_execution)
        graph_builder.add_node("Error Recovery", self.error_recovery)
        
        # Connect perception layer
        graph_builder.add_edge(START, "Sensor Processing")
        graph_builder.add_edge("Sensor Processing", "Semantic SLAM")
        graph_builder.add_edge("Semantic SLAM", "Environment Perception")
        graph_builder.add_edge("Environment Perception", "Entity Extraction")
        graph_builder.add_edge("Entity Extraction", "Risk Assessment")
        
        # Connect to planning layer
        graph_builder.add_edge("Risk Assessment", "After-Action Review")
        graph_builder.add_edge("After-Action Review", "Grounding & Learning")
        graph_builder.add_edge("After-Action Review", "Context Retrieval")
        graph_builder.add_edge("Context Retrieval", "Objective Reevaluation")
        graph_builder.add_edge("Grounding & Learning", "Objective Reevaluation")
        graph_builder.add_edge("Objective Reevaluation", "Action Selection")
        
        # Connect to execution layer
        graph_builder.add_edge("Action Selection", "Action Validation")
        graph_builder.add_edge("Action Validation", "Action Execution")
        graph_builder.add_edge("Action Validation", "Error Recovery")
        graph_builder.add_edge("Error Recovery", "Action Execution")
        
        # Define end points
        graph_builder.add_edge("Action Execution", END)
        graph_builder.add_edge("Grounding & Learning", END)
        
        return graph_builder.compile()
    
    async def forward(self, synapse: Observation) -> Observation:
        """Process the incoming observation and return an action"""
        self.step += 1
        config = RunnableConfig(
            configurable={"thread_id": f"step_{self.uid}_{self.step}"}
        )
        
        # Initialize state with observation and empty components
        initial_state = AgentState(
            observation=synapse, 
            navigation_locations=[], 
            perception_data={},
            reflection="", 
            action={}, 
            errors=set(),
            risk_assessment={},
            context={}
        )
        
        try:
            # Process through the graph
            state = await self.graph.ainvoke(initial_state, config)
            
            # If errors occurred, use fallback strategy
            if state["errors"]:
                bt.logging.error(
                    f"Errors in LLM Graph: {len(state['errors'])}. Fallback to recovery mechanism"
                )
                action = await self.execute_fallback_strategy(synapse, state["errors"])
            else:
                action = state["action"]
            
            # Save memory
            self.memory.save()
            
            # Return action in synapse
            bt.logging.info(f">> Agent Action: {action}")
            synapse.action = [action]
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Critical error in agent forward pass: {e}")
            traceback.print_exc()
            
            # Emergency fallback
            direction, distance = self.random_walk(synapse)
            action = {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
            synapse.action = [action]
            return synapse
    
    async def execute_fallback_strategy(self, observation: Observation, errors: Set[str]) -> Dict:
        """Execute fallback strategies in sequence until one succeeds"""
        self.recovery_attempts += 1
        
        for strategy in self.fallback_strategies:
            try:
                bt.logging.info(f"Trying fallback strategy: {strategy.__name__}")
                if strategy.__name__ == "navigate_to_safety":
                    result = await strategy(observation)
                else:
                    result = strategy(observation)
                    
                if isinstance(result, tuple) and len(result) == 2:
                    direction, distance = result
                    return {
                        "name": "move_in_direction",
                        "arguments": {"direction": direction, "distance": distance},
                    }
                elif isinstance(result, dict):
                    return result
            except Exception as e:
                bt.logging.error(f"Fallback strategy {strategy.__name__} failed: {e}")
        
        # Ultimate fallback - random direction with short distance
        bt.logging.warning("All fallback strategies failed, using random short move")
        direction = random.choice(self.directions)
        return {
            "name": "move_in_direction",
            "arguments": {"direction": direction, "distance": 5},
        }
    
    async def navigate_to_safety(self, observation: Observation) -> Dict:
        """Emergency navigation to a known safe location"""
        # Find the nearest safe location from memory
        if not self.memory.memory["landmarks"]:
            return None
        
        # Get current position
        try:
            current_x, current_y, _ = self.slam.get_current_pose()
            
            # Find nearest landmark that seems safe
            safe_locations = sorted(
                [l for l in self.memory.memory["landmarks"] 
                 if "dangerous" not in l["description"].lower()],
                key=lambda l: math.sqrt((l["x"] - current_x)**2 + (l["y"] - current_y)**2)
            )
            
            if safe_locations:
                safe_spot = safe_locations[0]
                return {
                    "name": "navigate_to",
                    "arguments": {"target": safe_spot["id"]},
                }
        except:
            pass
        
        return None

    # ==== Perception Layer Components ====
    
    def sensor_processing(self, state: AgentState) -> AgentState:
        """Process raw sensor data from LiDAR and odometry"""
        bt.logging.debug(">> Sensor Processing")
        try:
            synapse: Observation = state["observation"]
            
            # Process LiDAR data
            lidar_data = {}
            for data in synapse.sensor.lidar:
                if data[1][-1] == "+":
                    lidar_data[data[0]] = max(
                        float(data[1].split("m")[0]), SENSOR_MAX_RANGE + 1
                    )
                else:
                    lidar_data[data[0]] = float(data[1].split("m")[0])
            
            # Process odometry data
            odometry_direction = synapse.sensor.odometry[0]
            odometry_distance = float(synapse.sensor.odometry[1].split("m")[0])
            
            # Store processed data
            state["perception_data"]["lidar"] = lidar_data
            state["perception_data"]["odometry_direction"] = odometry_direction
            state["perception_data"]["odometry_distance"] = odometry_distance
            
            # Compute additional derived metrics
            state["perception_data"]["average_clearance"] = sum(lidar_data.values()) / len(lidar_data) if lidar_data else 0
            state["perception_data"]["min_clearance"] = min(lidar_data.values()) if lidar_data else 0
            state["perception_data"]["max_clearance"] = max(lidar_data.values()) if lidar_data else 0
            
            # Calculate movement properties
            state["perception_data"]["is_moving"] = odometry_distance > 0
            
        except Exception as e:
            bt.logging.error(f"Sensor Processing Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Sensor Processing Error: {e}")
        
        return state
    
    def semantic_slam(self, state: AgentState) -> AgentState:
        """Update semantic SLAM map and navigation information"""
        bt.logging.debug(">> Semantic SLAM")
        try:
            synapse: Observation = state["observation"]
            lidar_data = state["perception_data"]["lidar"]
            odometry_direction = state["perception_data"]["odometry_direction"]
            odometry_distance = state["perception_data"]["odometry_distance"]
            
            # Update SLAM if agent has moved
            if odometry_distance > 0:
                try:
                    bt.logging.debug(
                        f"SLAM: {lidar_data} {odometry_distance} {odometry_direction}"
                    )
                    # Run SLAM iteration with sensor data
                    self.slam.run_iteration(lidar_data, odometry_distance, odometry_direction)
                    
                    # Get updated pose
                    x, y, theta = self.slam.get_current_pose()
                    bt.logging.debug(f"SLAM: Update Navigation Topology: {x} {y}")
                    
                    # Update navigation topology
                    self.slam.grid_map.update_nav_topo(
                        self.slam.pose_index, x, y, allow_isolated=True
                    )
                    
                    # Store current position for other components
                    state["perception_data"]["position"] = (x, y, theta)
                    
                except Exception as e:
                    bt.logging.error(f"SLAM Error: {e}")
                    traceback.print_exc()
            
            # Get labeled navigation nodes
            nav_nodes_labeled_all = [
                f"{node_id} : {node_data[3]}"
                for node_id, node_data in self.slam.grid_map.nav_nodes.items()
                if not node_id.startswith(ANONYMOUS_NODE_PREFIX)
            ]
            state["navigation_locations"] = nav_nodes_labeled_all
            
            # Analyze map uncertainty areas
            # TODO: Implement uncertainty analysis for exploration
            
        except Exception as e:
            bt.logging.error(f"Semantic SLAM Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Semantic SLAM Error: {e}")
        
        return state
    
    async def environment_perception(self, state: AgentState) -> AgentState:
        """Process environment perception data"""
        bt.logging.debug(">> Environment Perception")
        try:
            synapse: Observation = state["observation"]
            
            # Extract environment description
            environment_text = synapse.perception.environment
            objects_text = synapse.perception.objects
            
            # Store raw perception data
            state["perception_data"]["environment_description"] = environment_text
            state["perception_data"]["objects_description"] = objects_text
            state["perception_data"]["full_perception"] = f"{environment_text}\n{objects_text}"
            
            # Process interaction data
            interactions = synapse.perception.interactions
            state["perception_data"]["interactions"] = [
                {"participants": interaction[:-1], "content": interaction[-1]}
                for interaction in interactions
            ]
            
            # Inventory analysis
            state["perception_data"]["inventory"] = [
                {
                    "name": item.name,
                    "count": item.count,
                    "description": item.description.strip()
                }
                for item in synapse.items
            ]
            
            # Perform environment feature extraction
            # This could be enhanced with visual input in the future
            features = await self._extract_environment_features(
                state["perception_data"]["full_perception"]
            )
            state["perception_data"]["environment_features"] = features
            
        except Exception as e:
            bt.logging.error(f"Environment Perception Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Environment Perception Error: {e}")
        
        return state
    
    async def entity_extraction(self, state: AgentState) -> AgentState:
        """Extract entities from perception data"""
        bt.logging.debug(">> Entity Extraction")
        try:
            # Skip if not enough steps have passed since last extraction
            if self.step - self.entity_extraction_step < 3:
                return state
                
            self.entity_extraction_step = self.step
            
            # Get perception data
            full_perception = state["perception_data"]["full_perception"]
            
            # Extract entities with LLM
            system_prompt = (
                "You are an entity extraction system. Identify entities in the text "
                "and return them in JSON format. Entity types: characters, objects, "
                "locations, hazards. Format each entity as: "
                "{'id': 'unique_name', 'type': 'entity_type', 'description': 'brief_description'}"
            )
            
            try:
                response = await self.llm.chat.completions.create(
                    model=self.model_small,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_perception}
                    ],
                    response_format={"type": "json_object"}
                )
                
                entities_json = response.choices[0].message.content
                entities = json.loads(entities_json).get("entities", [])
                
                # Store entities in state
                state["perception_data"]["entities"] = entities
                
                # Update memory with entities
                for entity in entities:
                    if "id" in entity and "type" in entity and "description" in entity:
                        self.memory.add_entity(
                            entity["id"], 
                            entity["type"], 
                            entity["description"]
                        )
                
            except Exception as e:
                bt.logging.error(f"Entity extraction LLM error: {e}")
        
        except Exception as e:
            bt.logging.error(f"Entity Extraction Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Entity Extraction Error: {e}")
        
        return state
    
    async def risk_assessment(self, state: AgentState) -> AgentState:
        """Assess risks in the current environment"""
        bt.logging.debug(">> Risk Assessment")
        try:
            # Skip if no position data (agent hasn't moved)
            if "position" not in state["perception_data"]:
                return state
                
            # Get perception and sensor data
            full_perception = state["perception_data"]["full_perception"]
            lidar_data = state["perception_data"]["lidar"]
            position = state["perception_data"]["position"]
            entities = state["perception_data"].get("entities", [])
            
            # Create risk assessment object
            risk_assessment = {
                "overall_risk": 0.0,
                "risk_factors": [],
                "safe_directions": [],
                "hazardous_directions": []
            }
            
            # Analyze LiDAR for navigational risks
            for direction, distance in lidar_data.items():
                if distance < 5.0:
                    risk_assessment["hazardous_directions"].append(direction)
                    risk_assessment["risk_factors"].append({
                        "type": "navigation",
                        "description": f"Very close obstacle in {direction} direction",
                        "severity": 0.8
                    })
                elif distance > 30.0:
                    risk_assessment["safe_directions"].append(direction)
            
            # Check for potential entity-based risks
            hazard_entities = [e for e in entities if e.get("type") == "hazard"]
            for hazard in hazard_entities:
                risk_assessment["risk_factors"].append({
                    "type": "entity",
                    "description": f"Hazard detected: {hazard.get('description', 'Unknown hazard')}",
                    "severity": 0.7
                })
            
            # Calculate overall risk
            if risk_assessment["risk_factors"]:
                avg_severity = sum(r["severity"] for r in risk_assessment["risk_factors"]) / len(risk_assessment["risk_factors"])
                risk_modifier = 1.0 - min(1.0, len(risk_assessment["safe_directions"]) / 8.0)
                risk_assessment["overall_risk"] = avg_severity * risk_modifier
            
            # Add significant risks to memory
            for risk in risk_assessment["risk_factors"]:
                if risk["severity"] > self.risk_assessment_threshold:
                    landmark_nearby = None
                    if state["navigation_locations"]:
                        landmark_nearby = state["navigation_locations"][0].split(" : ")[0]
                        
                    self.memory.add_risk_factor(
                        risk["description"],
                        risk["severity"],
                        landmark_nearby
                    )
            
            # Store assessment
            state["risk_assessment"] = risk_assessment
            
        except Exception as e:
            bt.logging.error(f"Risk Assessment Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Risk Assessment Error: {e}")
        
        return state
    
    async def _extract_environment_features(self, text: str) -> Dict[str, Any]:
        """Extract features from environment description"""
        # Simple keyword-based feature extraction
        features = {
            "terrain_type": "unknown",
            "visibility": "unknown",
            "weather": "unknown",
            "environment_type": "unknown",
            "notable_elements": []
        }
        
        # Extract terrain type
        terrain_keywords = {
            "rocky": ["rock", "rocky", "boulder", "crag"],
            "sandy": ["sand", "sandy", "dune", "desert"],
            "grassy": ["grass", "grassy", "vegetation", "plant"],
            "metallic": ["metal", "metallic", "steel", "iron"],
            "icy": ["ice", "icy", "frost", "frozen"],
            "watery": ["water", "puddle", "pool", "liquid"],
        }
        
        text_lower = text.lower()
        for terrain, keywords in terrain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                features["terrain_type"] = terrain
                break
                
        # Process for notable elements (simple approach)
        notable_elements = []
        for line in text.split("\n"):
            if ":" in line and len(line) < 100:  # Likely a key object description
                notable_elements.append(line.strip())
        
        features["notable_elements"] = notable_elements[:5]
        
        return features

    # ==== Planning Layer Components ====
    
    async def after_action_review(self, state: AgentState) -> AgentState:
        """Review previous actions and their outcomes"""
        bt.logging.debug(">> After-Action Review")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]

            # Update action log with feedback if available
            if synapse.action_log:
                self.memory.update_log(synapse.action_log[0])

            # Prepare recent reflections for context
            recent_reflections = ""
            for idx, r in enumerate(self.memory.memory["reflections"][-3:]):
                if isinstance(r, dict) and "text" in r:
                    recent_reflections += f"  {idx + 1}. {r['text']}\n"
                else:
                    recent_reflections += f"  {idx + 1}. {r}\n"
                    
            # Prepare recent action logs for context
            recent_action_log = ""
            for idx, l in enumerate(self.memory.memory["logs"][-10:]):
                repeat_str = (
                    f" (repeated {l['repeat_times']} times)"
                    if l.get('repeat_times', 1) > 1
                    else ""
                )
                recent_action_log += f"\n  - Log {idx + 1}\n    Action: {l['action']} {repeat_str}\n    Result: {l['feedback']}"

            # Prepare action space description
            action_space = ""
            for act in [*synapse.action_space, *self.local_action_space]:
                action_space += (
                    f"  - {act['function']['name']}: {act['function']['description']}\n"
                )

            # Prepare context for after action review
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
                "risk_assessment": self._format_risk_assessment(state["risk_assessment"]),
            }
            
            # Generate reflection
            prompt = self.after_action_review_prompt.format(**prompt_context)
            bt.logging.debug(f"After Action Review Prompt: {prompt}")
            
            response = await self.llm.chat.completions.create(
                model=self.model_large,
                messages=[{"role": "user", "content": prompt}],
            )

            # Process reflection
            reflection = response.choices[0].message.content.strip()
            bt.logging.debug(f"After Action Review Response: {reflection}")
            
            # Store reflection in state and memory
            state["reflection"] = reflection
            self.memory.push_reflection(reflection)
            
        except Exception as e:
            bt.logging.error(f"After Action Review Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"After Action Review Error: {e}")
            
        return state
    
    async def grounding_learning(self, state: AgentState) -> AgentState:
        """Process and learn from experiences to improve future decisions"""
        bt.logging.debug(">> Grounding & Learning")
        if state["errors"]:
            return state

        try:
            # Skip learning if no reflection available
            if not state["reflection"]:
                return state
                
            synapse: Observation = state["observation"]
            reflection = state["reflection"]
            
            # Process recent failures and successes to identify patterns
            recent_logs = self.memory.memory["logs"][-20:]
            success_patterns = []
            failure_patterns = []
            
            for log in recent_logs:
                action = log["action"]
                feedback = log["feedback"]
                
                # Simple heuristic to classify feedback
                is_success = any(kw in feedback.lower() for kw in ["success", "found", "completed", "acquired"])
                is_failure = any(kw in feedback.lower() for kw in ["cannot", "fail", "unable", "blocked", "no path"])
                
                if is_success:
                    success_patterns.append((action, feedback))
                elif is_failure:
                    failure_patterns.append((action, feedback))
            
            # Store learning patterns in state
            state["context"]["success_patterns"] = success_patterns[:5]
            state["context"]["failure_patterns"] = failure_patterns[:5]
            
            # TODO: Implement more sophisticated learning with embedding database
            # This is where we'd use vectorstore like Chroma or LancerDB to store and retrieve experiences
            
        except Exception as e:
            bt.logging.error(f"Grounding & Learning Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Grounding & Learning Error: {e}")
            
        return state
    
    async def context_retrieval(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from memory"""
        bt.logging.debug(">> Context Retrieval")
        if state["errors"]:
            return state

        try:
            # Create a query from current observation
            synapse: Observation = state["observation"]
            perception_text = f"{synapse.perception.environment}\n{synapse.perception.objects}"
            
            # Use the reflection and perception as query
            query = f"{state['reflection']}\n{perception_text}"
            
            # Retrieve relevant context
            relevant_context = await self.memory.retrieve_relevant_context(query)
            
            # Format the context for prompt use
            formatted_context = {
                "relevant_goals": self._format_items(relevant_context.get("goals", [])),
                "relevant_plans": self._format_items(relevant_context.get("plans", [])),
                "relevant_reflections": self._format_items(relevant_context.get("reflections", [])),
                "relevant_landmarks": self._format_items(relevant_context.get("landmarks", [])),
                "relevant_entities": self._format_items(relevant_context.get("entities", [])),
                "relevant_risks": self._format_items(relevant_context.get("risk_factors", [])),
            }
            
            # Store in state
            state["context"] = {**state["context"], **formatted_context}
            
        except Exception as e:
            bt.logging.error(f"Context Retrieval Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Context Retrieval Error: {e}")
            
        return state
    
    async def objective_reevaluation(self, state: AgentState) -> AgentState:
        """Reevaluate goals and plans based on current state"""
        bt.logging.debug(">> Objective Reevaluation")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]
            
            # Prepare context for objective reevaluation
            prompt_context = {
                "goals": "\n".join(self.memory.memory["goals"]),
                "plans": "\n".join(self.memory.memory["plans"]),
                "reflection": state["reflection"],
                "relevant_context": self._format_relevant_context(state["context"]),
                "risk_assessment": self._format_risk_assessment(state["risk_assessment"]),
            }
            
            # Generate updated goals and plans
            prompt = self.objective_reevaluation_prompt.format(**prompt_context)
            bt.logging.debug(f"Objective Reevaluation Prompt: {prompt}")
            
            response = await self.llm.chat.completions.create(
                model=self.model_large,
                messages=[{"role": "user", "content": prompt}],
            )

            # Process updated goals and plans
            content = response.choices[0].message.content.strip().split("\n\n")
            bt.logging.debug(f"Objective Reevaluation Response: {content}")
            
            # Extract and filter goals and plans
            new_goals = content[0].split("\n")
            new_plans = content[1].split("\n") if len(content) > 1 else []
            
            # Update memory with new goals and plans
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
            
        return state
    
    async def action_selection(self, state: AgentState) -> AgentState:
        """Select the most appropriate action based on current state"""
        bt.logging.debug(">> Action Selection")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]
            
            # Check action cache first
            cache_key_state = {
                "perception": synapse.perception.environment + synapse.perception.objects,
                "lidar": state["perception_data"]["lidar"] if "lidar" in state["perception_data"] else {},
                "location": state["perception_data"].get("position", (0, 0, 0))[:2],
            }
            
            cached_action = self.action_cache.get(cache_key_state)
            
            if cached_action:
                bt.logging.info("Using cached action")
                state["action"] = cached_action
                self.memory.push_log(
                    f"{cached_action['name']}, "
                    + ", ".join(
                        [f"{k}: {v}" for k, v in cached_action["arguments"].items()]
                    )
                )
                return state
            
            # Prepare context for action selection
            prompt_context = {
                "goals": "\n".join([f"  - {x}" for x in self.memory.memory["goals"]]),
                "plans": "\n".join([f"  - {x}" for x in self.memory.memory["plans"]]),
                "reflection": state["reflection"],
                "sensor_readings": "\n".join(
                    [f"  - {', '.join(items)}" for items in synapse.sensor.lidar]
                ),
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
                "risk_assessment": self._format_risk_assessment(state["risk_assessment"]),
                "relevant_context": self._format_relevant_context(state["context"]),
            }
            
            # Generate action selection
            prompt = self.action_selection_prompt.format(**prompt_context)
            bt.logging.debug(f"Action Selection Prompt: {prompt}")
            
            response = await self.llm.chat.completions.create(
                model=self.model_large,
                messages=[{"role": "user", "content": prompt}],
                tools=[*synapse.action_space, *self.local_action_space],
                tool_choice="auto",
            )

            bt.logging.debug(f"Action Selection Response: {response}")
            
            # Process selected action
            if response.choices[0].message.tool_calls:
                action = response.choices[0].message.tool_calls[0].function
                parsed_action = {
                    "name": action.name,
                    "arguments": json.loads(action.arguments),
                }
                
                # Store in state and memory
                state["action"] = parsed_action
                self.memory.push_log(
                    f"{parsed_action['name']}, "
                    + ", ".join(
                        [f"{k}: {v}" for k, v in parsed_action["arguments"].items()]
                    )
                )
                
                # Cache action if appropriate
                self.action_cache.put(cache_key_state, parsed_action)
            else:
                state["errors"].add("No action tool call in response")
                
        except Exception as e:
            bt.logging.error(f"Action Selection Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Action Selection Error: {e}")
            
        return state
    
    def _format_items(self, items):
        """Format items for context display"""
        if not items:
            return "None"
            
        formatted = []
        for item in items:
            if isinstance(item, dict):
                if "text" in item:
                    formatted.append(f"- {item['text']}")
                elif "id" in item and "description" in item:
                    formatted.append(f"- {item['id']}: {item['description']}")
                else:
                    formatted.append(f"- {str(item)}")
            else:
                formatted.append(f"- {str(item)}")
                
        return "\n".join(formatted)
    
    def _format_risk_assessment(self, risk_assessment):
        """Format risk assessment for prompt context"""
        if not risk_assessment:
            return "No risk assessment available."
            
        result = f"Overall risk level: {risk_assessment.get('overall_risk', 0):.2f}\n"
        
        if "risk_factors" in risk_assessment and risk_assessment["risk_factors"]:
            result += "Risk factors:\n"
            for factor in risk_assessment["risk_factors"]:
                result += f"- {factor.get('description', 'Unknown')} (Severity: {factor.get('severity', 0):.2f})\n"
        
        if "safe_directions" in risk_assessment and risk_assessment["safe_directions"]:
            result += f"Safe directions: {', '.join(risk_assessment['safe_directions'])}\n"
            
        if "hazardous_directions" in risk_assessment and risk_assessment["hazardous_directions"]:
            result += f"Hazardous directions: {', '.join(risk_assessment['hazardous_directions'])}\n"
            
        return result
    
    def _format_relevant_context(self, context):
        """Format relevant context for prompting"""
        if not context:
            return "No relevant context available."
            
        result = ""
        
        for key, value in context.items():
            if key.startswith("relevant_") and value and value != "None":
                header = key.replace("relevant_", "").capitalize()
                result += f"{header}:\n{value}\n\n"
                
        return result.strip()

    # ==== Execution Layer Components ====
    
    def action_validation(self, state: AgentState) -> AgentState:
        """Validate selected action for safety and feasibility"""
        bt.logging.debug(">> Action Validation")
        if state["errors"]:
            return state

        try:
            action = state["action"]
            if not action:
                state["errors"].add("No action to validate")
                return state
                
            # Check if action exists
            if "name" not in action:
                state["errors"].add("Invalid action format: missing name")
                return state
                
            # Validate move_in_direction actions
            if action["name"] == "move_in_direction":
                return self._validate_move_action(state)
                
            # Validate navigate_to actions
            elif action["name"] == "navigate_to":
                return self._validate_navigate_action(state)
                
            # Additional validations for other action types can be added here
            
        except Exception as e:
            bt.logging.error(f"Action Validation Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Action Validation Error: {e}")
            
        return state
    
    def _validate_move_action(self, state: AgentState) -> AgentState:
        """Validate movement actions for safety"""
        action = state["action"]
        risk_assessment = state["risk_assessment"]
        
        # Check if we have risk assessment data
        if not risk_assessment:
            return state
            
        # Extract parameters
        direction = action["arguments"].get("direction", "")
        distance = float(action["arguments"].get("distance", 0))
        
        # Validate direction
        if direction not in self.directions:
            state["errors"].add(f"Invalid direction: {direction}")
            return state
            
        # Check if direction is marked as hazardous
        if "hazardous_directions" in risk_assessment and direction in risk_assessment["hazardous_directions"]:
            # Adjust distance to be safer
            lidar_data = state["perception_data"].get("lidar", {})
            if direction in lidar_data:
                safe_distance = max(2.0, lidar_data[direction] - 3.0)
                action["arguments"]["distance"] = min(distance, safe_distance)
                bt.logging.warning(f"Adjusted move distance from {distance} to {action['arguments']['distance']} due to hazard")
        
        # Cap maximum movement distance for safety
        if distance > 30.0:
            action["arguments"]["distance"] = 30.0
            bt.logging.warning(f"Capped move distance to 30.0 (was {distance})")
            
        return state
    
    def _validate_navigate_action(self, state: AgentState) -> AgentState:
        """Validate navigation actions"""
        action = state["action"]
        
        # Check if target exists
        target = action["arguments"].get("target", "")
        if not target:
            state["errors"].add("Navigate action missing target")
            return state
            
        # Verify target exists in navigation system
        nav_locations = state["navigation_locations"]
        valid_targets = [loc.split(" : ")[0] for loc in nav_locations]
        
        if target not in valid_targets:
            bt.logging.warning(f"Navigation target {target} not in known locations: {valid_targets}")
            state["errors"].add(f"Unknown navigation target: {target}")
            
        return state
    
    def action_execution(self, state: AgentState) -> AgentState:
        """Execute the validated action"""
        bt.logging.debug(">> Action Execution")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]
            action = state["action"]
            
            if not action:
                bt.logging.error("No action to execute")
                direction, distance = self.maze_run(synapse)

                bt.logging.info(f"Fallback action: Direction: {direction}, Distance: {distance}")
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
                return state

            # Process custom actions that need transformation
            if action["name"] == "explore_wall_following":
                # Transform to move_in_direction
                direction, distance = self.maze_run(synapse)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
                
            elif action["name"] == "navigate_to":
                # Transform to move_in_direction using path planning
                target = action["arguments"].get("target")
                direction, distance = self.navigate_to(synapse, target)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
                
            elif action["name"] == "navigate_to_safety":
                # Handle emergency navigation to safety
                result = self.execute_navigate_to_safety(synapse, action["arguments"])
                if result:
                    state["action"] = result
                
            elif action["name"] == "explore_area":
                # Handle area exploration
                result = self.execute_explore_area(synapse, action["arguments"])
                if result:
                    state["action"] = result
                
            elif action["name"] == "mark_location":
                # Handle location marking
                result = self.execute_mark_location(synapse, action["arguments"])
                if result:
                    state["action"] = result
                
            elif action["name"] == "scan_environment":
                # Handle environment scanning
                result = self.execute_scan_environment(synapse, action["arguments"])
                if result:
                    state["action"] = result
                
            elif action["name"] == "create_path":
                # Handle path creation
                result = self.execute_create_path(synapse, action["arguments"])
                if result:
                    state["action"] = result
                
            elif action["name"] == "follow_path":
                # Handle path following
                result = self.execute_follow_path(synapse, action["arguments"])
                if result:
                    state["action"] = result
                
        except Exception as e:
            bt.logging.error(f"Action Execution Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Action Execution Error: {e}")
            
        return state
    
    def error_recovery(self, state: AgentState) -> AgentState:
        """Recover from action errors"""
        bt.logging.debug(">> Error Recovery")
        if not state["errors"]:
            # Reset recovery counter when successful
            self.recovery_attempts = 0
            return state

        try:
            synapse: Observation = state["observation"]
            
            # Log errors for diagnosis
            for error in state["errors"]:
                bt.logging.error(f"Recovering from error: {error}")
            
            # Increase recovery counter
            self.recovery_attempts += 1
            
            # If too many recovery attempts, use random movement
            if self.recovery_attempts >= self.max_recovery_attempts:
                bt.logging.warning(f"Multiple recovery attempts failed, using random walk")
                direction, distance = self.random_walk(synapse)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
                # Reset counter
                self.recovery_attempts = 0
            else:
                # Try maze run as recovery
                direction, distance = self.maze_run(synapse)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
            
            # Clear errors since we've handled them
            state["errors"] = set()
            
        except Exception as e:
            bt.logging.error(f"Error Recovery itself failed: {e}")
            traceback.print_exc()
            
            # Last resort fallback
            direction = random.choice(self.directions)
            state["action"] = {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": 5},
            }
            state["errors"] = set()
            
        return state

    # ==== Navigation Utilities ====
    
    def random_walk(self, synapse: Observation) -> tuple[str, float]:
        """Generate a random direction and distance, weighted by LiDAR readings"""
        weights = [1] * len(self.directions)

        # Update weights based on lidar data
        if synapse.sensor.lidar:
            readings = {}
            for data in synapse.sensor.lidar:
                readings[data[0]] = float(data[1].split("m")[0]) - 5.0

            for i, d in enumerate(self.directions):
                weights[i] = readings.get(d, 0) / 50.0

        # Avoid moving backwards
        if synapse.sensor.odometry[1] != "0m":
            i = self.directions.index(synapse.sensor.odometry[0])
            weights[(i + len(weights) // 2) % len(weights)] = 1e-6

        bt.logging.debug(f"Direction Weight: {[f'{f:.04f}' for f in weights]}")

        choice = random.choices(self.directions, weights=weights, k=1)[0]
        distance = random.randint(5, 30)

        return choice, distance
    
    def maze_run(self, synapse: Observation) -> tuple[str, float]:
        """Explore using wall-following algorithm"""
        r = [1] * len(self.directions)
        for data in synapse.sensor.lidar:
            r[self.directions.index(data[0])] = float(data[1].split("m")[0])

        l = len(self.directions)
        cdi = self.directions.index(self.maze_run_explore_direction)
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
                choice = self.directions[ldi]
                distance = 5
            else:
                # Walls on right: Go straight
                choice = self.directions[cdi]
                distance = r[cdi] // random.randint(2, 4)
        else:
            self.maze_run_counter += 1
            # No walls on right side: Turn right
            choice = self.directions[rdi]
            distance = r[rdi] // random.randint(2, 4)

        if self.maze_run_counter > 10:
            # Break cycle
            return self.random_walk(synapse)

        return choice, distance
    
    def navigate_to(self, synapse: Observation, target_node: str) -> tuple[str, float]:
        """Navigate to a specific target using SLAM path planning"""
        try:
            node = self.slam.grid_map.nav_nodes.get(target_node)
            if node is None:
                bt.logging.error(f"Navigation target node {target_node} not found")
                return self.random_walk(synapse)

            current_x, current_y, _ = self.slam.get_current_pose()
            
            # Get path from current position to target
            path = self.slam.grid_map.pose_navigation(
                current_x, current_y, node[0], node[1]
            )
            
            if path and len(path) > 1:
                next_node = path[1]
                direction = self._relative_direction(
                    current_x, current_y, next_node[0], next_node[1]
                )
                distance = self._relative_distance(
                    current_x, current_y, next_node[0], next_node[1]
                )
                return direction, distance
            else:
                bt.logging.error(f"No path found to target node {target_node}")
                return self.random_walk(synapse)
                
        except Exception as e:
            bt.logging.error(f"Navigation error: {e}")
            traceback.print_exc()
            return self.random_walk(synapse)
    
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

    # ==== Custom Action Implementation ====
    
    def execute_navigate_to_safety(self, synapse: Observation, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the navigate_to_safety action"""
        bt.logging.info(f"Executing navigate_to_safety with args: {args}")
        try:
            # Get priority level (default to high)
            priority = args.get("priority", "high")
            
            # Find the nearest safe location from memory
            if not self.memory.memory["landmarks"]:
                bt.logging.warning("No landmarks available for safety navigation")
                return self.fallback_move_action(synapse)
            
            current_x, current_y, _ = self.slam.get_current_pose()
            
            # Filter safe locations (those not marked as dangerous)
            safe_locations = [
                l for l in self.memory.memory["landmarks"] 
                if l.get("type") == "shelter" or 
                (l.get("type") != "danger" and "danger" not in l.get("description", "").lower())
            ]
            
            if not safe_locations:
                # Fall back to any non-danger location if no explicit safe locations
                safe_locations = [
                    l for l in self.memory.memory["landmarks"] 
                    if l.get("type") != "danger" and "danger" not in l.get("description", "").lower()
                ]
            
            if not safe_locations:
                bt.logging.warning("No safe locations found, using random walk")
                return self.fallback_move_action(synapse)
                
            # Sort by distance
            safe_locations = sorted(
                safe_locations,
                key=lambda l: math.sqrt((l["x"] - current_x)**2 + (l["y"] - current_y)**2)
            )
            
            # Select target based on priority
            target_location = safe_locations[0]
            if priority == "medium" and len(safe_locations) > 1:
                # Consider the second closest if medium priority to balance safety and exploration
                target_location = safe_locations[1]
            
            # Navigate to the selected safe spot
            direction, distance = self.navigate_to(synapse, target_location["id"])
            return {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
        except Exception as e:
            bt.logging.error(f"Safety navigation error: {e}")
            return self.fallback_move_action(synapse)

    def execute_explore_area(self, synapse: Observation, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the explore_area action"""
        bt.logging.info(f"Executing explore_area with args: {args}")
        try:
            # Get exploration parameters
            radius = float(args.get("radius", 50.0))
            strategy = args.get("strategy", "combined")
            
            if not hasattr(self, "exploration_state"):
                self.exploration_state = {
                    "iterations": 0,
                    "directions_tried": set(),
                    "last_direction": None,
                    "starting_point": None
                }
            
            # Get current position
            current_x, current_y, _ = self.slam.get_current_pose()
            
            if self.exploration_state["iterations"] == 0:
                # Initialize exploration from current position
                self.exploration_state["starting_point"] = (current_x, current_y)
            
            # Check if we've completed exploration (returned to starting point after trying multiple directions)
            if (self.exploration_state["iterations"] > 8 and 
                len(self.exploration_state["directions_tried"]) >= 6):
                # Reset exploration state
                self.exploration_state = {
                    "iterations": 0,
                    "directions_tried": set(),
                    "last_direction": None,
                    "starting_point": None
                }
                bt.logging.info("Exploration completed, area fully mapped")
                return {
                    "name": "move_in_direction",
                    "arguments": {"direction": random.choice(self.directions), "distance": 5},
                }
            
            self.exploration_state["iterations"] += 1
            
            # Strategy-based direction selection
            if strategy == "frontier":
                # Frontier-based: prefer directions with mid-range readings
                direction, distance = self.select_frontier_direction(synapse)
            elif strategy == "uncertainty":
                # Uncertainty-based: go to less-mapped areas
                direction, distance = self.select_uncertainty_direction(synapse, radius)
            else:
                # Combined: alternate between frontier and uncertainty
                if self.exploration_state["iterations"] % 2 == 0:
                    direction, distance = self.select_frontier_direction(synapse)
                else:
                    direction, distance = self.select_uncertainty_direction(synapse, radius)
            
            # Record the direction we're trying
            self.exploration_state["directions_tried"].add(direction)
            self.exploration_state["last_direction"] = direction
            
            return {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
        except Exception as e:
            bt.logging.error(f"Area exploration error: {e}")
            return self.fallback_move_action(synapse)

    def execute_mark_location(self, synapse: Observation, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the mark_location action (landmark creation)"""
        bt.logging.info(f"Executing mark_location with args: {args}")
        try:
            # Get parameters
            name = args.get("name")
            description = args.get("description", "")
            location_type = args.get("type", "general")
            
            if not name:
                bt.logging.error("Cannot mark location without a name")
                return {
                    "name": "talk_to",
                    "arguments": {
                        "target": "self",
                        "content": "I need to provide a name to mark this location."
                    }
                }
            
            # Check for duplicate names
            for landmark in self.memory.memory["landmarks"]:
                if landmark["id"] == name:
                    bt.logging.warning(f"Location name '{name}' already exists")
                    return {
                        "name": "talk_to",
                        "arguments": {
                            "target": "self",
                            "content": f"Location name '{name}' already exists. I should use a different name."
                        }
                    }
            
            # Get current position
            current_x, current_y, _ = self.slam.get_current_pose()
            
            # Add landmark to memory
            self.memory.add_landmark(name, description, current_x, current_y)
            
            # Add landmark to SLAM system
            self.slam.grid_map.update_nav_topo(
                self.slam.pose_index,
                current_x,
                current_y,
                node_id=name,
                node_desc=description,
                allow_isolated=True,
            )
            
            # Confirm the action
            return {
                "name": "talk_to",
                "arguments": {
                    "target": "self",
                    "content": f"I've marked this location as '{name}' ({location_type}): {description}"
                }
            }
        except Exception as e:
            bt.logging.error(f"Mark location error: {e}")
            return {
                "name": "talk_to",
                "arguments": {
                    "target": "self",
                    "content": "I couldn't mark this location due to an error."
                }
            }

    def execute_scan_environment(self, synapse: Observation, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the scan_environment action"""
        bt.logging.info(f"Executing scan_environment with args: {args}")
        try:
            # Get focus parameter
            focus = args.get("focus", "all")
            
            # Trigger an enhanced scan of the environment
            # This is implemented by setting a flag that will be detected by the perception
            # components, causing them to perform more detailed analyses
            self.entity_extraction_step = 0  # Force entity extraction on next cycle
            
            # Set a custom focus flag for the next perception cycle
            if not hasattr(self, "perception_focus"):
                self.perception_focus = {}
            
            self.perception_focus = {
                "type": focus,
                "intensity": 2.0,  # Enhanced scanning intensity
                "timestamp": time.time()
            }
            
            # Extract information from current observation
            environment_text = synapse.perception.environment
            objects_text = synapse.perception.objects
            
            # Extract entities on the spot (in addition to what will happen in the next cycle)
            system_prompt = (
                f"You are an entity extraction system. Identify {focus} entities in the text "
                "and return them in JSON format. Entity types: characters, objects, "
                "locations, hazards. Focus on detailed information and precise descriptions. "
                "Format each entity as: "
                "{'id': 'unique_name', 'type': 'entity_type', 'description': 'detailed_description', "
                "'properties': ['property1', 'property2'], 'risk_level': 0-1 if hazard}"
            )
            
            # Will be processed in the next agent cycle through the perception system
            
            # Perform a 360-degree scan by looking in all directions
            return {
                "name": "talk_to",
                "arguments": {
                    "target": "self",
                    "content": f"Scanning environment with focus on {focus}. Looking around carefully..."
                }
            }
        except Exception as e:
            bt.logging.error(f"Scan environment error: {e}")
            return {
                "name": "talk_to",
                "arguments": {
                    "target": "self",
                    "content": "I encountered an error while scanning the environment."
                }
            }

    def execute_create_path(self, synapse: Observation, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the create_path action"""
        bt.logging.info(f"Executing create_path with args: {args}")
        try:
            # Get parameters
            start = args.get("start", "current")
            end = args.get("end")
            path_name = args.get("path_name")
            
            if not end or not path_name:
                bt.logging.error("Missing required parameters for create_path")
                return {
                    "name": "talk_to",
                    "arguments": {
                        "target": "self",
                        "content": "I need both an end location and a path name to create a path."
                    }
                }
            
            # Initialize paths collection if not exists
            if "paths" not in self.memory.memory:
                self.memory.memory["paths"] = []
            
            # Check for duplicate path names
            for path in self.memory.memory["paths"]:
                if path["name"] == path_name:
                    bt.logging.warning(f"Path name '{path_name}' already exists")
                    return {
                        "name": "talk_to",
                        "arguments": {
                            "target": "self",
                            "content": f"Path name '{path_name}' already exists. I should use a different name."
                        }
                    }
            
            # Get current position if start is "current"
            current_x, current_y, _ = self.slam.get_current_pose()
            
            # Get start position
            start_x, start_y = current_x, current_y
            if start != "current":
                # Find start landmark
                start_found = False
                for landmark in self.memory.memory["landmarks"]:
                    if landmark["id"] == start:
                        start_x, start_y = landmark["x"], landmark["y"]
                        start_found = True
                        break
                
                if not start_found:
                    bt.logging.warning(f"Start landmark '{start}' not found")
                    return {
                        "name": "talk_to",
                        "arguments": {
                            "target": "self",
                            "content": f"I couldn't find the start landmark '{start}'."
                        }
                    }
            
            # Find end landmark
            end_found = False
            end_x, end_y = 0, 0
            for landmark in self.memory.memory["landmarks"]:
                if landmark["id"] == end:
                    end_x, end_y = landmark["x"], landmark["y"]
                    end_found = True
                    break
            
            if not end_found:
                bt.logging.warning(f"End landmark '{end}' not found")
                return {
                    "name": "talk_to",
                    "arguments": {
                        "target": "self",
                        "content": f"I couldn't find the end landmark '{end}'."
                    }
                }
            
            # Calculate path using SLAM
            path_coordinates = self.slam.grid_map.pose_navigation(
                start_x, start_y, end_x, end_y
            )
            
            if not path_coordinates:
                bt.logging.warning(f"No path found from {start} to {end}")
                return {
                    "name": "talk_to",
                    "arguments": {
                        "target": "self",
                        "content": f"I couldn't find a path from {start} to {end}."
                    }
                }
            
            # Store path in memory
            new_path = {
                "name": path_name,
                "start": start,
                "end": end,
                "coordinates": path_coordinates,
                "created_at": time.time()
            }
            self.memory.memory["paths"].append(new_path)
            self.memory.save()
            
            # Confirm path creation
            path_length = sum(
                math.sqrt((path_coordinates[i][0] - path_coordinates[i-1][0])**2 + 
                          (path_coordinates[i][1] - path_coordinates[i-1][1])**2)
                for i in range(1, len(path_coordinates))
            )
            
            return {
                "name": "talk_to",
                "arguments": {
                    "target": "self",
                    "content": f"I've created path '{path_name}' from {start} to {end} (estimated distance: {path_length:.1f}m with {len(path_coordinates)} waypoints)."
                }
            }
        except Exception as e:
            bt.logging.error(f"Create path error: {e}")
            return {
                "name": "talk_to",
                "arguments": {
                    "target": "self",
                    "content": "I encountered an error while creating the path."
                }
            }

    def execute_follow_path(self, synapse: Observation, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the follow_path action"""
        bt.logging.info(f"Executing follow_path with args: {args}")
        try:
            # Get parameters
            path_name = args.get("path_name")
            
            if not path_name:
                bt.logging.error("Missing path name parameter")
                return {
                    "name": "talk_to",
                    "arguments": {
                        "target": "self",
                        "content": "I need a path name to follow a path."
                    }
                }
            
            # Check if paths exist in memory
            if "paths" not in self.memory.memory or not self.memory.memory["paths"]:
                bt.logging.warning("No paths available in memory")
                return {
                    "name": "talk_to",
                    "arguments": {
                        "target": "self",
                        "content": "I don't have any paths in my memory."
                    }
                }
            
            # Find the requested path
            selected_path = None
            for path in self.memory.memory["paths"]:
                if path["name"] == path_name:
                    selected_path = path
                    break
            
            if not selected_path:
                bt.logging.warning(f"Path '{path_name}' not found")
                return {
                    "name": "talk_to",
                    "arguments": {
                        "target": "self",
                        "content": f"I couldn't find a path named '{path_name}'."
                    }
                }
            
            # Initialize path following state if not exists
            if not hasattr(self, "path_following_state"):
                self.path_following_state = {}
            
            # Setup path following state
            self.path_following_state = {
                "path_name": path_name,
                "waypoints": selected_path["coordinates"],
                "current_waypoint": 0,
                "started_at": time.time()
            }
            
            # Get current position
            current_x, current_y, _ = self.slam.get_current_pose()
            
            # Find the closest waypoint to start from
            min_dist = float('inf')
            closest_idx = 0
            
            for i, (wx, wy) in enumerate(selected_path["coordinates"]):
                dist = math.sqrt((wx - current_x)**2 + (wy - current_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Set closest waypoint as current (skip if we're already at the start)
            if closest_idx > 0 and min_dist > 5.0:
                self.path_following_state["current_waypoint"] = closest_idx
            
            # Navigate to the next waypoint
            if closest_idx + 1 < len(selected_path["coordinates"]):
                next_wp = closest_idx + 1
            else:
                next_wp = closest_idx
            
            next_x, next_y = selected_path["coordinates"][next_wp]
            
            # Calculate direction and distance to the next waypoint
            direction = self._relative_direction(current_x, current_y, next_x, next_y)
            distance = self._relative_distance(current_x, current_y, next_x, next_y)
            
            # Limit distance to a reasonable value
            distance = min(distance, 30.0)
            
            return {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
        except Exception as e:
            bt.logging.error(f"Follow path error: {e}")
            return self.fallback_move_action(synapse)
            
    # ==== Helper functions for the custom actions ====
    
    def select_frontier_direction(self, synapse: Observation) -> tuple[str, float]:
        """Select direction based on frontier exploration strategy"""
        readings = {}
        for data in synapse.sensor.lidar:
            readings[data[0]] = float(data[1].split("m")[0])
        
        # Find directions with mid-range readings (frontiers)
        frontier_directions = []
        for direction, distance in readings.items():
            # Ideal frontier distance is between 15m and 30m
            if 15.0 <= distance <= 30.0:
                frontier_directions.append((direction, distance))
        
        if frontier_directions:
            # Choose a random frontier direction
            chosen = random.choice(frontier_directions)
            return chosen[0], chosen[1] * 0.7  # Move 70% of the way to the frontier
        else:
            # If no good frontiers, find the most open direction
            max_dist = 0
            best_dir = None
            for direction, distance in readings.items():
                if distance > max_dist:
                    max_dist = distance
                    best_dir = direction
            
            if best_dir:
                return best_dir, min(max_dist * 0.6, 30.0)  # Move 60% of the way, max 30m
            else:
                return random.choice(self.directions), 10.0
    
    def select_uncertainty_direction(self, synapse: Observation, radius: float) -> tuple[str, float]:
        """Select direction based on uncertainty exploration strategy"""
        # This would ideally use a proper exploration map with uncertainty values
        # As a simplified approach, we'll avoid recently visited directions
        
        current_x, current_y, _ = self.slam.get_current_pose()
        
        # Get all landmarks within the specified radius
        nearby_landmarks = []
        for landmark in self.memory.memory["landmarks"]:
            dist = math.sqrt((landmark["x"] - current_x)**2 + (landmark["y"] - current_y)**2)
            if dist <= radius:
                nearby_landmarks.append((landmark, dist))
        
        # If there are nearby landmarks, go in a direction with fewer landmarks
        if nearby_landmarks:
            # Create a score for each direction based on inverse landmark density
            direction_scores = {direction: 1.0 for direction in self.directions}
            
            for landmark, dist in nearby_landmarks:
                # Calculate angle to landmark
                dx = landmark["x"] - current_x
                dy = landmark["y"] - current_y
                angle = (180 + (180 / 3.14) * math.atan2(dy, dx)) % 360
                angle = (angle + 22.5) % 360
                index = int(angle / 45)
                landmark_direction = self.directions[index]
                
                # Reduce score for this direction (inversely proportional to distance)
                impact = 1.0 - (dist / radius)
                direction_scores[landmark_direction] *= (1.0 - impact * 0.8)
            
            # Find direction with highest score (least explored)
            best_direction = max(direction_scores.items(), key=lambda x: x[1])[0]
            
            # Get LiDAR reading for this direction
            distance = 10.0  # Default
            for data in synapse.sensor.lidar:
                if data[0] == best_direction:
                    distance = float(data[1].split("m")[0])
                    break
            
            # Move a reasonable distance in the chosen direction
            return best_direction, min(distance * 0.7, 30.0)
        
        # If no landmarks within radius, use random walk with a bias towards open spaces
        return self.random_walk(synapse)
    
    def fallback_move_action(self, synapse: Observation) -> Dict[str, Any]:
        """Generate a fallback movement action"""
        direction, distance = self.random_walk(synapse)
        return {
            "name": "move_in_direction",
            "arguments": {"direction": direction, "distance": distance},
        }
