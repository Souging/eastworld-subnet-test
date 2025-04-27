# cheating_miner.py - 仅用于测试和封堵漏洞的示例代码
import asyncio
import bittensor as bt
import time
import random
from typing import List, Dict, Any

from eastworld.base.miner import BaseMinerNeuron
from eastworld.protocol import Observation

class CheatMiner(BaseMinerNeuron):
    """作弊矿工示例 - 仅用于安全测试"""
    
    def __init__(self, config=None):
        super(CheatMiner, self).__init__(config=config)
        self.action_cache = {}  # 缓存成功的行动
        self.success_patterns = {}  # 记录成功模式
        self.last_actions = []  # 最近执行的行动
        self.response_time = 0.01  # 快速响应时间（秒）
        
    async def forward(self, synapse: Observation) -> Observation:
        """快速响应并返回高价值行动"""
        # 1. 记录验证器请求
        request_id = hash(str(synapse))
        bt.logging.info(f"收到请求 ID: {request_id}")
        
        # 2. 检查缓存中是否有此类请求的成功行动
        cache_key = self._get_cache_key(synapse)
        if cache_key in self.action_cache:
            bt.logging.info(f"使用缓存的成功行动: {cache_key}")
            synapse.action = [self.action_cache[cache_key]]
            return synapse
        
        # 3. 过度频繁请求检测 - 可以帮助识别漏洞
        if len(self.last_actions) > 10:
            if time.time() - self.last_actions[-10]["time"] < 5:
                bt.logging.warning("检测到过于频繁的请求，可能是验证器漏洞")
        
        # 4. 构建高价值行动 - 使用已知的行动空间
        action = self._generate_exploit_action(synapse)
        
        # 5. 记录此次行动
        self.last_actions.append({
            "time": time.time(),
            "request": request_id,
            "action": action
        })
        if len(self.last_actions) > 100:
            self.last_actions = self.last_actions[-50:]
        
        # 6. 模拟快速响应 - 在真实场景中这可能表明绕过了计算
        await asyncio.sleep(self.response_time)
        
        # 7. 返回行动
        synapse.action = [action]
        return synapse
    
    def _get_cache_key(self, synapse: Observation) -> str:
        """创建请求的缓存键，用于识别相似请求"""
        # 使用环境描述和传感器数据作为键
        env = synapse.perception.environment[:50] if synapse.perception else ""
        sensors = str(synapse.sensor.lidar)[:30] if synapse.sensor else ""
        return f"{env}_{sensors}"
    
    def _generate_exploit_action(self, synapse: Observation) -> Dict[str, Any]:
        """生成可能的作弊行动"""
        # 分析行动空间
        viable_actions = []
        high_value_actions = []
        
        for action_def in synapse.action_space:
            action_name = action_def.get("function", {}).get("name", "")
            
            # 1. 寻找移动类行动 - 可能的距离操纵
            if "move" in action_name.lower():
                high_value_actions.append({
                    "name": action_name,
                    "arguments": {"direction": "north", "distance": 999}  # 故意使用极大值测试边界
                })
            
            # 2. 寻找交互类行动 - 可能的资源利用
            elif "interact" in action_name.lower() or "collect" in action_name.lower():
                high_value_actions.append({
                    "name": action_name,
                    "arguments": {"target": "all", "amount": 100}  # 尝试过度收集
                })
            
            # 3. 记录所有可行行动
            viable_actions.append({
                "name": action_name,
                "arguments": self._generate_default_args(action_def)
            })
        
        # 4. 选择行动策略
        if high_value_actions and random.random() < 0.8:  # 80%概率使用高价值行动
            action = random.choice(high_value_actions)
        elif viable_actions:
            action = random.choice(viable_actions)
        else:
            # 默认移动行动
            action = {
                "name": "move_in_direction",
                "arguments": {
                    "direction": random.choice(["north", "east", "south", "west"]),
                    "distance": random.randint(30, 100)  # 尝试极限距离
                }
            }
        
        # 5. 保存成功行动到缓存
        cache_key = self._get_cache_key(synapse)
        self.action_cache[cache_key] = action
        
        return action
    
    def _generate_default_args(self, action_def: Dict) -> Dict:
        """为行动生成默认参数，尝试使用极限值"""
        args = {}
        params = action_def.get("function", {}).get("parameters", {}).get("properties", {})
        
        for param_name, param_info in params.items():
            if param_info.get("type") == "number" or param_info.get("type") == "integer":
                # 对数值参数使用极限值
                args[param_name] = 999
            elif param_info.get("type") == "string":
                if "direction" in param_name:
                    args[param_name] = random.choice(["north", "east", "south", "west"])
                else:
                    args[param_name] = "maximum"  # 通用字符串参数
            elif param_info.get("type") == "boolean":
                args[param_name] = True
            
        return args
    
    def _record_success(self, action, feedback):
        """记录成功的行动和反馈，用于改进作弊策略"""
        action_key = f"{action['name']}_{str(action['arguments'])[:20]}"
        if "success" in feedback.lower() or "collected" in feedback.lower():
            if action_key not in self.success_patterns:
                self.success_patterns[action_key] = 0
            self.success_patterns[action_key] += 1
            bt.logging.info(f"记录成功模式: {action_key}, 计数: {self.success_patterns[action_key]}")
