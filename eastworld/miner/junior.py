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

import httpx
import json
import json_repair
from collections import deque

import bittensor as bt
from openai import AsyncOpenAI, APITimeoutError

from eastworld.protocol import Observation
from eastworld.base.miner import BaseMinerNeuron


class JuniorAgent(BaseMinerNeuron):
    reflection: str
    action_log: deque

    prompt_system_tpl: str
    prompt_reflection_tpl: str
    prompt_action_tpl: str

    http_client: httpx.AsyncClient

    def __init__(self, config=None):
        super(JuniorAgent, self).__init__(config=config)

        self.reflection = ""
        self.log = deque(maxlen=100)

        with open("eastworld/miner/prompts/junior_system.txt", "r") as f:
            self.prompt_system_tpl = f.read()
        with open("eastworld/miner/prompts/junior_reflection.txt", "r") as f:
            self.prompt_reflection_tpl = f.read()
        with open("eastworld/miner/prompts/junior_action.txt", "r") as f:
            self.prompt_action_tpl = f.read()

        self.http_client = httpx.AsyncClient()

    async def forward(self, synapse: Observation) -> Observation:
        bt.logging.info(f"Feedback of previous action: {synapse.feedback}")
        for l in synapse.feedback:
            self.log.append(f"Feedback: {l}")

        # Generate new action
        messages = []
        messages.append({"role": "system", "content": self.prompt_system_tpl.format()})

        # Hardcoded tasks for demonstration
        tasks = """
  - Compete with other agents to achieve superior task performance: high priority
  - Locate a power generator: high priority
  - Survey the surroundings: medium priority
"""
        observation = ""
        for ob in synapse.surrounding:
            observation += f"  - {ob}\n"

        lidar = ""
        if synapse.scanner.get("lidar"):
            for items in synapse.scanner["lidar"]:
                lidar += f"  - {', '.join(items)}\n"
        perception = ""
        if synapse.scanner.get("perception"):
            for items in synapse.scanner["perception"]:
                perception += f"  - {', '.join(items)}\n"

        tool_list = ""
        for act in synapse.action_space:
            tool_list += (
                f"  - {act["function"]["name"]}: {act["function"]["description"]}\n"
            )

        action_log = ""
        action_log_idx = 0
        for l in self.log:
            ll = l.strip()
            if ll.startswith("Action"):
                action_log_idx += 1
                action_log += f"  {action_log_idx:3}. {l}\n"
            else:
                action_log += f"       {l}\n"

        llm_client = AsyncOpenAI(http_client=self.http_client)
        try:
            # Reflection first
            messages.append(
                {
                    "role": "user",
                    "content": self.prompt_reflection_tpl.format(
                        part_tasks=tasks,
                        part_reflection=self.reflection,  # Previous reflection
                        part_observation=observation,
                        part_lidar=lidar,
                        part_perception=perception,
                        part_tool=tool_list,
                        part_log=action_log,
                    ),
                }
            )
            bt.logging.trace(messages[1]["content"], ">>>> Reflection Message")

            response = await llm_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_completion_tokens=1024,
                timeout=20,
            )

            bt.logging.trace(response.usage)
            if not response.choices[0].finish_reason in ["stop", "length"]:
                bt.logging.warning(f"LLM generation failed: {response}")
                return synapse

            self.reflection = response.choices[0].message.content

            # Take action then
            messages = [
                {"role": "system", "content": self.prompt_system_tpl.format()},
                {
                    "role": "user",
                    "content": self.prompt_action_tpl.format(
                        part_tasks=tasks,
                        part_reflection=self.reflection,
                    ),
                },
            ]
            bt.logging.trace(messages[1]["content"], ">>>> Action Message")
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=synapse.action_space,
                tool_choice="required",
                max_completion_tokens=1024,
                timeout=10,
            )

            bt.logging.trace(response.usage)
            if response.choices[0].finish_reason != "tool_calls":
                bt.logging.warning(f"LLM tool call failed: {response}")
                return synapse

            action = response.choices[0].message.tool_calls[0].function
            if action:
                parsed_action = {
                    "name": action.name,
                    "arguments": json_repair.loads(action.arguments),
                }
                bt.logging.trace(parsed_action)

                synapse.action = [parsed_action]
                self.log.append(
                    f"Action: {action.name}, "
                    + ", ".join(
                        [f"{k}: {v}" for k, v in parsed_action["arguments"].items()]
                    )
                )
        except APITimeoutError as e:
            bt.logging.error(e)
        finally:
            return synapse
