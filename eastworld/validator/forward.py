# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

import asyncio
import time
import traceback
from urllib.parse import urlparse

import bittensor as bt
import httpx
import numpy as np
from openai import AsyncOpenAI

from eastworld.base.validator import BaseValidatorNeuron
from eastworld.protocol import Observation
from eastworld.validator.reward import get_rewards
from eastworld.utils.uids import check_uid_availability


async def forward(self: BaseValidatorNeuron):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    # miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    try:

        # Call Eastworld observation API
        data = await observate(self)

        ob_status = data["code"]
        ob_message = data["message"]
        if ob_status == 429:
            bt.logging.info("The next turn is not available yet. Wait for 10s.")
            await asyncio.sleep(10)
            return
        elif ob_status != 200:
            bt.logging.error(
                f"Failed to get observation from Eastworld. {ob_status} {ob_message}"
            )
            await asyncio.sleep(10)
            return

        ob_turns = data["turns"]
        ob_uid = data["uid"]
        ob_key = data["key"]

        uid_is_available = check_uid_availability(
            self.metagraph, ob_uid, self.config.neuron.vpermit_tao_limit
        )
        bt.logging.info(
            f"UID {ob_uid} {self.metagraph.axons[ob_uid].hotkey} {uid_is_available}"
        )
        if not uid_is_available:
            bt.logging.info(f"UID {ob_uid} is not available for mining.")
            await asyncio.sleep(10)
            return
        if ob_key != self.metagraph.axons[ob_uid].hotkey:
            bt.logging.info(
                f"UID {ob_uid} hotkey mismatch API:{ob_key} Metagraph:{self.metagraph.axons[ob_uid].hotkey}"
            )
            await asyncio.sleep(10)
            return

        axon = self.metagraph.axons[ob_uid]
        bt.logging.info(f"Selected miners: {ob_uid} {axon.ip}:{axon.port}")
        miner_uids = np.array([ob_uid])
        synapse = await create_synapse(self, data)

        # The dendrite client queries the network.
        timeout = 60
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse,
            deserialize=False,
            timeout=timeout,
        )

        # Log the results for monitoring purposes.
        # bt.logging.info(f"Received responses: {responses}")

        # TODO: Validate response synapse hotkey
        await action(self, ob_turns, ob_uid, responses[0])

        # TODO(developer): Define how the validator scores responses.
        # Adjust the scores based on responses from miners.
        rewards = get_rewards(self, query=self.step, responses=responses)

        # bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        # self.update_scores(rewards, miner_uids)
        await asyncio.sleep(10)
    except Exception as e:
        traceback.print_exc()
        raise e


async def observate(self) -> dict:
    """ """
    endpoint_config = urlparse(self.config.eastworld.endpoint)
    endpoint = f"{endpoint_config.scheme}://{endpoint_config.netloc}/sn/env"
    req = self.http_client.build_request("GET", endpoint)
    r = await self.http_client.send(req)
    if r.status_code != 200:
        raise Exception(
            f"Failed to get observation from Eastworld. {r.status_code} {r.text}"
        )

    ob = r.json()
    return ob


async def action(self, turns: int, uid: int, synapse: Observation):
    """ """
    endpoint_config = urlparse(self.config.eastworld.endpoint)
    endpoint = f"{endpoint_config.scheme}://{endpoint_config.netloc}/sn/step"

    for act in synapse.action:
        if not isinstance(act, dict):
            bt.logging.debug("Synapse Observation.action item is not a dict")
            # TODO: Raise bad action exception?
            return

    data = {
        "turns": turns,
        "uid": uid,
        "key": synapse.axon.hotkey,
        "action": synapse.action,
    }
    req = self.http_client.build_request("POST", endpoint, json=data)

    r = await self.http_client.send(req)
    if r.status_code != 200:
        raise Exception(
            f"Failed to submit action to eastworld. {r.status_code} {r.text}"
        )

    ob = r.json()


async def create_synapse(self, ob: dict) -> Observation:
    ob_context = ob["context"]

    bt.logging.info(ob_context)

    agent_stats = ob_context["stats"]
    agent_item = ob_context["item"]

    env = ob_context["env"]
    scanner = ob_context["scanner"]

    scanner = {"lidar": ob_context["scanner"]["lidar"], "perception": []}
    scanner["perception"].extend(env["location"])
    scanner["perception"].extend(env["entity"])

    interaction = ob_context["interaction"]
    feedback = ob_context["log"] or []

    action_space = ob_context["action"]

    env_position = env["position"]
    env_terrain = env["terrain"]
    env_surroundings = env["location"]
    env_entity = env["entity"]

    part_position = ", ".join(env_position)

    part_terrain = ""
    for detail in env_terrain:
        part_terrain += f'- {", ".join(detail)} \n'

    part_surroundings = ""
    for detail in env_surroundings:
        part_surroundings += f'- {", ".join(detail)} \n'

    part_entities = ""
    for detail in env_entity:
        part_entities += f'- {", ".join(detail)} \n'

    with open("eastworld/validator/prompts/env_report.txt", "r") as f:
        report_template = f.read()

    msg_user = report_template.format(
        part_position=part_position,
        part_terrain=part_terrain,
        part_surroundings=part_surroundings,
        part_entities=part_entities,
    )
    bt.logging.info(msg_user)

    messages = [{"role": "user", "content": msg_user}]

    async with httpx.AsyncClient() as client:
        client = AsyncOpenAI(http_client=client)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        env = response.choices[0].message.content

    bt.logging.info(env)

    synapse = Observation(
        stats={},
        item=agent_item,
        surrounding=[env],
        interaction=[],
        scanner=scanner,
        feedback=feedback,
        action_space=action_space,
        action=[],
        reward=0.0,
    )

    return synapse
