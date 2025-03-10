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
from eastworld.utils.uids import check_uid_availability


class Validator(BaseValidatorNeuron):
    """
    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    http_client: httpx.AsyncClient
    http_auth: httpx.BasicAuth
    inactive_miners: dict

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.http_client = httpx.AsyncClient()
        self.http_auth = httpx.BasicAuth(
            username=self.config.eastworld.endpoint_auth_user,
            password=self.config.eastworld.endpoint_auth_password,
        )

        self.inactive_miners = {}

    async def forward(self):
        """
        The forward function is called by the validator every time step.

        It is responsible for querying the network and scoring the responses.

        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

        """
        try:
            # Call Eastworld API to get miner's state and observation.
            data = await self.get_observation()

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

            # Miner data
            ob_turns: int = data["turns"]
            ob_uid: int = data["uid"]
            ob_key: str = data["key"]

            uid_is_available = check_uid_availability(
                self.metagraph, ob_uid, self.config.neuron.vpermit_tao_limit
            )
            bt.logging.debug(
                f"UID {ob_uid} {self.metagraph.axons[ob_uid].hotkey} {uid_is_available}"
            )
            if not uid_is_available:
                bt.logging.info(f"UID {ob_uid} from API is not available for mining.")
                await asyncio.sleep(10)
                return
            if ob_key != self.metagraph.axons[ob_uid].hotkey:
                bt.logging.info(
                    f"UID {ob_uid} hotkey mismatch API:{ob_key} Metagraph:{self.metagraph.axons[ob_uid].hotkey}"
                )
                await asyncio.sleep(10)
                return

            # Skip the miner for a certain period if it is inactive.
            if self.config.subtensor.network == "test":
                notuntil, interval = self.inactive_miners.get(ob_uid, (0, 0))
                if notuntil and notuntil > time.time():
                    bt.logging.info(f"Skip for inactive miner #{ob_uid}.")
                    return

            axon = self.metagraph.axons[ob_uid]
            bt.logging.info(f"Selected miners: {ob_uid} {axon.ip}:{axon.port}")
            miner_uids = np.array([ob_uid])
            synapse = await self.create_synapse(data)

            # The dendrite client queries the network.
            timeout = self.config.neuron.timeout
            responses = await self.dendrite(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=synapse,
                deserialize=False,
                timeout=timeout,
            )

            # Log the results for monitoring purposes.
            bt.logging.trace(f"Received responses: {responses}")

            synapse: bt.Synapse = responses[0]
            # Add skip time for inactive miners.
            if self.config.subtensor.network == "test":
                if synapse.is_failure or not len(synapse.action):
                    notuntil, interval = self.inactive_miners.get(
                        ob_uid, (time.time(), 60)
                    )
                    # Increase the skip time by 3 minutes each time, up to 30 minutes.
                    interval = min(interval + 180, 1800)
                    self.inactive_miners[ob_uid] = (notuntil + interval, interval)
                    bt.logging.info(
                        f"Inactive miner #{ob_uid}. Skip for {interval} seconds."
                    )
                else:
                    self.inactive_miners.pop(ob_uid, None)

            if synapse.is_failure or not len(synapse.action):
                bt.logging.warning(
                    f"Failed to get action from miner #{ob_uid}. Code: {synapse.dendrite.status_code}"
                )
                return

            await self.submit_action(ob_turns, ob_uid, synapse)
            await self.update_scores()
        except Exception as e:
            traceback.print_exc()
            await asyncio.sleep(10)
            raise e

    async def get_observation(self) -> dict:
        """ """
        endpoint_url = urlparse(self.config.eastworld.endpoint_url)
        endpoint = f"{endpoint_url.scheme}://{endpoint_url.netloc}/sn/env"
        req = self.http_client.build_request("GET", endpoint)
        r = await self.http_client.send(req, auth=self.http_auth)
        if r.status_code != 200:
            raise Exception(
                f"Failed to get observation from Eastworld. {r.status_code} {r.text}"
            )

        ob = r.json()
        return ob

    async def submit_action(self, turns: int, uid: int, synapse: Observation):
        """ """
        endpoint_url = urlparse(self.config.eastworld.endpoint_url)
        endpoint = f"{endpoint_url.scheme}://{endpoint_url.netloc}/sn/step"

        for act in synapse.action:
            if not isinstance(act, dict):
                bt.logging.warning("Synapse Observation.action item is not a dict")
                return

        data = {
            "turns": turns,
            "uid": uid,
            "key": synapse.axon.hotkey,
            "action": synapse.action,
        }
        req = self.http_client.build_request("POST", endpoint, json=data)

        r = await self.http_client.send(req, auth=self.http_auth)
        if r.status_code != 200:
            raise Exception(
                f"Failed to submit action to Eastworld server. {r.status_code} {r.text}"
            )
        ob = r.json()
        if ob.get("code") != 200:
            raise Exception(
                f"Failed to submit action to Eastworld server. {ob.get('code')} {ob.get('message')}"
            )
        bt.logging.trace(
            f"Action of miner UID {uid} in turn {turns} submitted successfully."
        )

    async def create_synapse(self, data: dict) -> Observation:
        context = data["context"]

        # Agent integrity, energy level, etc. Will be implemented in the future.
        agent_stats = context["stats"]
        # Items in Agent's inventory.
        agent_item = context["item"]
        # Environment observation.
        agent_ob = context["observation"]
        # Environment interaction to Agent. Conversation started by others, environmental damage, etc.
        agent_interaction = context["interaction"]

        # Available function to call to perform actions.
        action_space = context["action"]
        # Action execution log of last round.
        action_log = context["log"]

        # Agent's current location description.
        ob_location = agent_ob["location"]
        # Notable terrains features around.
        ob_terrain = agent_ob["terrain"]
        # Structures around.
        ob_structure = agent_ob["structure"]
        # Intelligent entities around.
        ob_character = agent_ob["character"]
        # Interactive objects around.
        ob_object = agent_ob["object"]
        # Environment description. Wind, temperature, weather, etc. Will be implemented in the future.
        ob_env = agent_ob["environment"]
        # LiDAR scanner data.
        ob_lidar = agent_ob["lidar"]

        # Synapse data
        # scanner: More accurate data obtained from the instrument and sensors.
        scanner = {"lidar": ob_lidar}
        # perception: General output of detecting, classifying, and ranging objects in the environment.
        perception = ""

        # Summarize perception with LLM
        prompt_context = {
            "location": "",
            "terrain": "",
            "structure": "",
            "character": "",
            "object": "",
        }
        for loc in ob_location:
            prompt_context["location"] += f"- {', '.join(loc)}\n"
        for t in ob_terrain:
            prompt_context["terrain"] += f"- {', '.join(t)}\n"
        for s in ob_structure:
            prompt_context["structure"] += f"- {', '.join(s)}\n"
        for c in ob_character:
            prompt_context["character"] += f"- {', '.join(c)}\n"
        for o in ob_object:
            prompt_context["object"] += f"- {', '.join(o)}\n"

        messages = [{"role": "user", "content": ""}]
        with open("eastworld/validator/prompts/perception.txt", "r") as f:
            prompt_tpl = f.read()
            messages[0]["content"] = prompt_tpl.format(**prompt_context)

        # bt.logging.debug(
        #     f"Prompt in perception summarization: \n{messages[0]['content']}"
        # )
        async with httpx.AsyncClient() as client:
            llm = AsyncOpenAI(http_client=client)

            response = await llm.chat.completions.create(
                model=self.config.eastworld.llm_model,
                messages=messages,
            )
            perception = response.choices[0].message.content

        bt.logging.debug(f"LLM response in perception summarization: \n{perception}")
        synapse = Observation(
            stats=agent_stats,
            items=agent_item,
            scanner=scanner,
            perception=perception,
            action_log=action_log,
            action_space=action_space,
            action=[],
            reward=0.0,
        )

        return synapse

    async def update_scores(self):
        """Fetch latest miners' scores from Eastworld server"""
        # Update scores every 30 steps.
        if self.step % 30 != 0:
            return

        endpoint_url = urlparse(self.config.eastworld.endpoint_url)
        endpoint = f"{endpoint_url.scheme}://{endpoint_url.netloc}/sn/score"

        req = self.http_client.build_request("GET", endpoint)
        r = await self.http_client.send(req, auth=self.http_auth)
        if r.status_code != 200:
            raise Exception(
                f"Failed to get miner scores from Eastworld. {r.status_code} {r.text}"
            )

        data = r.json()
        uids = [d["uid"] for d in data["scores"]]
        new_scores = [d["score"] for d in data["scores"]]

        # Check if rewards contains NaN values.
        if np.isnan(new_scores).any():
            bt.logging.warning(f"NaN values detected in rewards: {new_scores}")
            # Replace any NaN values in scores with 0.
            new_scores = np.nan_to_num(new_scores, nan=0)

        # Ensure new_scores is a numpy array.
        new_scores = np.asarray(new_scores)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either new_scores or uids_array is empty.
        if new_scores.size == 0 or uids_array.size == 0:
            bt.logging.info(f"new_scores: {new_scores}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either new_scores or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of new_scores and uids_array match.
        if new_scores.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: new_scores array of shape {new_scores.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # If things work as expected, scores from Eastworld API should be same size as metagraph.
        # shape: [ metagraph.n ]
        scattered_scores: np.ndarray = np.zeros_like(self.scores)
        scattered_scores[uids_array] = new_scores
        bt.logging.debug(f"Scattered scores: {scattered_scores}")

        # Update local scores.
        # shape: [ metagraph.n ]
        self.scores: np.ndarray = scattered_scores
        bt.logging.debug(f"Updated scores: {self.scores}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator is running... {time.time()}")
            time.sleep(30)
