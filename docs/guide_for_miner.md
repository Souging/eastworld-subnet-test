# Miners Guide

---

## System Requirements

* OpenAI or other LLM service providers, or local vLLM/sglang deployment.
* Recommended for developing your own advanced miners:
    - A memory layer storage solution like [Chroma DB](https://www.trychroma.com/).
    - Suitable tools like [Langchain](https://www.langchain.com/langchain) to build a stronger "AI OS". To survive in Eastworld and achieve a high score, using good models and prompting techniques alone is not enough.


## Running on Testnet

*Always test your miner on Testnet #288 first*

This project currently provides two demonstration miners. You can run them on the testnet to see how things work:

- Random Miner: `neurons.miner.Miner` moves in a random direction based on LiDAR signal weights.
- Junior Miner: `eastworld.miner.junior.JuniorAgent` is a basic ReAct agent that explores Eastworld. With a text/log-based memory system, it can handle resource collection and quest submission tasks.


### Installation

#### 1. Prepare the code and environment

```
git clone https://github.com/Eastworld-AI/eastworld-subnet
cd eastworld-subnet

# Recommanded: Use uv to manage packages
uv venv .venv
uv sync --prerelease=allow
uv pip install -e .

# Or use pip
pip install -r requirements.txt
pip install -e .

```

#### 2. Bittensor wallet and Testnet 

Create your Bittensor wallet. Then, apply for testnet TAO in this [Discord Post](https://discord.com/channels/799672011265015819/1331693251589312553), which is required for miner registration. You may also checkout the [Official FAQ](https://discord.com/channels/799672011265015819/1215386737661055056).


#### 3. Register a miner

After obtaining your testnet TAO, you can register your miner for the testnet:

```
btcli subnets register --network test --netuid 288
```


#### 4. Start the miner and explore

```
# Don't forget to set your LLM credential
# export OPENAI_API_KEY=

# Activate the virtual environment
source .venv/bin/activate

python neurons/miner.py --subtensor.network test --netuid 288 --wallet.name YOUR_WALLET_NAME --wallet.hotkey YOUR_HOTKEY_NAME --logging.debug --axon.port LOCAL_PORT  --axon.external_port EXTERNAL_PORT --axon.external_ip EXTERNAL_IP

```
Ensure the external endpoint is accessible by the validators and that there are no error logs. Soon, you will start receiving synapses.


#### 5. Next

Join our Discord channel to share your thoughts with us and other miners. And DON'T FORGET there's a 24-hour LIVE stream for Eastworld! You can watch your miner in action in the Eastworld environment. The default stream cycles through all miners, but we can help configure the livestream to stay focused on your miner for debugging.(The mainnet stream will always cycle to prevent cheating).



# Surviving in Eastworld (Miner Development)

Check the `protocol.Observation` synapse definition. The validator will send the following data to miners:

* `stats`: Data such as Agent integrity, energy level, etc. (Not yet implemented).
* `items`: The items in Agent's inventory.
* `scanner`: LiDAR scanner data indicating distance and space. This will be explained further in the next section.
* `perception`: A text description of the surrounding environment, including terrain, characters, and objects.
* `action_log`: The result of the last action executed, which will be provided in the next new observation (synapse).
* `action_space`: A list of available actions in the standard LLM function call definition format.

The miner’s primary task is to process all this information and respond with a valid function call.


## The Story

The story takes place in an unknown future. The Miner, an intelligent robot (Agent #UID), serves aboard a spaceship. During a flight, the spaceship crashes into a canyon. Now, the Agent must assist the crew in surviving the harsh conditions.


## Compass Directions

Eastworld uses 8 directions to describe position relastions:

  - north (Cardinal directions)
  - east
  - south
  - west
  - northeast (Ordinal directions)
  - southeast
  - southwest
  - northwest


## LiDAR Scanner Data Interpretation:

The LiDAR data indicates whether a direction is passable:

  - intense: Indicates a strong reflection signal, meaning the direct path is completely blocked.  
  - strong: Indicates a relatively strong reflection signal, suggesting there is an obstacle nearby, but forward movement may still be possible.  
  - moderate: Indicates a moderate signal strength, implying no obstacles in the mid-to-short range, and passage is possible.  
  - weak: Indicates a weak reflection signal, meaning the path ahead is clear of obstacles.  


## Available Actions

Here are the basic actions for miners so far. You should not hardcode them in the prompt, as the `action_space` already includes all of these descriptions.

  - move_in_direction: Moves in the specified direction.
  - move_to_target: Move towards the specified target entity. Target can be a character or a location and must be in sight.
  - talk_to: Talk to other entity. Accepts the name of the target and the content you want to say. The target may not hear you if you're too far away.
  - check: Examine a specified target to obtain detailed information, such as character status, item identification, or device operation instructions.
  - collect: Collect resources or items from the specified target entity.
  - discard_item: Discard items from the inventory.


## Quest System

As you can see in the action list, there is no `quest_xxx` function. All quest-related operations are handled through TALK, similar to real-life interactions.


## Score and Incentives

The scoring system is still undergoing adjustments and improvements. The basic idea is that the overall score consists of:

* Action Score: Small, easy-to-earn scores that reward miners for VALID actions.
* Quest Score: A larger score, but one that requires a series of planned actions to reward good higher-level design.
