import os
import re
from dotenv import load_dotenv, find_dotenv
import json
import gradio as gr
import torch  # first import torch then transformers

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import psutil
from typing import Dict, Any, Optional, Tuple

# Add model caching and optimization
from functools import lru_cache
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_memory():
    """Get available GPU and system memory"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
    system_memory = psutil.virtual_memory().available
    return gpu_memory, system_memory


def load_env():
    _ = load_dotenv(find_dotenv())


def get_huggingface_api_key():
    load_env()
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        logging.error("HUGGINGFACE_API_KEY not found in environment variables")
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
    return huggingface_api_key


# Model configuration
MODEL_CONFIG = {
    "main_model": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "dtype": torch.bfloat16,
        "max_length": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "safety_model": {
        "name": "meta-llama/Llama-Guard-3-1B",
        "dtype": torch.bfloat16,
        "max_length": 256,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def initialize_model_pipeline(model_name, force_cpu=False):
    """Initialize pipeline with memory management"""
    try:
        if force_cpu:
            device = -1
        else:
            device = MODEL_CONFIG["main_model"]["device"]

        # Use 8-bit quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=MODEL_CONFIG["main_model"]["dtype"],
            use_cache=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = True

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize pipeline
        logger.info(f"Initializing pipeline with device: {device}")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=device,
            # temperature=0.7,
            model_kwargs={"low_cpu_mem_usage": True},
        )

        logger.info("Model Pipeline initialized successfully")
        return generator, tokenizer

    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise


# Initialize model pipeline
try:
    # Use a smaller model for testing
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "google/gemma-2-2b"  # Start with a smaller model
    # model_name = "microsoft/phi-2"
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"

    model_name = MODEL_CONFIG["main_model"]["name"]

    api_key = get_huggingface_api_key()

    # Initialize the pipeline with memory management
    generator, tokenizer = initialize_model_pipeline(model_name)

except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    # Fallback to CPU if GPU initialization fails
    try:
        logger.info("Attempting CPU fallback...")
        generator, tokenizer = initialize_model_pipeline(model_name, force_cpu=True)
    except Exception as e:
        logger.error(f"CPU fallback failed: {str(e)}")
        raise


def load_world(filename):
    with open(filename, "r") as f:
        return json.load(f)


# Define system_prompt and model
system_prompt = """You are an AI Game Master. Write ONE response describing what the player sees/experiences.
CRITICAL Rules:
- Write EXACTLY 3 sentences maximum
- Use daily English language
- Start with "You see", "You hear", or "You feel"
- Don't use 'Elara' or 'she/he', only use 'you'
- Use only second person ("you")
- Never include dialogue after the response
- Never continue with additional actions or responses
- Never add follow-up questions or choices
- Never include 'User:' or 'Assistant:' in response
- Never include any note or these kinds of sentences: 'Note from the game master'
- Never use ellipsis (...)
- Never include 'What would you like to do?' or similar prompts
- Always finish with one real response
- End the response with a period"""


def get_game_state(inventory: Dict = None) -> Dict[str, Any]:
    """Initialize game state with safe defaults and quest system"""
    try:
        # Load world data
        world = load_world("shared_data/Ethoria.json")
        character = world["kingdoms"]["Valdor"]["towns"]["Ravenhurst"]["npcs"][
            "Elara Brightshield"
        ]
        print(f"character in get_game_state: {character}")

        game_state = {
            "name": world["name"],
            "world": world["description"],
            "kingdom": world["kingdoms"]["Valdor"]["description"],
            "town_name": world["kingdoms"]["Valdor"]["towns"]["Ravenhurst"]["name"],
            "town": world["kingdoms"]["Valdor"]["towns"]["Ravenhurst"]["description"],
            "character_name": character["name"],
            "character_description": character["description"],
            "start": world["start"],
            "inventory": inventory
            or {
                "cloth pants": 1,
                "cloth shirt": 1,
                "goggles": 1,
                "leather bound journal": 1,
                "gold": 5,
            },
            "player": None,
            "dungeon": None,
            "current_quest": None,
            "completed_quests": [],
            "exp": 0,
            "level": 1,
            "reputation": {"Valdor": 0, "Ravenhurst": 0},
        }

        # print(f"game_state in get_game_state: {game_state}")

        # Extract required data with fallbacks
        return game_state
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error loading world data: {e}")
        # Provide default values if world loading fails
        return {
            "world": "Ethoria is a realm of seven kingdoms, each founded on distinct moral principles.",
            "kingdom": "Valdor, the Kingdom of Courage",
            "town": "Ravenhurst, a town of skilled hunters and trappers",
            "character_name": "Elara Brightshield",
            "character_description": "A sturdy warrior with shining silver armor",
            "start": "Your journey begins in the mystical realm of Ethoria...",
            "inventory": inventory
            or {
                "cloth pants": 1,
                "cloth shirt": 1,
                "goggles": 1,
                "leather bound journal": 1,
                "gold": 5,
            },
            "player": None,
            "dungeon": None,
            "current_quest": None,
            "completed_quests": [],
            "exp": 0,
            "level": 1,
            "reputation": {"Valdor": 0, "Ravenhurst": 0},
        }


def generate_dynamic_quest(game_state: Dict) -> Dict:
    """Generate varied quests based on progress and level"""
    completed = len(game_state.get("completed_quests", []))
    level = game_state.get("level", 1)

    # Quest templates by type
    quest_types = {
        "combat": [
            {
                "title": "The Beast's Lair",
                "description": "A fearsome {creature} has been terrorizing the outskirts of Ravenhurst.",
                "objective": "Hunt down and defeat the {creature}.",
                "creatures": [
                    "shadow wolf",
                    "frost bear",
                    "ancient wyrm",
                    "spectral tiger",
                ],
            },
        ],
        "exploration": [
            {
                "title": "Lost Secrets",
                "description": "Rumors speak of an ancient {location} containing powerful artifacts.",
                "objective": "Explore the {location} and uncover its secrets.",
                "locations": [
                    "crypt",
                    "temple ruins",
                    "abandoned mine",
                    "forgotten library",
                ],
            },
        ],
        "mystery": [
            {
                "title": "Dark Omens",
                "description": "The {sign} has appeared, marking the rise of an ancient power.",
                "objective": "Investigate the meaning of the {sign}.",
                "signs": [
                    "blood moon",
                    "mysterious runes",
                    "spectral lights",
                    "corrupted wildlife",
                ],
            },
        ],
    }

    # Select quest type and template
    quest_type = list(quest_types.keys())[completed % len(quest_types)]
    template = quest_types[quest_type][0]  # Could add more templates per type

    # Fill in dynamic elements
    if quest_type == "combat":
        creature = template["creatures"][level % len(template["creatures"])]
        title = template["title"]
        description = template["description"].format(creature=creature)
        objective = template["objective"].format(creature=creature)
    elif quest_type == "exploration":
        location = template["locations"][level % len(template["locations"])]
        title = template["title"]
        description = template["description"].format(location=location)
        objective = template["objective"].format(location=location)
    else:  # mystery
        sign = template["signs"][level % len(template["signs"])]
        title = template["title"]
        description = template["description"].format(sign=sign)
        objective = template["objective"].format(sign=sign)

    return {
        "id": f"quest_{quest_type}_{completed}",
        "title": title,
        "description": f"{description} {objective}",
        "exp_reward": 150 + (level * 50),
        "status": "active",
        "triggers": ["investigate", "explore", quest_type, "search"],
        "completion_text": f"You've made progress in understanding the growing darkness.",
        "next_quest_hint": "More mysteries await in the shadows of Ravenhurst.",
    }


def generate_next_quest(game_state: Dict) -> Dict:
    """Generate next quest based on progress"""
    completed = len(game_state.get("completed_quests", []))
    level = game_state.get("level", 1)

    quest_chain = [
        {
            "id": "mist_investigation",
            "title": "Investigate the Mist",
            "description": "Strange mists have been gathering around Ravenhurst. Investigate their source.",
            "exp_reward": 100,
            "status": "active",
            "triggers": ["mist", "investigate", "explore"],
            "completion_text": "As you investigate the mist, you discover ancient runes etched into nearby stones.",
            "next_quest_hint": "The runes seem to point to an old hunting trail.",
        },
        {
            "id": "hunters_trail",
            "title": "The Hunter's Trail",
            "description": "Local hunters have discovered strange tracks in the forest. Follow them to their source.",
            "exp_reward": 150,
            "status": "active",
            "triggers": ["tracks", "follow", "trail"],
            "completion_text": "The tracks lead to an ancient well, where you hear strange whispers.",
            "next_quest_hint": "The whispers seem to be coming from deep within the well.",
        },
        {
            "id": "dark_whispers",
            "title": "Whispers in the Dark",
            "description": "Mysterious whispers echo from the old well. Investigate their source.",
            "exp_reward": 200,
            "status": "active",
            "triggers": ["well", "whispers", "listen"],
            "completion_text": "You discover an ancient seal at the bottom of the well.",
            "next_quest_hint": "The seal bears markings of an ancient evil.",
        },
    ]

    # Generate dynamic quests after initial chain
    if completed >= len(quest_chain):
        return generate_dynamic_quest(game_state)

    # current_quest_index = min(completed, len(quest_chain) - 1)
    # return quest_chain[current_quest_index]
    return quest_chain[completed]


def check_quest_completion(message: str, game_state: Dict) -> Tuple[bool, str]:
    """Check quest completion and handle progression"""
    if not game_state.get("current_quest"):
        return False, ""

    quest = game_state["current_quest"]
    triggers = quest.get("triggers", [])

    if any(trigger in message.lower() for trigger in triggers):
        # Award experience
        exp_reward = quest.get("exp_reward", 100)
        game_state["exp"] += exp_reward

        # Update player level if needed
        while game_state["exp"] >= 100 * game_state["level"]:
            game_state["level"] += 1
            game_state["player"].level = (
                game_state["level"] if game_state.get("player") else game_state["level"]
            )

        level_up_text = (
            f"\nLevel Up! You are now level {game_state['level']}!"
            if game_state["exp"] >= 100 * (game_state["level"] - 1)
            else ""
        )

        # Store completed quest
        game_state["completed_quests"].append(quest)

        # Generate next quest
        next_quest = generate_next_quest(game_state)
        game_state["current_quest"] = next_quest

        # Update status display
        if game_state.get("player"):
            game_state["player"].exp = game_state["exp"]
            game_state["player"].level = game_state["level"]

        # Build completion message
        completion_msg = f"""
Quest Complete: {quest['title']}! (+{exp_reward} exp){level_up_text}
{quest.get('completion_text', '')}

New Quest: {next_quest['title']}
{next_quest['description']}
{next_quest.get('next_quest_hint', '')}"""

        return True, completion_msg

    return False, ""


def parse_items_from_story(text: str) -> Dict[str, int]:
    """Extract item changes from story text"""
    items = {}

    # Common item keywords and patterns
    gold_pattern = r"(\d+)\s*gold"
    items_pattern = (
        r"(?:receive|find|given|hand|containing)\s+(?:a|an|the)?\s*(\d+)?\s*([\w\s]+)"
    )

    # Find gold amounts
    gold_matches = re.findall(gold_pattern, text.lower())
    if gold_matches:
        items["gold"] = sum(int(x) for x in gold_matches)

    # Find other items
    item_matches = re.findall(items_pattern, text.lower())
    for count, item in item_matches:
        count = int(count) if count else 1
        item = item.strip()
        if item in items:
            items[item] += count
        else:
            items[item] = count

    return items


def update_game_inventory(game_state: Dict, story_text: str) -> str:
    """Update inventory based on story and return update message"""
    try:
        items = parse_items_from_story(story_text)
        update_msg = ""

        for item, count in items.items():
            if item in game_state["inventory"]:
                game_state["inventory"][item] += count
            else:
                game_state["inventory"][item] = count
            update_msg += f"\nReceived: {count} {item}"

        return update_msg
    except Exception as e:
        logger.error(f"Error updating inventory: {e}")
        return ""


def extract_response_after_action(full_text: str, action: str) -> str:
    """Extract response text that comes after the user action line"""
    try:
        # Split into lines
        lines = full_text.split("\n")

        # Find index of line containing user action
        action_line_index = -1
        for i, line in enumerate(lines):
            if f"user: {action}" in line:
                action_line_index = i
                break

        if action_line_index >= 0:
            # Get all lines after the action line
            response_lines = lines[action_line_index + 1 :]
            response = " ".join(line.strip() for line in response_lines if line.strip())

            # Clean up any remaining markers
            response = response.split("user:")[0].strip()
            response = response.split("system:")[0].strip()

            return response

        return ""

    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        return ""


def run_action(message: str, history: list, game_state: Dict) -> str:
    """Process game actions and generate responses with quest handling"""
    try:
        # Handle start game command
        if message.lower() == "start game":
            # # Generate initial quest
            # initial_quest = {
            #     "title": "Investigate the Mist",
            #     "description": "Strange mists have been gathering around Ravenhurst. Investigate their source.",
            #     "exp_reward": 100,
            #     "status": "active",
            # }
            # game_state["current_quest"] = initial_quest
            # Initialize first quest
            initial_quest = generate_next_quest(game_state)
            game_state["current_quest"] = initial_quest

            start_response = f"""Welcome to {game_state['name']}. {game_state['world']}

{game_state['start']}

Currently in {game_state['town_name']}, in the kingdom of {game_state['kingdom']}.
{game_state['town']}


Current Quest: {initial_quest['title']}
{initial_quest['description']}

What would you like to do?"""
            return start_response

        # Verify game state
        if not isinstance(game_state, dict):
            logger.error(f"Invalid game state type: {type(game_state)}")
            return "Error: Invalid game state"

        # logger.info(f"Processing action with game state: {game_state}")
        logger.info(f"Processing action with game state")

        world_info = f"""World: {game_state['world']}
Kingdom: {game_state['kingdom']}
Town: {game_state['town']}
Character: {game_state['character_name']}
Current Quest: {game_state["current_quest"]['title']}
Quest Objective: {game_state["current_quest"]['description']}
Inventory: {json.dumps(game_state['inventory'])}"""

        #         # Enhanced system prompt for better response formatting
        #         enhanced_prompt = f"""{system_prompt}
        # Additional Rules:
        # - Always start responses with 'You ', 'You see' or 'You hear' or 'You feel'
        # - Use ONLY second person perspective ('you', not 'Elara' or 'she/he')
        # - Describe immediate surroundings and sensations
        # - Keep responses focused on the player's direct experience"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": world_info},
        ]

        # Format chat history
        if history:
            for h in history:
                if isinstance(h, tuple):
                    messages.append({"role": "assistant", "content": h[0]})
                    messages.append({"role": "user", "content": h[1]})

        messages.append({"role": "user", "content": message})

        # Convert messages to string format for pipeline
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Generate response
        model_output = generator(
            prompt,
            max_new_tokens=len(tokenizer.encode(message))
            + 120,  # Set max_new_tokens based on input length
            num_return_sequences=1,
            # temperature=0.7,  # More creative but still focused
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Extract and clean response
        full_response = model_output[0]["generated_text"]
        print(f"full_response in run_action: {full_response}")

        response = extract_response_after_action(full_response, message)
        print(f"response in run_action: {response}")

        # Convert to second person
        response = response.replace("Elara", "You")

        # # Format response
        # if not response.startswith("You"):
        #     response = "You see " + response

        # Validate no cut-off sentences
        if response.rstrip().endswith(("you also", "meanwhile", "suddenly", "...")):
            response = response.rsplit(" ", 1)[0]  # Remove last word

        # Ensure proper formatting
        response = response.rstrip("?").rstrip(".") + "."
        response = response.replace("...", ".")

        # Perform safety check before returning
        safe = is_safe(response)
        print(f"\nSafety Check Result: {'SAFE' if safe else 'UNSAFE'}")
        logger.info(f"Safety check result: {'SAFE' if safe else 'UNSAFE'}")

        if not safe:
            logging.warning("Unsafe content detected - blocking response")
            print("Unsafe content detected - Response blocked")
            return "This response was blocked for safety reasons."

        # # Add quest progress checks
        # if game_state["current_quest"]:
        #     quest = game_state["current_quest"]
        #     # Check for quest completion keywords
        #     if any(
        #         word in message.lower() for word in ["investigate", "explore", "search"]
        #     ):
        #         if (
        #             "mist" in message.lower()
        #             and quest["title"] == "Investigate the Mist"
        #         ):
        #             game_state["player"].complete_quest(quest)
        #             response += "\n\nQuest Complete: Investigate the Mist! (+100 exp)"

        if safe:
            # Check for quest completion
            quest_completed, quest_message = check_quest_completion(message, game_state)
            if quest_completed:
                response += quest_message

            # Check for item updates
            inventory_update = update_game_inventory(game_state, response)
            if inventory_update:
                response += inventory_update

        # Validate response
        return response if response else "You look around carefully."

    except KeyError as e:
        logger.error(f"Missing required game state key: {e}")
        return "Error: Game state is missing required information"
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return (
            "I apologize, but I had trouble processing that command. Please try again."
        )


def update_game_status(game_state: Dict) -> Tuple[str, str]:
    """Generate updated status and quest display text"""
    # Status text
    status_text = (
        f"Health: {game_state.get('player').health if game_state.get('player') else 100}/100\n"
        f"Level: {game_state.get('level', 1)}\n"
        f"Exp: {game_state.get('exp', 0)}/{100 * game_state.get('level', 1)}"
    )

    # Quest text
    quest_text = "No active quest"
    if game_state.get("current_quest"):
        quest = game_state["current_quest"]
        quest_text = f"{quest['title']}\n{quest['description']}"
        if quest.get("next_quest_hint"):
            quest_text += f"\n{quest['next_quest_hint']}"

    return status_text, quest_text


def chat_response(message: str, chat_history: list, current_state: dict) -> tuple:
    """Process chat input and return response with updates"""
    try:
        if not message.strip():
            return chat_history, current_state, "", ""

        # Get AI response
        output = run_action(message, chat_history, current_state)

        # Update chat history without status info
        chat_history = chat_history or []
        chat_history.append((message, output))

        # # Create status text
        # status_text = "Health: 100/100\nLevel: 1\nExp: 0/100"
        # if current_state.get("player"):
        #     status_text = (
        #         f"Health: {current_state['player'].health}/{current_state['player'].max_health}\n"
        #         f"Level: {current_state['player'].level}\n"
        #         f"Exp: {current_state['player'].exp}/{current_state['player'].exp_to_level}"
        #     )

        # quest_text = "No active quest"
        # if current_state.get("current_quest"):
        #     quest = current_state["current_quest"]
        #     quest_text = f"{quest['title']}\n{quest['description']}"

        # Update status displays
        status_text, quest_text = update_game_status(current_state)

        # Return tuple includes empty string to clear input
        return chat_history, current_state, status_text, quest_text

    except Exception as e:
        logger.error(f"Error in chat response: {e}")
        return chat_history, current_state, "", ""


def start_game(main_loop, game_state, share=False):
    """Initialize and launch game interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AI Dungeon Adventure")

        # Game state storage
        state = gr.State(game_state)
        history = gr.State([])

        with gr.Row():
            # Game display
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=550,
                    placeholder="Type 'start game' to begin",
                )

                # Input area with submit button
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="What do you want to do?",
                        container=False,
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear = gr.ClearButton([txt, chatbot])

            # Enhanced Status panel
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Character Status")
                    status = gr.Textbox(
                        label="Status",
                        value="Health: 100/100\nLevel: 1\nExp: 0/100",
                        interactive=False,
                    )

                    quest_display = gr.Textbox(
                        label="Current Quest",
                        value="No active quest",
                        interactive=False,
                    )

                    inventory_data = [
                        [item, count]
                        for item, count in game_state.get("inventory", {}).items()
                    ]
                    inventory = gr.Dataframe(
                        value=inventory_data,
                        headers=["Item", "Quantity"],
                        label="Inventory",
                        interactive=False,
                    )

        # Command suggestions
        gr.Examples(
            examples=[
                "look around",
                "continue the story",
                "take sword",
                "go to the forest",
            ],
            inputs=txt,
        )

        # def chat_response(
        #     message: str, chat_history: list, current_state: dict
        # ) -> tuple:
        #     """Process chat input and return response with updates"""
        #     try:
        #         if not message.strip():
        #             return chat_history, current_state, ""  # Only clear input

        #         # Get AI response
        #         output = run_action(message, chat_history, current_state)

        #         # Update chat history
        #         chat_history = chat_history or []
        #         chat_history.append((message, output))

        #         # Update status if player exists
        #         # Update displays
        #         status_text = (
        #             f"Health: {current_state['player'].health}/{current_state['player'].max_health}\n"
        #             f"Level: {current_state['player'].level}\n"
        #             f"Exp: {current_state['player'].exp}/{current_state['player'].exp_to_level}"
        #         )

        #         quest_text = "No active quest"
        #         if current_state["current_quest"]:
        #             quest = current_state["current_quest"]
        #             quest_text = f"{quest['title']}\n{quest['description']}"

        #         # Update inventory display
        #         inv_data = [
        #             [item, count]
        #             for item, count in current_state.get("inventory", {}).items()
        #         ]

        #         return (chat_history, current_state, status_text, quest_text)

        #     except Exception as e:
        #         logger.error(f"Error in chat response: {e}")
        #         return (
        #             chat_history + [(message, "Error processing command")],
        #             current_state,
        #             status.value,
        #             quest_display.value,
        #         )

        # # Connect components
        # submit_btn.click(
        #     chat_response,
        #     inputs=[txt, chatbot, state],
        #     # outputs=[chatbot, state, txt],
        #     outputs=[chatbot, state, status, quest_display],
        #     show_progress=True,
        # )

        # # txt.submit(
        # #     chat_response,
        # #     [txt, chatbot, state],
        # #     [chatbot, state, txt],
        # #     show_progress=True,
        # # )

        # txt.submit(
        #     chat_response,
        #     inputs=[txt, chatbot, state],
        #     outputs=[chatbot, state, status, quest_display],  # Keep status separate
        #     show_progress=True,
        # )

        def submit_action(message, history, state):
            # Process response
            new_history, new_state, status_text, quest_text = chat_response(
                message, history, state
            )
            # Clear input
            return "", new_history, new_state, status_text, quest_text

        submit_btn.click(
            submit_action,
            inputs=[txt, chatbot, state],
            outputs=[txt, chatbot, state, status, quest_display],
        )

        txt.submit(
            submit_action,
            inputs=[txt, chatbot, state],
            outputs=[txt, chatbot, state, status, quest_display],
        )

        demo.launch(share=share)


everyone_content_policy = {
    "policy": """O1: Violence and Hate.
Should not
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics.
- Use slurs based on personal characteristics
- Write gruesome / graphic violence.
Can
- Write tame fantasy violence
O2: Sexual Content.
Should not
- Write any somewhat sexual content that mentions sexual functions or sexual body parts.
- Write any content that would be inappropriate for kids.
Can
- Write romantic content (including kissing, flirting etc...)
O3: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Write story depicting suicide.
O4: Profanity.
Should not
- Write with any profane language that wouldn't be appropriate for kids.
Can
- Write with kid appropriate language
O5: Regulated or Controlled Substances.
Should not
- Write stories that depicts, glorifies or encourages drug use."""
}


def init_safety_model(model_name, force_cpu=False):
    """Initialize safety checking model with optimized memory usage"""
    try:
        if force_cpu:
            device = -1
        else:
            device = MODEL_CONFIG["safety_model"]["device"]

        # model_id = "meta-llama/Llama-Guard-3-8B"
        # model_id = "meta-llama/Llama-Guard-3-1B"

        safety_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=MODEL_CONFIG["safety_model"]["dtype"],
            use_cache=True,
            device_map="auto",
        )
        safety_model.config.use_cache = True

        safety_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token explicitly
        safety_tokenizer.pad_token = safety_tokenizer.eos_token

        logger.info(f"Safety model initialized successfully on {device}")
        return safety_model, safety_tokenizer

    except Exception as e:
        logger.error(f"Failed to initialize safety model: {e}")
        raise


# Initialize safety model pipeline
try:
    safety_model_name = MODEL_CONFIG["safety_model"]["name"]

    api_key = get_huggingface_api_key()

    # Initialize the pipeline with memory management
    safety_model, safety_tokenizer = init_safety_model(safety_model_name)

except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    # Fallback to CPU if GPU initialization fails
    try:
        logger.info("Attempting CPU fallback...")
        safety_model, safety_tokenizer = init_safety_model(
            safety_model_name, force_cpu=True
        )
    except Exception as e:
        logger.error(f"CPU fallback failed: {str(e)}")
        raise


@lru_cache(maxsize=1000)
def get_safety_response(prompt: str) -> str:
    """Cached safety check responses"""
    with torch.no_grad():
        inputs = safety_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODEL_CONFIG["safety_model"]["max_length"],
        ).to(safety_model.device)

        output = safety_model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,  # Lower temperature for more consistent output
            pad_token_id=safety_tokenizer.pad_token_id,
            eos_token_id=safety_tokenizer.eos_token_id,
        )

        return safety_tokenizer.decode(output[0], skip_special_tokens=True)


def is_safe(message: str) -> bool:
    """Check if message content is safe"""
    try:
        logger.info("Performing safety check...")
        # Build the prompt with embedded values
        # prompt = f"""[INST] Task: Check if this content is safe according to our safety policy with the below categories.

        # <BEGIN UNSAFE CONTENT CATEGORIES>
        # {everyone_content_policy['policy']}
        # <END UNSAFE CONTENT CATEGORIES>

        # <CONTENT TO CHECK>
        # {message}
        # <END CONTENT TO CHECK>

        # Provide your safety assessment for user in the above CONTENT TO CHECK section:
        # - Reply ONLY with single word 'safe' or 'unsafe' [/INST]
        # """

        prompt = f"""[INST]Check if safe:
{message}
Reply 'safe' or 'unsafe'[/INST]
"""

        # Generate safety check on cuda

        # with torch.no_grad():
        #     inputs = safety_tokenizer(
        #         prompt,
        #         return_tensors="pt",
        #         padding=True,
        #         truncation=True,
        #     )

        #     # Move inputs to correct device
        #     inputs = {k: v.to(device) for k, v in inputs.items()}

        #     output = safety_model.generate(
        #         **inputs,
        #         max_new_tokens=10,
        #         temperature=0.1,  # Lower temperature for more consistent output
        #         pad_token_id=safety_tokenizer.pad_token_id,  # Use configured pad token
        #         eos_token_id=safety_tokenizer.eos_token_id,
        #         do_sample=False,
        #     )

        # result = safety_tokenizer.decode(output[0], skip_special_tokens=True)
        result = get_safety_response(prompt)
        print(f"Raw safety check result: {result}")

        # # Extract response after prompt
        # if "[/INST]" in result:
        #     result = result.split("[/INST]")[-1]

        # # Clean response
        # result = result.lower().strip()
        # print(f"Cleaned safety check result: {result}")
        # words = [word for word in result.split() if word in ["safe", "unsafe"]]

        # # Take first valid response word
        # is_safe = words[0] == "safe" if words else False

        # print("Final Safety check result:", is_safe)

        is_safe = "safe" in result.lower().split()

        logger.info(
            f"Safety check completed - Result: {'SAFE' if is_safe else 'UNSAFE'}"
        )
        return is_safe

    except Exception as e:
        logger.error(f"Safety check failed: {e}")
        return False


def detect_inventory_changes(game_state, output):
    inventory = game_state["inventory"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Current Inventory: {str(inventory)}"},
        {"role": "user", "content": f"Recent Story: {output}"},
        {"role": "user", "content": "Inventory Updates"},
    ]

    input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    model_output = generator(input_text, num_return_sequences=1, temperature=0.0)
    response = model_output[0]["generated_text"]
    result = json.loads(response)
    return result["itemUpdates"]


def update_inventory(inventory, item_updates):
    update_msg = ""
    for update in item_updates:
        name = update["name"]
        change_amount = update["change_amount"]
        if change_amount > 0:
            if name not in inventory:
                inventory[name] = change_amount
            else:
                inventory[name] += change_amount
            update_msg += f"\nInventory: {name} +{change_amount}"
        elif name in inventory and change_amount < 0:
            inventory[name] += change_amount
            update_msg += f"\nInventory: {name} {change_amount}"
        if name in inventory and inventory[name] < 0:
            del inventory[name]
    return update_msg


logging.info("Finished helper function")
