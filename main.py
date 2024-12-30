from game.dungeon import Dungeon
from game.player import Player
from game.npc import NPC
from game.combat import combat, adjust_difficulty
from game.items import Item, use_item
from assets.ascii_art import display_dungeon
from helper import (
    get_game_state,
    run_action,
    start_game,
    is_safe,
    detect_inventory_changes,
    update_inventory,
)

import pdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_loop(message, history, game_state):
    """Main game loop that processes commands and returns responses"""

    try:
        print(f"\nProcessing command: {message}")  # Debug print

        # Get AI response
        output = run_action(message, history, game_state)
        logger.info(f"Generated response: {output}")
        print(f"\nGenerated output: {output}")  # Debug print

        # Safety check
        safe = is_safe(output)
        print(f"\nSafety Check Result: {'SAFE' if safe else 'UNSAFE'}")
        logger.info(f"Safety check result: {'SAFE' if safe else 'UNSAFE'}")

        if not safe:
            logging.warning("Unsafe output detected")
            logger.warning("Unsafe content detected - blocking response")
            print("Unsafe content detected - Response blocked")
            return "This response was blocked for safety reasons.", history

        # Update history with safe response
        if not history:
            history = []
        history.append((message, output))

        return output, history

    except Exception as e:
        logger.error(f"Error in main_loop: {str(e)}")
        return "Error processing command", history


def main():
    """Initialize game and start interface"""
    try:
        logger.info("Starting main function")
        print("\nInitializing game...")
        game_state = get_game_state(
            inventory={
                "cloth pants": 1,
                "cloth shirt": 1,
                "goggles": 1,
                "leather bound journal": 1,
                "gold": 5,
            }
        )
        logger.debug(f"Initialized game state: {game_state}")

        # Create and add game objects
        dungeon = Dungeon(10, 10)
        player = Player("Hero")

        # Update game state with objects
        game_state["dungeon"] = dungeon
        game_state["player"] = player

        # logger.info(f"Game state in main(): {game_state}")

        # Start game interface
        print("Starting game interface...")
        start_game(main_loop, game_state, True)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# def main_loop(message, history):
#     logging.info(f"main_loop called with message: {message}")

#     # Initialize history if None
#     history = history or []

#     # Get AI response
#     output = run_action(message, history, game_state)

#     # Safety check
#     safe = is_safe(output)
#     if not safe:
#         logging.error("Unsafe output detected")
#         return "Invalid Output"

#     # Format the output nicely
#     output_lines = [output]

#     # Handle movement and exploration
#     if message.lower().startswith(("go", "move", "walk")):
#         direction = message.split()[1]
#         game_state["player"].move(direction, game_state["dungeon"])
#         room_desc = game_state["dungeon"].get_room_description(
#             game_state["dungeon"].current_room, game_state
#         )
#         output_lines.append(f"\n{room_desc}")

#     # Handle NPC interactions
#     elif message.lower().startswith("talk"):
#         npc_name = message.split()[2]
#         for npc in game_state["dungeon"].npcs:
#             if npc.name.lower() == npc_name.lower():
#                 dialogue = game_state["player"].interact(npc, game_state)
#                 output_lines += f"\n{dialogue}"

#     # Handle item interactions and inventory
#     elif message.lower().startswith(("take", "pick up")):
#         item_name = " ".join(message.split()[1:])
#         for item in game_state["dungeon"].items:
#             if item.name.lower() == item_name.lower():
#                 game_state["player"].inventory.append(item)
#                 game_state["dungeon"].items.remove(item)
#                 output += f"\nYou picked up {item.name}"
#                 item_desc = game_state["player"].examine(item, game_state)
#                 output_lines += f"\n{item_desc}"

#     # Format final output
#     final_output = "\n".join(output_lines)
#     history.append((message, final_output))
#     logging.info(f"main_loop output: {final_output}")

#     return final_output, history


# def main():
#     logging.info("Starting main function")

#     try:
#         # Initialize game state with error handling
#         global game_state
#         game_state = get_game_state(
#             inventory={
#                 "cloth pants": 1,
#                 "cloth shirt": 1,
#                 "goggles": 1,
#                 "leather bound journal": 1,
#                 "gold": 5,
#             }
#         )

#         # Verify game state initialization
#         if not game_state:
#             raise ValueError("Failed to initialize game state")

#         # Create dungeon and populate with AI-generated content
#         dungeon = Dungeon(10, 10)
#         game_state["dungeon"] = dungeon

#         # Create player and add to game state
#         player = Player("Hero")
#         game_state["player"] = player

#         # Start game interface
#         start_game(main_loop, True)

#     except Exception as e:
#         logging.error(f"Error in main: {str(e)}")
#         raise
