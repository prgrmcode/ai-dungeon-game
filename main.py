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
