from helper import run_action
from typing import Dict, List


# Implement basic movement logic for the player.
class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.max_health = 100
        self.level = 1
        self.exp = 0
        self.exp_to_level = 100
        self.position = [0, 0]
        self.inventory = []
        self.current_quest = None
        self.completed_quests = []
        self.skills = {"combat": 1, "stealth": 1, "magic": 1}

    def gain_exp(self, amount: int) -> bool:
        """Award experience and handle leveling"""
        self.exp += amount
        if self.exp >= self.exp_to_level:
            self.level_up()
            return True
        return False

    def level_up(self):
        """Handle level up effects"""
        self.level += 1
        self.exp -= self.exp_to_level
        self.exp_to_level = int(self.exp_to_level * 1.5)
        self.max_health += 10
        self.health = self.max_health

    def add_quest(self, quest: Dict):
        """Add a new quest to track"""
        self.current_quest = quest

    def complete_quest(self, quest: Dict):
        """Complete a quest and gain rewards"""
        if quest in self.completed_quests:
            return False

        self.completed_quests.append(quest)
        self.gain_exp(quest.get("exp_reward", 50))
        return True

    def interact(self, npc, game_state):
        """Get AI-generated dialogue with NPCs"""
        prompt = f"You talk to {npc.name}. What do they say?"
        dialogue = run_action(prompt, [], game_state)
        return dialogue

    def examine(self, item, game_state):
        """Get AI-generated item description"""
        prompt = f"You examine the {item.name} closely. What do you discover?"
        description = run_action(prompt, [], game_state)
        return description
