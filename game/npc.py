# Create NPCs with simple AI behaviors.
class NPC:
    def __init__(self, name):
        self.name = name
        self.health = 100

    def decide_action(self, player):
        if self.health < 30:
            return "flee"
        else:
            return "attack"
