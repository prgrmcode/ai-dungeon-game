# Implement an inventory system for the player to collect and use items.
class Item:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect


def use_item(player, item):
    if item.effect == "heal":
        player.health += 20
        print(f"{player.name} used {item.name}. Health is now {player.health}")
