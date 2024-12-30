# Create a combat system where the player and NPCs can engage in battles.
def combat(player, npc):
    while player.health > 0 and npc.health > 0:
        player_action = input("Choose action (attack/flee): ")
        if player_action == "attack":
            npc.health -= 10
            print(f"You attacked {npc.name}. {npc.name} health: {npc.health}")
        elif player_action == "flee":
            print("You fled the battle.")
            break
        if npc.health > 0:
            npc_action = npc.decide_action(player)
            if npc_action == "attack":
                player.health -= 10
                print(f"{npc.name} attacked you. Your health: {player.health}")
            elif npc_action == "flee":
                print(f"{npc.name} fled the battle.")
                break
        adjust_difficulty(player, npc)


def adjust_difficulty(player, npc):
    if player.health < 30:
        npc.health += 10  # Increase NPC health to make it more challenging
    elif player.health > 70:
        npc.health -= 10  # Decrease NPC health to make it easier
