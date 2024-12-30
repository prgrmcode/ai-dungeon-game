import random
from helper import run_action


class Dungeon:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = self.generate_dungeon()
        self.items = []
        self.npcs = []
        self.story = ""
        self.current_room = "entrance"

    def generate_dungeon(self):
        # Enhanced procedural generation
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        # Create rooms
        rooms = []
        for _ in range(3):
            room_w = random.randint(3, 5)
            room_h = random.randint(3, 5)
            x = random.randint(1, self.width - room_w - 1)
            y = random.randint(1, self.height - room_h - 1)
            rooms.append((x, y, room_w, room_h))

            # Carve out room
            for i in range(y, y + room_h):
                for j in range(x, x + room_w):
                    grid[i][j] = "."

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            x1, y1 = rooms[i][0] + rooms[i][2] // 2, rooms[i][1] + rooms[i][3] // 2
            x2, y2 = (
                rooms[i + 1][0] + rooms[i + 1][2] // 2,
                rooms[i + 1][1] + rooms[i + 1][3] // 2,
            )

            # Horizontal corridor
            for x in range(min(x1, x2), max(x1, x2) + 1):
                grid[y1][x] = "."

            # Vertical corridor
            for y in range(min(y1, y2), max(y1, y2) + 1):
                grid[y2][x2] = "."

        return grid

    def display(self, level=0, player_pos=None):
        """Return ASCII representation of dungeon"""
        grid = self.grid.copy()

        # Mark player position
        if player_pos:
            x, y = player_pos
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = "@"

        # Mark NPCs
        for npc in self.npcs:
            x, y = npc.position
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = "N"

        # Mark items
        for item in self.items:
            x, y = item.position
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = "i"

        # Convert to string
        return "\n".join("".join(row) for row in grid)

    def get_room_description(self, room_name, game_state):
        """Get AI-generated room description based on game state"""
        prompt = f"Describe the {room_name} of the dungeon in the kingdom of {game_state['kingdom']}"
        # Use helper.run_action() to get AI description
        return run_action(prompt, [], game_state)
