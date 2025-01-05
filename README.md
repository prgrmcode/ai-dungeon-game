---
title: AI Dungeon Game
emoji: 🎮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: main.py
pinned: false
---

# AI-Powered Dungeon Adventure Game

## Table of Contents
- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Key Features](#key-features)
- [Game Mechanics](#game-mechanics)
- [AI/ML Implementation](#aiml-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)

## Overview
An advanced text-based adventure game powered by Large Language Models (LLMs) that demonstrates the practical application of AI/ML in interactive entertainment. The game features dynamic quest generation, intelligent NPC interactions, and content safety validation using state-of-the-art language models.

## Technical Architecture
- **Core Engine**: Python-based game engine with modular architecture
- **AI Integration**: Hugging Face Transformers pipeline for text generation
- **UI Framework**: Gradio for interactive web interface
- **Safety Layer**: LLaMA Guard for content moderation
- **State Management**: Dynamic game state handling with quest progression
- **Memory Management**: Optimized for GPU utilization with 8-bit quantization

## Key Features
1. **Dynamic Quest System**
   - Procedurally generated quests based on player progress
   - Multi-chain quest progression
   - Experience-based leveling system

2. **Intelligent Response Generation**
   - Context-aware narrative responses
   - Dynamic world state adaptation
   - Natural language understanding (NLP)

3. **Advanced Safety System**
   - Real-time content moderation
   - Multi-category safety checks
   - Cached response validation

4. **Inventory Management**
   - Dynamic item tracking
   - Automated inventory updates
   - Natural language parsing for item detection

## Game Mechanics
- **Dungeon Generation:** Randomly generated dungeons with obstacles.
- **Player and NPCs:** Players can move, fight NPCs, and use items.
- **Combat System:** Turn-based combat with simple AI decision-making.
- **Inventory Management:** Collect and use items to aid in your adventure.
- **Quest System:** Complete quests to earn rewards and progress through the game.


## AI/ML Implementation
1. **Language Models**
   - Primary: LLaMA-3.2-3B-Instruct
   - Safety: LLaMA-Guard-3-1B
   - Optimized with 8-bit quantization

2. **Natural Language Processing**
   - Context embedding
   - Response generation
   - Content safety validation

3. **Memory Optimization**
   - GPU memory management
   - Response caching
   - Efficient token handling

## Installation
```bash
# Clone repository
git clone https://github.com/prgrmcode/ai-dungeon-game.git
cd ai-dungeon-game

# Create virtual environment
python -m venv dungeon-env
source dungeon-env/Scripts/activate  # Windows
source dungeon-env/bin/activate      # Linux/Mac

# Install dependencies (`pip freeze > requirements.txt` to get requirements)
pip install -r requirements.txt
# Or install libraries directly:
pip install numpy matplotlib pygame
pip install python-dotenv
pip install transformers
pip install gradio
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install psutil
pip install 'accelerate>=0.26.0

# Create a .env file in the root directory of the project and add your environment variables:
HUGGINGFACE_API_KEY=your_api_key_here
```

## Usage
```bash
# Run the game locally using gpu-compute branch
git checkout gpu-compute
python main.py

# Start the game using deployed main branch
git checkout main
python main.py

# Access via web browser
http://localhost:7860
# or:
http://127.0.0.1:7860
```

## Project Structure
```
ai_dungeon_game/
├── assets/
│   └── ascii_art.py
├── game/
│   ├── combat.py
│   ├── dungeon.py
│   ├── items.py
│   ├── npc.py
│   └── player.py
├── shared_data/
│   └── Ethoria.json
├── helper.py
└── main.py
```

## Technologies Used
- **Python 3.10+**
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face's transformer models
- **Gradio**: Web interface framework
- **CUDA**: GPU acceleration
- **JSON**: Data storage
- **Logging**: Advanced error tracking

## Skills Demonstrated
1. **AI/ML Engineering**
   - Large Language Model implementation
   - Model optimization
   - Prompt engineering
   - Content safety systems

2. **Software Engineering**
   - Clean architecture
   - Object-oriented design
   - Error handling
   - Memory optimization

3. **Data Science**
   - Natural language processing
   - State management
   - Data validation
   - Pattern recognition

4. **System Design**
   - Modular architecture
   - Scalable systems
   - Memory management
   - Performance optimization

## Future Enhancements
1. **Advanced AI Features**
   - Multi-modal content generation
   - Improved context understanding
   - Dynamic difficulty adjustment

2. **Technical Improvements**
   - Distributed computing support
   - Advanced caching mechanisms
   - Real-time model updating

3. **Gameplay Features**
   - Multiplayer support
   - Advanced combat system
   - Dynamic world generation

4. **Visual Enhancements**
   - **Graphical User Interface (GUI):** Implement a GUI using Pygame to provide a more interactive and visually appealing experience.

   - **2D/3D Graphics:** Use libraries like Pygame or Pyglet for 2D graphics.

   - **Animations:** Add animations for player and NPC movements, combat actions, and other in-game events.

   - **Visual Effects:** Implement visual effects such as particle systems for magic spells, explosions, and other dynamic events.

   - **Map Visualization:** Create a visual representation of the dungeon map that updates as the player explores.


## Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Hugging Face API key


## Deployment Options

### Local Docker Deployment
```bash
# Build and run with Docker
docker-compose --env-file .env up --build
```

### Hugging Face Spaces Deployment
1. Fork repository
2. Connect to Hugging Face Spaces
3. Deploy through GitHub Actions

### AWS Deployment

1. Push to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker build -t ai-dungeon .
docker tag ai-dungeon:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-dungeon:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-dungeon:latest
```

2. Deploy to ECS/EKS


### Kubernetes Deployment

```bash
kubectl apply -f kubernetes/
```


This project showcases practical implementation of AI/ML in interactive entertainment, 
demonstrating skills in LLM implementation, system design, and performance optimization.



