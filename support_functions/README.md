# Support Functions for GLAMICA Project
## Author: Lorenz Krause

This directory contains support functions that have been outsourced from the main project directory. These functions are essential for the following tasks:

1. ```meal_knowledge.py``` and ```knowledge_graph.csv```: 
    - These are used to get the possible meals/tasks from the given objects in the user's kitchen.
2. ```looking_at.py```:
    - This function is used to determine which item the user is looking at to give this information to the LLM (Large Language Model) for further processing.
3. ```action_recognition.py```:
    - This file contains functions to recognize actions based on the audio and the recognized objects in the user's kitchen.