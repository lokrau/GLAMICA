import pandas as pd
from collections import defaultdict

# Load knowledge graph
df = pd.read_csv("support_functions/knowledge_graph.csv")

# Graph structures
ingredient_components = defaultdict(set)
meals = {}
ingredients = set()
meal_tools = defaultdict(set)

# Parse entities and relationships
for _, row in df.iterrows():
    source, relation, target = row["Source"], row["Relation"], row["Target"]

    if relation == "has_ingredient":
        ingredient_components[source].add(target)

    elif relation == "is_a":
        if target == "Meal":
            meals[source] = set()
        elif target == "Ingredient":
            ingredients.add(source)

# Fill in ingredients for each meal
for _, row in df.iterrows():
    source, relation, target = row["Source"], row["Relation"], row["Target"]
    if relation == "has_ingredient" and source in meals:
        meals[source].add(target)

# Fill in tools for each meal
for _, row in df.iterrows():
    source, relation, target = row["Source"], row["Relation"], row["Target"]
    if relation == "uses_tool" and source in meals:
        meal_tools[source].add(target)

# Recursive resolution of ingredients (e.g., sub-ingredients)
def resolve_ingredients(item, seen=None):
    if seen is None:
        seen = set()
    if item in seen:
        return set()
    seen.add(item)

    if item in ingredient_components:
        resolved = set()
        for sub in ingredient_components[item]:
            resolved |= resolve_ingredients(sub, seen)
        return resolved
    else:
        return {item}

# Check if a meal can be made with the given items and tools
def can_make_meal(meal, available_items, available_tools=None):
    if available_tools is None:
        available_tools = set()

    needed_ingredients = set()
    for ing in meals[meal]:
        needed_ingredients |= resolve_ingredients(ing)

    has_ingredients = needed_ingredients <= available_items
    has_tools = meal_tools[meal] <= available_tools

    return has_ingredients and has_tools
