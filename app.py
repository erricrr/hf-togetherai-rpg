import json
from together import Together
from huggingface_hub import InferenceClient
from helper import get_huggingface_api_key, get_together_api_key, get_game_state, is_safe, start_game

def detect_inventory_changes(game_state, output):
    """
    Detects changes to player's inventory based on the most recent game output.

    Args:
        game_state (dict): Current game state including inventory
        output (str): Latest game output/story text

    Returns:
        list: List of item updates with name and change amount
    """
    inventory = game_state['inventory']
    system_prompt = """You are an AI Game Assistant. \
    Your job is to detect changes to a player's \
    inventory based on the most recent story and game state.
    If a player picks up, or gains an item add it to the inventory \
    with a positive change_amount.
    If a player loses an item remove it from their inventory \
    with a negative change_amount.
    Given a player name, inventory and story, return a list of json update
    of the player's inventory in the following form.
    Only take items that it's clear the player (you) lost.
    Only give items that it's clear the player gained.
    Don't make any other item updates.
    If no items were changed return {"itemUpdates": []}
    and nothing else.

    Response must be in Valid JSON
    Don't add items that were already added in the inventory

    Inventory Updates:
    {
        "itemUpdates": [
            {"name": <ITEM NAME>,
            "change_amount": <CHANGE AMOUNT>}...
        ]
    }
    """

    client = InferenceClient(token=get_huggingface_api_key())

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Current Inventory: {str(inventory)}'},
        {"role": "user", "content": f'Recent Story: {output}'},
        {"role": "user", "content": 'Inventory Updates'}
    ]

    chat_completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0.0,
        messages=messages
    )

    response = chat_completion.choices[0].message.content
    result = json.loads(response)
    return result['itemUpdates']


def update_inventory(inventory, item_updates):
    """
    Updates the inventory based on the detected changes.

    Args:
        inventory (dict): Current inventory
        item_updates (list): List of item updates

    Returns:
        str: Message describing inventory updates
    """
    update_msg = ''

    for update in item_updates:
        name = update['name']
        change_amount = update['change_amount']

        if change_amount > 0:
            if name not in inventory:
                inventory[name] = change_amount
            else:
                inventory[name] += change_amount
            update_msg += f'\nInventory: {name} +{change_amount}'
        elif name in inventory and change_amount < 0:
            inventory[name] += change_amount
            update_msg += f'\nInventory: {name} {change_amount}'

        if name in inventory and inventory[name] <= 0:
            del inventory[name]

    return update_msg


def run_action(message, history, game_state):
    """
    Processes player actions and returns the next part of the story.

    Args:
        message (str): Player's input message
        history (list): Previous interactions
        game_state (dict): Current game state

    Returns:
        str: Game response to the player's action
    """
    if message == 'start game':
        return game_state['start']

    system_prompt = """You are an AI Game master. Your job is to write what \
                    happens next in a player's adventure game.\
                    Instructions: \
                    You must on only write 1-3 sentences in response. \
                    Always write in second person present tense. \
                    Ex. (You look north and see...) \
                    Don't let the player use items they don't have in their inventory.
                    """

    world_info = f"""
                    World: {game_state['world']}
                    Kingdom: {game_state['kingdom']}
                    Town: {game_state['town']}
                    Your Character:  {game_state['character']}
                    Inventory: {json.dumps(game_state['inventory'])}
                """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": world_info}
    ]

    for action in history:
        messages.append({"role": "assistant", "content": action[0]})
        messages.append({"role": "user", "content": action[1]})

    messages.append({"role": "user", "content": message})

    client = InferenceClient(token=get_huggingface_api_key())
    model_output = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=messages
    )

    result = model_output.choices[0].message.content
    return result


def main_loop(message, history):
    """
    Main game loop that processes player input and maintains game state.

    Args:
        message (str): Player's input message
        history (list): Previous interactions

    Returns:
        str: Game response including story and inventory updates
    """
    output = run_action(message, history, game_state)

    safe = is_safe(output)
    if not safe:
        return 'Invalid Output'

    item_updates = detect_inventory_changes(game_state, output)
    update_msg = update_inventory(
        game_state['inventory'],
        item_updates
    )
    output += update_msg

    return output


if __name__ == '__main__':
    # Initialize game state with starting inventory
    game_state = get_game_state(inventory={
        "cloth pants": 1,
        "cloth shirt": 1,
        "goggles": 1,
        "leather bound journal": 1,
        "gold": 5
    })

    # Start the game
    start_game(main_loop, True)
