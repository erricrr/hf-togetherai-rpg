# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
import json
import gradio as gr
from together import Together
from huggingface_hub import InferenceClient


# these expect to find a .env file at the directory above the lesson.                                                         # the format for that file is (without the comment)                                                                           # API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService
def load_env():
    _ = load_dotenv(find_dotenv())

def load_world(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_huggingface_api_key():
    load_env()
    return os.getenv("HF_TOKEN")

def get_together_api_key():

     load_env()
     together_api_key = os.getenv("TOGETHER_API_KEY")
     return together_api_key


def save_world(world, filename):
    with open(filename, 'w') as f:
        json.dump(world, f)

def get_game_state(inventory={}):
    world = load_world('Kyrethia.json')
    kingdom = world['kingdoms']['Vorgath']
    town = kingdom['towns']['Kragnir']
    character = town['npcs']['Eira Shadowglow']

    start = world['start']

    game_state = {
        "world": world['description'],
        "kingdom": kingdom['description'],
        "town": town['description'],
        "character": character['description'],
        "start": start,
        "inventory": inventory
    }
    return game_state

def run_action(message, history, game_state):

    if(message == 'start game'):
        return game_state['start']

    system_prompt = """You are an AI Game master. Your job is to write what \
happens next in a player's adventure game.\
Instructions: \
You must on only write 1-3 sentences in response. \
Always write in second person present tense. \
Ex. (You look north and see...)"""

    world_info = f"""
World: {game_state['world']}
Kingdom: {game_state['kingdom']}
Town: {game_state['town']}
Your Character:  {game_state['character']}"""


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


def start_game(main_loop, share=False):
    demo = gr.ChatInterface(
        main_loop,
        chatbot=gr.Chatbot(height=250,
                           placeholder="Type 'start game' to begin",
                           ),
        textbox=gr.Textbox(placeholder="What do you do next?", container=False, scale=7),
        title="Text-based Fantasy-world Adventure",
        description="AI Role-playing Game (RPG)",
        theme="citrus",
        examples=["Look around", "Continue the story"],
        cache_examples=False,
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
                           )
    demo.launch(share=share, server_name="0.0.0.0")

    return demo


everyone_content_policy = {
    "policy": """O1: Violence and Hate.
Should not
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics.
- Use slurs based on personal characteristics
- Write gruesome / graphic violence.
Can
- Write tame fantasy violence
O2: Sexual Content.
Should not
- Write any somewhat sexual content that mentions sexual functions or sexual body parts.
- Write any content that would be inappropriate for kids.
Can
- Write romantic content (including kissing, flirting etc...)
O3: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Write story depicting suicide.
O4: Profanity.
Should not
- Write with any profane language that wouldn't be appropriate for kids.
Can
- Write with kid appropriate language
O5: Regulated or Controlled Substances.
Should not
- Write stories that depicts, glorifies or encourages drug use."""
}

def is_safe(message):
    # Build the prompt with embedded values
    prompt = f"""[INST] Task: Check if there is unsafe content in 'user' messages in conversations according to our safety policy with the below categories.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {everyone_content_policy['policy']}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>
    user: {message}

    Provide your safety assessment for user in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""

    client = Together(api_key=get_together_api_key())

    response = client.completions.create(
        model="Meta-Llama/LlamaGuard-2-8b",
        prompt=prompt,
    )
    result = response.choices[0].text
    return result.strip() == 'safe'
