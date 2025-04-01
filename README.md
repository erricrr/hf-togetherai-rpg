# RPG Text-Based Fantasy Game

A text-based fantasy role-playing game (RPG) powered by Together AI and Hugging Face AI interactions.

## Setup and Install

**Fork** this repository and **clone** it in your virtual environment.
Ensure you're in the repo folder, then install the requirements:
```bash
pip install -r requirements.txt
```
## Set Up API Keys
- Create a `.env` file in the root directory
- Add your API keys:
```
TOGETHER_API_KEY=your_together_api_key
HF_TOKEN=your_huggingface_token
```

## Start Gradio UI
```bash
python ./app.py
```
## Credits

This project is based on the **[Building an AI-Powered Game](https://www.deeplearning.ai/short-courses/building-an-ai-powered-game/)** short course by DeepLearning.AI.  

I have modified the original implementation by replacing Together AI with Hugging Face for inference while keeping Together AI for LlamaGuard.
