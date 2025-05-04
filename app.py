import os
import gradio as gr
import requests
import pandas as pd
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    load_tool,
)

# Load OCR tool from HF Hub
ocr_tool = load_tool("microsoft/vision-ocr", trust_remote_code=True)

# Initialize LLM model from Hugging Face Inference API
model = InferenceClientModel(model_id="Qwen/Qwen3-30B-A3B")

# Compose agent with tools
agent_core = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        WikipediaSearchTool(),
        ocr_tool,
    ],
    model=model,
    max_steps=10,
    verbosity_level=1,
    add_base_tools=True,
)

# Correct system prompt to match GAIA format

agent_core.prompt_templates["system_prompt"] += """

--- GAIA FORMAT ---
After your reasoning and code steps, finish with:
FINAL ANSWER: [YOUR ANSWER]

YOUR ANSWER should be a number or as few words as possible or a comma-separated list.
No commas in numbers, no units, no articles/abbreviations in strings.
"""

# BasicAgent wrapper for Gradio
class BasicAgent:
    def __init__(self):
        print("SmolAgent initialized.")

    def __call__(self, question: str) -> str:
        raw = agent_core.run(question)
        # Extract only the final answer part
        return raw.split("FINAL ANSWER:")[-1].strip()

# --- Submission Logic ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    if profile:
        username = profile.username
    else:
        return "Please Login to Hugging Face with the button.", None

    api_url = "https://agents-course-unit4-scoring.hf.space"
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    try:
        resp = requests.get(questions_url, timeout=15)
        resp.raise_for_status()
        questions = resp.json()
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log, answers_payload = [], []
    for item in questions:
        tid = item.get("task_id")
        q = item.get("question")
        if not tid or q is None:
            continue
        try:
            ans = agent(q)
            answers_payload.append({"task_id": tid, "submitted_answer": ans})
            results_log.append({"Task ID": tid, "Question": q, "Submitted Answer": ans})
        except Exception as e:
            results_log.append({"Task ID": tid, "Question": q, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        return "Agent did not produce any answers.", pd.DataFrame(results_log)

    data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    try:
        resp = requests.post(submit_url, json=data, timeout=60)
        resp.raise_for_status()
        r = resp.json()
        status = (
            f"Submission Successful!\n"
            f"User: {r.get('username')}\n"
            f"Score: {r.get('score')}% ({r.get('correct_count')}/{r.get('total_attempted')})\n"
            f"Message: {r.get('message')}"
        )
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)

    return status, pd.DataFrame(results_log)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SmolAgent GAIA Level-1 Evaluation")
    gr.Markdown("""
- log in to HF
- click Run Evaluation & Submit All Answers
""")
    gr.LoginButton()
    run_btn = gr.Button("Run Evaluation & Submit All Answers")
    status_out = gr.Textbox(label="Run Status", lines=5, interactive=False)
    results_tb = gr.DataFrame(label="Results")
    run_btn.click(fn=run_and_submit_all, outputs=[status_out, results_tb])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
