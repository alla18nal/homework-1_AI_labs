from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from task_5.prompts import SCI_PROMPT, ORCH_PROMPT, TASK_5, EXEC_PROMPT, GEN_PROMPT

import subprocess, sys, os, re


from langchain_core.rate_limiters import InMemoryRateLimiter

limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,   # 4 calls per minute, safely under the limit of 5
    check_every_n_seconds=1,
    max_bucket_size=1,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    rate_limiter=limiter,
    max_retries=6,                # if a 429 still slips through, wait and try again
)

WORKDIR = os.path.abspath("agent_work")          # the executor's working folder
DATA_PATH = os.path.abspath("data.csv")          # absolute path, so the script finds it from anywhere
os.makedirs(WORKDIR, exist_ok=True)

def write_file(filename, code):
    path = os.path.join(WORKDIR, os.path.basename(filename))   # basename: can't write outside the folder
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    return path

def run_python(path, timeout=300):                 # 300 s = the 5-minute limit
    try:
        r = subprocess.run([sys.executable, path], cwd=WORKDIR,
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return -1, (e.stdout or "") if isinstance(e.stdout, str) else "", "TIMEOUT: script ran longer than 5 minutes"

# 1) STATE: the shared notebook
class State(TypedDict):
    dialogue: Annotated[list, operator.add]   # "operator.add" = new turns get APPENDED
    next_role: str                            # the orchestrator's latest decision
    decisions: int
    rmse_ok: bool




class Decision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(description="What that role should do next")
    reason: str = Field(description="One sentence explaining the choice")

decider = llm.with_structured_output(Decision)

def dialogue_text(dialogue):
    """Turn the list of turns into one readable text for the model."""
    return "\n\n".join(f"[{t['role']}]: {t['text']}" for t in dialogue)

class Script(BaseModel):
    filename: str = Field(description="e.g. solve_task5.py")
    code: str = Field(description="The complete Python script")

coder = llm.with_structured_output(Script)

def orchestrator(state):
    d = decider.invoke([
        ("system", ORCH_PROMPT),
        ("human", "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"])),
    ])
    text = f"NEXT: {d.next_role}\nINSTRUCTION: {d.instruction}\nREASON: {d.reason}"
    return {"dialogue": [{"role": "orchestrator", "text": text}],
            "next_role": d.next_role,
            "decisions": state["decisions"] + 1}


def scientist(state):
        d = llm.invoke([
        ("system", SCI_PROMPT),
        ("human", "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"])),
        ])
        text = d.content
        return {"dialogue": [{"role": "scientist", "text": text}]}
def tail(s, n=2000):
    return s if len(s) <= n else "...(cut)...\n" + s[-n:]

def executor(state):
    s = coder.invoke([
        ("system", EXEC_PROMPT),
        ("human", "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"])),
    ])
    path = write_file(s.filename, s.code)
    code, out, err = run_python(path)

    m = re.search(r"TEST RMSE:\s*([0-9.eE+-]+)", out)
    ok = (code == 0) and (m is not None)

    text = (f"Wrote {os.path.basename(path)} and ran it.\n"
            f"EXIT CODE: {code}\n"
            f"STDOUT:\n{tail(out)}\n"
            f"STDERR:\n{tail(err)}")
    return {"dialogue": [{"role": "executor", "text": text}],
            "rmse_ok": ok}

def general(state):
    d = llm.invoke([
                ("system", GEN_PROMPT),
                ("human", "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"])),
                ])
    text = d.content
    return {"dialogue": [{"role": "general", "text": "I am here if u need anything!"}]}
def route(state):
    if state["next_role"] == "finish" or state["decisions"] >= 8:
        return END
    return state["next_role"]
# Build the graph
g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("scientist", scientist)
g.add_node("executor", executor)
g.add_node("general", general)
g.add_edge(START, "orchestrator")                 # the user's message goes to the orchestrator
g.add_conditional_edges("orchestrator", route)   # orchestrator chooses
g.add_edge("scientist", "orchestrator")          # 3) EDGES: specialists hand the turn back
g.add_edge("executor", "orchestrator")
g.add_edge("general", "orchestrator")
app = g.compile()


# Run it
result = app.invoke({"dialogue": [{"role": "user", "text": f"You must solve {TASK_5}"}],
                     "next_role": "", "decisions": 0, "rmse_ok": False})
for turn in result["dialogue"]:
    print(turn["role"], ":", turn["text"])



import json
with open("dialogue.json", "w", encoding="utf-8") as fh:
    json.dump(result["dialogue"], fh, indent=2, ensure_ascii=False)