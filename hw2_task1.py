from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from prompts_hw2 import SCI_PROMPT, ORCH_PROMPT, TASK_5, EXEC_PROMPT, GEN_PROMPT
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import subprocess, sys, os, re
import os
from langchain_openai import ChatOpenAI
from langchain_core import OutputParserException

from langchain_core.rate_limiters import InMemoryRateLimiter

limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,   
    check_every_n_seconds=1,
    max_bucket_size=1,
)

llm = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",         
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    rate_limiter=limiter,
    max_retries=6,
    extra_body={
        "models": [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "openrouter/free",   
        ],
    }                             
)

WORKDIR = os.path.abspath("agent_work")         
DATA_PATH = os.path.abspath("data.csv")          
os.makedirs(WORKDIR, exist_ok=True)

def write_file(filename, code):
    path = os.path.join(WORKDIR, os.path.basename(filename))   
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    return path

def run_python(path, timeout=300):                 
    try:
        r = subprocess.run([sys.executable, path], cwd=WORKDIR,
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return -1, (e.stdout or "") if isinstance(e.stdout, str) else "", "TIMEOUT: script ran longer than 5 minutes"

class Decision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "human", "finish"]
    instruction: str = Field(description="What the chosen role should do next")
    question: str = Field(default="", description="Only when next_role is human: the exact question to ask the person")
    pending_action: Literal["none", "executor", "finish"] = Field(
        default="none",
        description="Only when next_role is human for a hard decision: the step that runs if the person accepts. "
                    "'none' for an ambiguous query.")
    reason: str = Field(description="One sentence explaining the choice")

decider = llm.with_structured_output(Decision, method="function_calling").with_retry(
    retry_if_exception_type=(ValueError,),
    stop_after_attempt=3,)

class State(TypedDict):
    dialogue: Annotated[list, operator.add]
    next_role: str
    decisions: int
    rmse_ok: bool
    question: str
    pending_action: str


def dialogue_text(dialogue):
    """Turn the list of turns into one readable text for the model."""
    return "\n\n".join(f"[{t['role']}]: {t['text']}" for t in dialogue)

class Script(BaseModel):
    filename: str = Field(description="e.g. solve_task5.py")
    code: str = Field(description="The complete Python script")

coder = llm.with_structured_output(Script, method="function_calling")

def orchestrator(state):
    files = ", ".join(sorted(os.listdir(WORKDIR))) or "(none)"
    d = decider.invoke([
        ("system", ORCH_PROMPT),
        ("human", f"FILES ON DISK: {files}\n\nDIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"])),
    ])
    text = f"NEXT: {d.next_role}\nINSTRUCTION: {d.instruction}\nREASON: {d.reason}"
    if d.next_role == "human":
        text += f"\nQUESTION: {d.question}\nPENDING STEP: {d.pending_action}"
    return {"dialogue": [{"role": "orchestrator", "text": text}],
            "next_role": d.next_role,
            "question": d.question,
            "pending_action": d.pending_action,
            "decisions": state["decisions"] + 1}
ACCEPT = ("y", "yes", "ok", "accept", "proceed", "go")

def human(state) -> Command[Literal["orchestrator", "executor", "__end__"]]:
    
    reply = interrupt({"question": state["question"],
                       "pending_action": state["pending_action"]})

    turns = [{"role": "human", "text": reply}]
    pending = state["pending_action"]

    if pending == "none":                       
        return Command(update={"dialogue": turns}, goto="orchestrator")

    if reply.strip().lower().startswith(ACCEPT):  
        return Command(update={"dialogue": turns}, goto=END if pending == "finish" else pending)

    
    turns.append({"role": "system", "text": f"The step '{pending}' was refused by the human and did not run."})
    return Command(update={"dialogue": turns, "pending_action": "none"}, goto="orchestrator")

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



g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("scientist", scientist)
g.add_node("executor", executor)
g.add_node("general", general)
g.add_node("human", human)
g.add_edge(START, "orchestrator")                 # the user's message goes to the orchestrator
g.add_conditional_edges("orchestrator", route)   # orchestrator chooses
g.add_edge("scientist", "orchestrator")          # 3) EDGES: specialists hand the turn back
g.add_edge("executor", "orchestrator")
g.add_edge("general", "orchestrator")
app = g.compile(checkpointer=InMemorySaver())


def run(request, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = app.invoke({"dialogue": [{"role": "user", "text": request}],
                        "next_role": "", "decisions": 0, "rmse_ok": False,
                        "question": "", "pending_action": "none"}, config)
    while "__interrupt__" in state:              # paused: show the question, wait for your reply
        q = state["__interrupt__"][0].value
        print("\n=== THE GRAPH ASKS ===\n" + q["question"])
        reply = input("your reply > ")
        state = app.invoke(Command(resume=reply), config)   # resume the SAME thread
    return state

result = run(TASK_5 + "\n\nREQUEST: Report some metric of the model.", "run-a")

for turn in result["dialogue"]:
    print(turn["role"], ":", turn["text"])



import json
with open("dialogue.json", "w", encoding="utf-8") as fh:
    json.dump(result["dialogue"], fh, indent=2, ensure_ascii=False)