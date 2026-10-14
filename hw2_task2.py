import os, csv, json, operator
from typing import Literal, TypedDict, Annotated

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # save plots without opening windows
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, model_validator
from langgraph.graph import StateGraph, START, END
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

import bench

# ---------------- setup (bench is created ONCE, outside the graph) ----------------
my_bench = bench.SimulatedBench()
DMD_SIDE, CAM_SIDE = my_bench.dmd_side, my_bench.cam_side
VALID_MACROS = [m for m in range(1, DMD_SIDE) if DMD_SIDE % m == 0]

RUNS_DIR = os.path.abspath("runs")
os.makedirs(RUNS_DIR, exist_ok=True)
MAX_RUNS = 8            
MAX_DECISIONS = 25      
N_REGIONS = 4           

limiter = InMemoryRateLimiter(requests_per_second=4 / 60, check_every_n_seconds=1, max_bucket_size=1)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0,
                             rate_limiter=limiter, max_retries=6)

# ---------------- prompts ----------------
TASK_FOCUS = ("Focus light through a scattering medium: find settings that give the highest "
              "enhancement (focus brightness / background brightness) at the camera center.")

SCI_PROMPT = f"""You are the scientist. You choose settings for a focusing experiment on a simulated bench.
Facts: a DMD with {DMD_SIDE}x{DMD_SIDE} pixels sends light through a scattering medium onto a
{CAM_SIDE}x{CAM_SIDE} camera. The spot to brighten is at the camera center. The camera is noisy
(about {my_bench.noise * 100:.0f}% multiplicative noise), so tiny differences between runs can be noise.
Valid macro values: {VALID_MACROS}.
Read the earlier runs in the dialogue and pick settings that should give a higher enhancement.
Change only one or two settings at a time, so we learn what helped. Never repeat a run that already happened.
If the dialogue says INVALID SETTINGS or RUN FAILED, fix that problem first."""

ORCH_PROMPT = f"""You are the orchestrator of a focusing experiment. Roles you can choose:
- scientist: picks new settings for the next run.
- executor: runs the settings the scientist just chose. Only choose it right after a valid scientist answer.
- finish: stop the experiment.
Normal cycle: scientist, then executor, then scientist, and so on.
Choose finish only when at least 4 runs are done AND the best enhancement improved by less than 5% over the
previous best during the last 2 runs. Never finish before the first run. At most {MAX_RUNS} runs are allowed.
If the scientist's last message says INVALID SETTINGS, choose scientist again."""

# ---------------- schemas ----------------
class Decision(BaseModel):
    next_role: Literal["scientist", "executor", "finish"]
    instruction: str = Field(description="What the chosen role should do next")
    reason: str = Field(description="One sentence explaining the choice")

class Inputs(BaseModel):
    macro: int = Field(
        description=f"Side of one block in DMD pixels. Allowed values: {VALID_MACROS}. "
                    f"Blocks per side = {DMD_SIDE}/macro. Smaller macro = more blocks = finer but slower.")
    focus_size: int = Field(
        ge=1, description="Side of the square focus spot at the camera center, in camera pixels.")
    bg_size: int = Field(
        ge=1, description="Side of each background square around the focus spot, in camera pixels. "
                          f"Must satisfy focus_size + 2*bg_size <= {CAM_SIDE}.")
    iterations: int = Field(
        ge=1000, le=10000,
        description="Number of trial changes. More iterations = better focus but slower.")
    n_flips: int = Field(
        ge=1, description="How many blocks are flipped in one trial. Must be smaller than "
                          "blocks per side (DMD side / macro).")

    @model_validator(mode="after")
    def check_rules(self):
        if DMD_SIDE % self.macro != 0 or self.macro >= DMD_SIDE:
            raise ValueError(f"macro must divide {DMD_SIDE} and be smaller than it")
        if self.focus_size + 2 * self.bg_size > CAM_SIDE:
            raise ValueError(f"focus_size + 2*bg_size must be <= {CAM_SIDE}")
        if self.n_flips >= DMD_SIDE // self.macro:
            raise ValueError("n_flips must be smaller than DMD side / macro")
        return self

decider = llm.with_structured_output(Decision).with_retry(
    retry_if_exception_type=(ValueError,), stop_after_attempt=3)
sci = llm.with_structured_output(Inputs)

class State(TypedDict):
    dialogue: Annotated[list, operator.add]
    next_role: str
    decisions: int

# ---------------- helpers ----------------
def dialogue_text(dialogue):
    return "\n\n".join(f"[{t['role']}]: {t['text']}" for t in dialogue)

def n_runs(dialogue):
    return sum(1 for e in dialogue if e["role"] == "executor" and "enh" in e)

def pending_settings(dialogue):
    """Settings from the scientist that have not been run yet, else None."""
    for e in reversed(dialogue):
        if e["role"] in ("scientist", "executor"):
            return e.get("inputs") if e["role"] == "scientist" else None
    return None

# ---------------- nodes ----------------
def orchestrator(state):
    d = decider.invoke([
        ("system", ORCH_PROMPT),
        ("human", f"RUNS DONE: {n_runs(state['dialogue'])} of {MAX_RUNS}\n\n"
                  "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"])),
    ])
    text = f"NEXT: {d.next_role}\nINSTRUCTION: {d.instruction}\nREASON: {d.reason}"
    return {"dialogue": [{"role": "orchestrator", "text": text}],
            "next_role": d.next_role, "decisions": state["decisions"] + 1}

def scientist(state):
    try:
        inp = sci.invoke([
            ("system", SCI_PROMPT),
            ("human", "DIALOGUE SO FAR:\n\n" + dialogue_text(state["dialogue"])),
        ])
        d = inp.model_dump()
        entry = {"role": "scientist", "text": "Settings: " + json.dumps(d), "inputs": d}
    except Exception as e:
        entry = {"role": "scientist", "text": f"INVALID SETTINGS, try again. Error: {e}"}
    return {"dialogue": [entry]}

def executor(state):
    inp = pending_settings(state["dialogue"])
    if inp is None:
        return {"dialogue": [{"role": "executor", "text": "No new valid settings to run."}]}

    run_id = n_runs(state["dialogue"]) + 1
    folder = os.path.join(RUNS_DIR, f"run_{run_id:02d}")
    os.makedirs(folder, exist_ok=True)

    try:
        best, history, best_blocks = bench.focus(
            my_bench, macro=inp["macro"], focus_size=inp["focus_size"], bg_size=inp["bg_size"],
            n_regions=N_REGIONS, iterations=inp["iterations"],
            n_flip=inp["n_flips"], seed=run_id)
    except Exception as e:
        return {"dialogue": [{"role": "executor", "text": f"RUN FAILED: {e}"}]}

    pattern = np.kron(best_blocks, np.ones((inp["macro"], inp["macro"])))
    img = my_bench.shoot(pattern)
    curve = np.maximum.accumulate(history)           # best-so-far curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history, alpha=0.4, label="trial"); ax[0].plot(curve, label="best so far")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("enhancement"); ax[0].legend()
    ax[1].imshow(img); ax[1].set_title("best camera image")
    fig.savefig(os.path.join(folder, "result.png")); plt.close(fig)
    plt.imsave(os.path.join(folder, "dmd_pattern.png"), pattern, cmap="gray")

    k = max(1, len(curve) // 10)
    gain = float(curve[-1] - curve[-k])

    csv_path = os.path.join(RUNS_DIR, "results.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["run", *inp.keys(), "seed", "enhancement", "gain_last_10pct"])
        w.writerow([run_id, *inp.values(), run_id, round(best, 3), round(gain, 3)])

    text = (f"Run {run_id}: enhancement {best:.2f}. "
            f"Gain during the last 10% of iterations: {gain:.2f}. Files in {folder}.")
    return {"dialogue": [{"role": "executor", "text": text, "enh": float(best)}]}

def route(state):
    d = state["dialogue"]
    if state["decisions"] >= MAX_DECISIONS or n_runs(d) >= MAX_RUNS:
        return END
    nxt = state["next_role"]
    if nxt == "finish":
        return END if n_runs(d) > 0 else "scientist"     # no finish before the first run
    if nxt == "executor" and pending_settings(d) is None:
        return "scientist"                               # nothing new to run
    return nxt

# ---------------- graph ----------------
g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("scientist", scientist)
g.add_node("executor", executor)
g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", route)
g.add_edge("scientist", "orchestrator")
g.add_edge("executor", "orchestrator")
app = g.compile()

# ---------------- run (prints every step live) ----------------
def show(entry):
    print(f"\n[{entry['role'].upper()}]\n{entry['text']}", flush=True)

def run(request):
    dialogue = [{"role": "user", "text": request}]
    show(dialogue[0])
    init = {"dialogue": list(dialogue), "next_role": "", "decisions": 0}
    for chunk in app.stream(init, config={"recursion_limit": 100}, stream_mode="updates"):
        for node, update in chunk.items():
            for entry in update.get("dialogue", []):
                dialogue.append(entry)
                show(entry)
    return dialogue

if __name__ == "__main__":
    dialogue = run(TASK_FOCUS)
    with open(os.path.join(RUNS_DIR, "dialogue.json"), "w", encoding="utf-8") as fh:
        json.dump(dialogue, fh, indent=2, ensure_ascii=False)
    print(f"\nDone. {n_runs(dialogue)} runs. Table: {os.path.join(RUNS_DIR, 'results.csv')}")