import subprocess, sys, os, re
WORKDIR = os.path.abspath("agent_work")          # the executor's working folder
DATA_PATH = os.path.abspath("data.csv")  

SCI_PROMPT = """ You are the scientist of the team. You are here to solve a physics problem
based on a differential equation. 
- You must understand the ODE and make a plan
for a trainable neural network which will model the output
- You must make a plan for the network for an executor.
- You cannot write code or execute anything. You own the theory and the plan
- The model needs to train in under five minutes, so make considerations of hyperparameters such
as epochs or chunk lengths """

ORCH_PROMPT = """You are the orchestrator of a team. You never write theory or code.
Roles you can call:
- scientist: writes the differential equation, the network, the training plan, numbered steps.
- general: answers high-level questions (what the data contains, what a term means).
- executor: writes and RUNS a Python script (the coding tool) and reports what it printed.
- human: the person you work for. Pausing for them costs their time, so use it only when needed.
- finish: end the dialogue and report the measurement.

Choose human in exactly two situations:
1. AMBIGUOUS QUERY: the latest request has more than one reasonable reading, and those readings
   would send the work to different roles or change the number that gets reported
   (e.g. train vs test RMSE, which model version, which settings). Ask which reading to follow,
   listing the readings. Set pending_action = "none".
2. HARD DECISION: the next step would (a) run the coding tool, (b) overwrite a file that already
   exists on disk (see FILES ON DISK), or (c) end the dialogue and report a measurement.
   State the step plainly and ask whether to proceed. Set pending_action = "executor" for (a)/(b),
   "finish" for (c). Never choose executor or finish directly.

After a human reply: follow the reading they chose. If they refused a step, do not ask for the
same step again unchanged; first change the plan or ask what they want instead.
'finish' may only be proposed after an executor turn shows a TEST RMSE printed by a script that ran."""

EXEC_PROMPT = f"""You are the executor on a team. You turn the scientist's plan into ONE complete,
self-contained Python script. Your code is saved and run automatically; you will see what it printed.
Rules:
- Load the data from this exact path: {DATA_PATH} (columns V1 = input u, V2 = output y).
- Use only numpy, pandas and torch.
- The script must finish in under 5 minutes in total (training + free run). Keep training small.
- Follow the plan and the orchestrator's latest instruction. If a previous run in the dialogue
  failed, read its error and fix it.
- At the end the script MUST print exactly these two lines:
  TRAIN RMSE: <number>
  TEST RMSE: <number>
- Never invent results. Only the script's real output counts."""

GEN_PROMPT = """You are the general agent of this team. You are responsible for answering 
any high level questions the user or the other team members have."""

TASK_5 = """Task 5. PINN (Physics Informed Neural Network) for the Silverbox
oscillator
The Silverbox is a laboratory electronic circuit built to act as a Duffing oscillator: a second-order
system with a spring that stiffens as the displacement grows. The same equation shows up for a
mass on a cubic spring, and the circuit was built so that the equation can be measured cleanly. The
public record is the Silverbox benchmark of Wigren and Schoukens (European Control Conference,
4
2013), file SNLS80mV.mat, available from https://www.nonlinearbenchmark.org/. Channel V1 is
the input voltage u(t). Channel V2 is the output voltage y(t). Both are sampled at
fs = 610.35 Hz, Δt = 1/fs.
The file contains 131,072 samples. The estimation experiment is a random-phase multisine, and it
begins at sample 40650 (indices into the file, starting at 0).
Problem.
The excerpt. Use the 4096 samples starting at index 42650, that is the slice [42650 : 46746].
This sits 2000 samples after the multisine begins. Subtract the mean of u on this excerpt and the
mean of y on this excerpt, and use those centered signals afterwards. The first 3072 samples are the
training record, about 5.03 s. The last 1024 samples are the test record, about 1.68 s. On the test
record the input u is known, because it is the voltage that was applied. The output y is what the
model has to produce.
The differential equation. Write the centered output as y and the centered input as u. The circuit
is modeled by
y¨+ a y˙ + b y + c y3 = g u(t). (1)
The four coefficients are properties of the circuit.
• a multiplies the velocity. It is the damping. On this circuit it is a few tens of s−1.
• b multiplies y. It is the linear stiffness, in s−2. The output of this excerpt is dominated by
frequencies near 70 Hz.
• c multiplies y3. It is the cubic stiffness, the term a linear spring does not have. At the peaks
of this excerpt, where |y| is near 0.2 V, the cubic force is a visible fraction of the linear force.
• g multiplies the input voltage. It is an input gain, because u and y are voltages on opposite
sides of the circuit.
They are not the weights of the network. The acceleration is produced by a network that has to
represent this force.
The network. The acceleration is a multilayer fully connected network, implemented in PyTorch.
You choose the depth, the widths, the activation, and how the inputs and the output are scaled.
With velocity v = y˙, the state is the pair (y, v) and
y˙ = v, v˙ = αθ(y, v, u). (2)
Integrate that state at step Δt, driven by the measured input. Classical Runge–Kutta of order 4
is a sound choice for the step. The excerpt lasts 6.7 s near 70 Hz, several hundred cycles, so the
integrator is what draws the waveform. Try more than one architecture, and for each architecture
tune the training hyperparameters: the optimizer, the learning rate, the number of steps, and the
length of the rollout you train on. Keep the version with the best test score under the metric below.
The metric. After the weights of an architecture are fixed, do one free run of the whole excerpt.
Start at the first sample, with the measured position and a finite-difference velocity, and integrate
through the training samples and on into the test samples. Keep the same state when the test segment
begins. On the test segment the rollout may read u, and it may not read y. The performance of the
architecture is the root-mean-square error of this free run on the 1024 test samples,
RMSE =
vuut
1
1024
1X024
i=1
􀀀ˆyi − yi
2
.
5
A smaller test RMSE is a better version. Predicting zero scores the RMS of the test output, about
0.06 V, so that number is the baseline. Also compute the same free-run RMSE on the 3072 training
samples. A network that fits the training record and then drifts on the test segment shows up as a
large gap between the two"""