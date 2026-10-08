# Notes for CODEX

- Write implementations as simple as possible, reusing exsisting functionality and avoiding overly complicated structures.
- The actual project is in "alphaproof/".
- "pseudocode.py" is not part of the actual project. It is used as a guide for how to structure the code. The code in "alphaproof/" should closely follow the structure of "pseudocode.py".
- When you want to use py_compile, dont place the cache files in this repo.
- Always place imports at the top of files. Not inside functions.
- Backward compatability doesnt matter, it just complicates the code unnecessarily.
- Dont make fallbacks. Its better to assume that the input to a function always is in the correct format, and just let the code fail if thats not the case.
- Write code as simple as possible, without complicated structures. If you can remove more code than you write as you complete a task, thats great.
- Do not pass a function to another function as an argument.
- When possible functions should only depend on what they take as input.
- Avoid nested functions.
- After writing code fix any potential Pylance errors.
- If CLI Pyright misses VS Code Pylance errors, inspect the VS Code Python Language Server log for interpreter and editable-install resolution issues.
- Scripts that should only be executed once, like downloading a dataset, should be in the scripts folder.
- Dont write tests. Its not necessary for this project.
- LeanTree is used as a submodule. Dont edit this, just see it as a library.
- I want the main AlphaProof algorithm files (like train.py and actors.py) to be as minimal as possible, focusing on just implementing the core algorithm. Less important code like logging should be placed in separate files.
- Parameters that are in config.py (like batch_size, lr, etc.) should not be set as defaults in any other places. config.py should be the single source of truth for these parameters.
- Dont edit the README.md, unless I explicitly ask you to.
- When I ask a question give me a concise answer that gets straight to the point.


# Design choices

These are the design choices we have made so far.
They might differ from the pseudocde, which is ok.

- Used LeanTree for interacting with Lean 4.
- SFT dataset (state, action, proof_depth) was generated in a similar fashion as in NanoProof.
- SFT code might be similar to NanoProof SFT code.
- Replay buffer samples uniformly.
- Computes tactic prior by summing token logprobs. This is used as the prior in PUCT.
- Value head uses mean pooled encoder output.
- Value head is currently linear layer.
- Data for each run is stored like this, so runs can be resumed:
  runs/
    0/
    config.json
    matchmaker_stats.json
    results.jsonl
    replay_buffer.jsonl
    checkpoints/
      step_0001000.pt
      step_0002000.pt
      step_0003000.pt
- Actors are run in parallel, for a specific amount of rollouts each.
- Encoder called again every time a node is expanded.
- The autoformalizer only generates one lean problem per natural language problem.
- Models used:
  - Data cleaning: Qwen3.6-27B
  - Autoformalization: Goedel-Prover-V2-32B
  - Prover: Salesforce--codet5p-770m
- Datasets:
  - SFT: 1/4 random subset of NanoProof's SFT dataset
  - RL: 1/3 random subset of NanoProof's RL dataset
- Only n (usually 32) lean processes are started concurrently to limit congestion
- yaml files with both SFT and RL hyperparameters are used for reproducibility
- There are three repos for this project all in the GitHub parent folder: delta-proof with the AlphaProof implementation, alpha-STP with the STP implementation and thesis-latex with the thesis report written in Latex.