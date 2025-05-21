# Agent Memory Management

This repository contains the source code (for experiments related to [AgentDriver](https://github.com/USC-GVL/Agent-Driver) only) used in our paper **How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior**.

## Usage

AgentDriver code does not come together in this repository. Instead, we provide only the necessary files to patch the original AgentDriver code for simplicity.

Steps:

1. Clone this repository, then `cd` into it.
2. Clone the original AgentDriver code. This step can be done at any location on your machine:

```bash
git clone https://github.com/USC-GVL/Agent-Driver.git
```

3. Move `agentdriver`, `assets`, `data`, `experiments`, `scripts` folders, and `requirements.txt` from inside the original Agent-Driver project to this project, **but not the `Agent-Driver` folder itself**. You should be able to see `agentdriver`, `assets`, `data`, `experiments`, `scripts` folders beside `agentdriver_patch`, `scripts_patch` and `data_patch` folders in this project now.

4. Follow the *Installation* and *Data Preparation* guide on https://github.com/USC-GVL/Agent-Driver, and get all the data prepared.
5. **Patch the original code with our code**: Move everything inside `agentdriver_patch` folder to `agentdriver` folder. Move everything inside `scripts_patch` folder to `scripts` folder. Move everything inside `data_patch` folder to `data` folder. Overwrite all existing files. Now you can safely delete `agentdriver_patch`, `scripts_patch` and `data_patch` folder.

> The filename typo `planning_prmopts` is deliberate to ensure capability with the original code. DO NOT FIX IT.

6. Provide your API keys in `./agentdriver/llm_core/api_keys.py`. Refer to the example file.

> `OPENAI_ORG` is not required and can be empty.

7. Now you are good to go. Run an experiment by the following line. Hyperparameters are provided inside the code:

```bash
# For example, the *strict addition* with any *deletion* in main experiment
python main_strict.py
```

8. To run *distribution shift* experiments, remember to modify the data split loading section in each code file (see `main_strict.py:120` for detail).

9. To evaluate the results, run the following line:

```bash
# For example, if the experiment name (the result folder name) is `example_addstrict_exp`
bash ./scripts/run_evaluation.sh uniad 2.5 results/example_addstrict_exp/prediction_results.pkl
```

10. To export the results from `.pkl` to `.json` for further processing, use `./result_extract.ipynb`. Modify the code inside it accordingly (see the content for detail).

## Code Files

- `main_strict.py`: Main experiment, strict addition, any deletion
- `main_coarse.py`: Main experiment, coarse addition, any deletion
- `main_strict_noisymemory.py`: Memory with noise applied, strict addition, any deletion
- `main_coarse_noisymemory.py`: Memory with noise applied, coarse addition, any deletion
- `main_noerrorprop.py`: Error-free variants of the main experiments. Make sure you have already obtained the results from the corresponding error-not-free main experiment, because the result file from that experiment is needed in this one.
