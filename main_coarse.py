import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

from agentdriver.llm_core.api_keys import (
    FINETUNE_PLANNER_NAME,
    OPENAI_API_KEY,
    OPENAI_ORG,
)
from agentdriver.llm_core.chat import run_one_round_conversation
from agentdriver.main.language_agent import LanguageAgent
from agentdriver.memory.memory_agent import MemoryAgent
from agentdriver.perception.perception_agent import PerceptionAgent
from agentdriver.planning.planning_agent import PlanningAgent
from agentdriver.reasoning.reasoning_agent import ReasoningAgent

EXPERIMENT_NAME = "example_addcoarse_exp"
RANDOM_SEED = 0
N_SAMPLES = 2000
DELETION_L2_THRESHOLD = 99999.0  # del when history l2 > this. 99999.0 = no deletion    #0.5
DELETION_RETREIVAL_N_TIMES = 99999  # del when ret time >= this. 99999 = no deletion      #3
CLEANUP_RETRIEVAL_N_TIMES = 0  # clean up when ret time < this. 0 = no cleanup            #1
EXIST_TIME_THRESHOLD = 99999  # clean up when exist time > this. 99999 = no cleanup        #500
MIN_MEMORY_SIZE = 1   # won't delete when < this
MAX_MEMORY_SIZE = 99999   # force delete (ignore l2 threshold) mem when > this (but will still add)
ADD_GT = False # WE NEVER USED THIS OPTION. TO RUN ERROR-FREE EXPERIMENT, USE `./main_noerrorprop.py`.


random.seed(RANDOM_SEED)
openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY


def convert_working_memory_to_memory(working_memory, perception_agent, planning_traj):
    """
    Convert working_memory to the format required by the memory database.

    Args:
        working_memory (dict): The working memory dictionary.
        perception_agent (PerceptionAgent): The perception agent instance.
        planning_traj (tuple): The planning trajectory output from the planning agent.

    Returns:
        dict: A dictionary in the format required by the memory database.
    """
    # Get ego_data from working_memory
    ego_data = working_memory["ego_data"]
    # Add 'objects' from perception_agent.data_dict
    ego_data["objects"] = perception_agent.data_dict.get("objects", [])
    # Add 'ego_fut_traj' from planning_traj[0]
    # Ensure that ego_hist_traj exists
    if "ego_hist_traj" in ego_data and ego_data["ego_hist_traj"].shape[0] > 0:
        last_ego_point = ego_data["ego_hist_traj"][-1]
    else:
        # If ego_hist_traj is empty
        raise ValueError("ego_hist_traj is empty")
    # Construct ego_fut_traj by adding last_ego_point to the beginning
    predicted_traj = planning_traj  # Shape (6, 2)
    ego_fut_traj = np.vstack([last_ego_point, predicted_traj])  # Shape (7, 2)
    ego_data["ego_fut_traj"] = ego_fut_traj
    # Compute ego_fut_traj_diff
    ego_fut_traj_diff = np.diff(ego_fut_traj, axis=0)  # Shape (6, 2)
    ego_data["ego_fut_traj_diff"] = ego_fut_traj_diff

    # Prepare the memory entry
    memory_entry = {"token": working_memory["token"], "ego_data": ego_data}
    return memory_entry


def eval_l2(pred_traj, gt_traj, metric="uniad"):
    assert metric in ["uniad", "stp3"], f"Invalid metric: {metric}"
    assert pred_traj.shape == gt_traj.shape, "Shape mismatch between pred and gt trajs"
    assert pred_traj.shape == (6, 2), "Invalid shape for pred_traj and gt_traj"
    l2s = np.sqrt(np.sum((pred_traj - gt_traj) ** 2, axis=1))
    assert l2s.shape == (6,), "DEBUG: Invalid shape for l2s"
    if metric == "uniad":
        avg = (l2s[1] + l2s[3] + l2s[5]) / 3
    elif metric == "stp3":
        l2s_stp3 = []
        for i in range(6):
            l2s_stp3.append(np.mean(l2s[: i + 1]))
        avg = (l2s_stp3[1] + l2s_stp3[3] + l2s_stp3[5]) / 3

    return avg


def coarse_judge(
    ego_prompts: str,
    perception_prompts: str,
    commonsense_mem: str,
    planning_target: str,
    pred_traj: np.ndarray,
    model_name: str = FINETUNE_PLANNER_NAME,
    temperature: float = 0.0,
) -> tuple[bool, str]:
    system_message = (
        "You are a highly knowledgeable and rigorous judge for autonomous driving. "
        "You are judging a *short-horizon* trajectory (e.g., 6 steps). "
        # "Hence, we do not require the vehicle to fully reach the final goal in these steps. "
        "We only require the following:\n"
        "1) The predicted trajectory should *generally* move towards or align with the goal.\n"
        "2) It should stay within a drivable area (i.e., allowed region).\n"
        "3) It should avoid collisions with other objects.\n\n"
        "Your output format:\n"
        "- First line: strictly output 'yes' or 'no'.\n"
        "- Following lines: provide your reasoning (Chain-of-Thought is allowed).\n"
        "Be mindful that small lateral or partial forward movements can be acceptable as long as "
        "the overall direction is consistent with the planning target and safety requirements. "
        "Our coordinate system is such that the x-axis is lateral, and the y-axis "
        "is forward. Therefore, moving forward means an increase in y values.\n"
        "Be mindful that minor lateral adjustments or minimal forward movements are acceptable.\n"
        "If the y coordinate is increasing from step to step (and there's no collision or out-of-lane), that may be considered a success.\n"
    )

    user_message = f"""
Below are the relevant information for this autonomous driving task:

1) Current state of the ego vehicle:
{ego_prompts}

2) Perception of the environment:
{perception_prompts}

3) Commonsense:
{commonsense_mem}

4) Planning target:
{planning_target}

5) Predicted trajectory:
{pred_traj}

Please decide if the predicted trajectory is successful under the above criteria,
then provide your reasoning (you may use chain-of-thought).
Remember:
- First line of your answer: 'yes' or 'no' ONLY.
- Following lines: your reasons or chain-of-thought.
"""

    conversation = []
    conversation, response_message = run_one_round_conversation(
        full_messages=conversation,
        system_message=system_message,
        user_message=user_message,
        temperature=temperature,
        model_name=model_name,
    )

    full_reply = response_message["content"].strip()
    lines = full_reply.split("\n")

    if not lines:
        raise ValueError("No answer returned from the Coarse Judge.")

    first_line = lines[0].lower().strip()
    if first_line.startswith("yes"):
        reason = "\n".join(lines[1:]).strip()
        return True, reason
    elif first_line.startswith("no"):
        reason = "\n".join(lines[1:]).strip()
        return False, reason
    else:
        raise ValueError(
            f"Unexpected format from the Coarse Judge. Full reply:\n{full_reply}"
        )


print(f"Experiment name: {EXPERIMENT_NAME}")
print(f"Experiment time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Number of samples: {N_SAMPLES}")
print(f"Deletion L2 threshold: {DELETION_L2_THRESHOLD}")
print(f"Deletion retrieval n times: {DELETION_RETREIVAL_N_TIMES}")
print(f"Cleanup retrieval n times: {CLEANUP_RETRIEVAL_N_TIMES}")
print(f"Exist time threshold: {EXIST_TIME_THRESHOLD}")
print(f"Minimum memory size: {MIN_MEMORY_SIZE}")
print(f"Maximum memory size: {MAX_MEMORY_SIZE}")
print(f"Add Ground-Truth: {ADD_GT}")
os.makedirs(f"./results/{EXPERIMENT_NAME}", exist_ok=True)

# Set the data path and split
data_path = Path("./data/")
split = "val"

# Load the split data
with open(data_path / "split.json", "r") as f:
    split_data = json.load(f)


##### MODIFY THE SECTION BELOW #####
# The following section will decide which data split to use.
# To run a *NORMAL* experiment, uncomment the *NORMAL* section and comment the *DISTRIBUTION SHIFT* section.
# To run a *DISTRIBUTION SHIFT* experiment, uncomment the *DISTRIBUTION SHIFT* section and comment the *NORMAL* section.

#### NORMAL
val_tokens = split_data["val"]
token_list = random.sample(val_tokens, N_SAMPLES)
assert "distribshift" not in EXPERIMENT_NAME

#### DISTRIBUTION SHIFT
# val_tokens = split_data["distribshift"]
# token_list = val_tokens
# assert token_list[0] == "64d019bf33ba4dcb9eca5e5ab2ef967e"
# assert "distribshift" in EXPERIMENT_NAME

##### END OF MODIFICATION #####


assert len(token_list) == N_SAMPLES

# For evaluation, Load the ground truth trajectory
with open("./data/metrics/gt_traj.pkl", "rb") as f:
    gt_data = pickle.load(f)

# Initialize the LanguageAgent
language_agent = LanguageAgent(
    data_path,
    split,
    model_name=FINETUNE_PLANNER_NAME,
    planner_model_name=FINETUNE_PLANNER_NAME,
    verbose=False,
)
language_agent.tokens = token_list

# Initialize the results dictionary
results = {}

# Initialize MemoryAgent, ReasoningAgent, and PlanningAgent
print("Initializing agents...")
print("Initializing MemoryAgent...")
memory_agent = MemoryAgent(
    data_path=data_path,
    model_name=language_agent.model_name,
    verbose=language_agent.verbose,
)
assert memory_agent.experience_memory.get_n_remaining_memory() == 180

## Experiment: add selected tokens to memory_agent
# There's one memory in the pool because we can't set it to zero
# So the first thing is to remove that memory
# assert memory_agent.experience_memory.get_n_remaining_memory() == 1
# only_memory_token = memory_agent.experience_memory.tokens[0]
# with open(f"./data/split.json", "r") as f:
#     cluster1_180mem = json.load(f)["sphere0_180mem"]

# for token in cluster1_180mem:
#     try:
#         perception_agent = PerceptionAgent(
#             token=token,
#             split=split,
#             data_path=data_path,
#             model_name=language_agent.model_name,
#             verbose=language_agent.verbose,
#         )
#         ego_prompts, perception_prompts, working_memory = perception_agent.run()
#         pred_traj = gt_data[token][0]
#         memory = convert_working_memory_to_memory(
#             working_memory, perception_agent, pred_traj
#         )
#         entry_time = 0
#         memory_agent.experience_memory.add_memory(memory, time=entry_time)
#         print(f"Token {token} was not retrieved before.")
#         print(f"Added memory for token {token}, entry time: {entry_time}")
#     except Exception as e:
#         print(f"An error occurred for token {token}:")
#         print(f"{type(e).__name__}: {e}")
#         continue

# _ = memory_agent.experience_memory.delete_memory(only_memory_token, time=0)

## End of experiment mem init

print("Initializing ReasoningAgent...")
reasoning_agent = ReasoningAgent(
    model_name=language_agent.model_name, verbose=language_agent.verbose
)
print("Initializing PlanningAgent...")
planning_agent = PlanningAgent(
    model_name=language_agent.planner_model_name, verbose=language_agent.verbose
)


# For each token, run the pipeline
print("Running the pipeline...")
invalid_tokens = []
success_count = 0
records = []
time_step = 0
for token in tqdm(token_list, desc="Processing tokens"):
    #### Experiment
    # if token in val_list:
    #     print(f"Skipping token {token}")
    #     continue

    sys.stdout.flush()
    # Make sure the token is in the language_agent tokens
    assert token in language_agent.tokens
    time_step += 1

    n_remaining_memories = memory_agent.experience_memory.get_n_remaining_memory()
    print(f"Remaining memories: {n_remaining_memories}")

    try:
        # Create PerceptionAgent
        perception_agent = PerceptionAgent(
            token=token,
            split=split,
            data_path=data_path,
            model_name=language_agent.model_name,
            verbose=language_agent.verbose,
        )
        # Perception
        ego_prompts, perception_prompts, working_memory = perception_agent.run()
        # Memory
        commonsense_mem, experience_mem = memory_agent.run(working_memory)
        # Reasoning
        reasoning = reasoning_agent.run(
            data_dict=perception_agent.data_dict,
            env_info_prompts=ego_prompts + perception_prompts,
            working_memory=working_memory,
            use_cot_rules=language_agent.finetune_cot,
        )
        # Planning
        planning_target = planning_agent.generate_planning_target(
            perception_agent.data_dict
        )
        data_sample = {
            "token": token,
            "ego": ego_prompts,
            "perception": perception_prompts,
            "commonsense": commonsense_mem,
            "experiences": experience_mem,
            "chain_of_thoughts": reasoning,
            "reasoning": reasoning,
            "planning_target": planning_target,
        }
        planning_traj, planning_info = planning_agent.run(
            data_dict=perception_agent.data_dict,
            data_sample=data_sample,
        )
        # Save the planning trajectory to the results dictionary
        results[token] = planning_traj  # Shape (6, 2)

        ##### Evaluation
        ground_truth = gt_data[token][0]
        # Compute L2 error
        # l2_error = np.linalg.norm(ground_truth - planning_traj, ord=2)
        l2_error = eval_l2(planning_traj, ground_truth, metric="uniad")
        # Record this retrieve. No matter if succeeded or not
        memory_token = memory_agent.experience_memory.get_retrieved_memory(token)
        memory_traj = memory_agent.experience_memory.lookup_memory_fut_traj(
            memory_token
        )
        # NEW: record whether accepted or not, not the L2, because this is coarse judge
        # memory_agent.experience_memory.record_l2(memory_token, l2_error, time_step)
        pred_mem_l2_error = eval_l2(planning_traj, memory_traj, metric="uniad")
        mem_gt_l2_error = eval_l2(memory_traj, ground_truth, metric="uniad")

        print(f"Pred vs. GT  L2: {l2_error}")
        print(f"Pred vs. mem L2: {pred_mem_l2_error}")
        print(f"Mem vs. GT  L2: {mem_gt_l2_error}")

        # Adding
        # ask a llm judge
        is_success, judge_reason = coarse_judge(
            ego_prompts=ego_prompts,
            perception_prompts=perception_prompts,
            commonsense_mem=commonsense_mem,
            planning_target=planning_target,
            pred_traj=planning_traj,
            model_name=FINETUNE_PLANNER_NAME,
            temperature=0.0,
        )
        if is_success:
            print(f"Token {token} succeeded")
            print(f"Coarse Judge reason:\n{judge_reason}")
            success_count += 1
            ####### Add the successful token to the memory database
            # if l2_error < pred_mem_l2_error:
            #     print(f"Prediction is better than demonstration")
            # NEW: record whether accepted or not, not the L2, because this is coarse judge
            memory_agent.experience_memory.record_l2(
                memory_token, L2=0.0, timestep=time_step
            )
            if True:
                # Convert working_memory to memory
                memory_entry = convert_working_memory_to_memory(
                    working_memory,
                    perception_agent,
                    # planning_traj,  # add coarse
                    ground_truth if ADD_GT else planning_traj,
                )
                is_memory_full = memory_agent.experience_memory.add_memory(
                    new_memory=memory_entry,
                    time=time_step,
                )
                print(f"Added memory {token}")
        else:
            print(f"Token {token} failed")
            print(f"Coarse Judge reason:\n{judge_reason}")
            # NEW: record whether accepted or not, not the L2, because this is coarse judge
            # Bigger L2 means worse, so 1.0 means failed
            memory_agent.experience_memory.record_l2(
                memory_token, L2=1.0, timestep=time_step
            )

        # Save the results to a pickle file
        with open(f"./results/{EXPERIMENT_NAME}/prediction_results.pkl", "wb") as f:
            pickle.dump(results, f)

        # Record the results
        input_similarity = (
            memory_agent.experience_memory.get_input_exp_similarity_of_query(
                token, data_dict=working_memory["ego_data"]
            )
        )
        output_similarity_cos = (
            memory_agent.experience_memory.get_output_cosine_similarity_of_query(
                token, planning_traj
            )
        )
        output_similarity_exp = (
            memory_agent.experience_memory.get_output_exp_similarity_of_query(
                token, planning_traj
            )
        )

        ####### Clean up dead memory
        current_memory_size = memory_agent.experience_memory.get_n_remaining_memory()
        if current_memory_size > MIN_MEMORY_SIZE:
            # dead_memory_tokens = memory_agent.experience_memory.find_dead_memory(
            #     current_time_step=time_step,
            #     exist_time_threshold=EXIST_TIME_THRESHOLD,
            #     retrieval_n_times=CLEANUP_RETRIEVAL_N_TIMES,
            # )
            dead_memory_tokens = memory_agent.experience_memory.find_dead_memory2(
                current_time_step=time_step,
                past_time_window=EXIST_TIME_THRESHOLD,
                retrieval_n_times=CLEANUP_RETRIEVAL_N_TIMES,
            )
            if current_memory_size - len(dead_memory_tokens) < MIN_MEMORY_SIZE:
                # Make sure we only clean-up to MIN_MEMORY_SIZE
                dead_memory_tokens = dead_memory_tokens[
                    : current_memory_size - MIN_MEMORY_SIZE
                ]
                # print(f"Clipped to {len(dead_memory_tokens)} dead memories")
            if len(dead_memory_tokens) > 0:
                for dead_memory_token in dead_memory_tokens:
                    _ = memory_agent.experience_memory.delete_memory(
                        dead_memory_token, time=time_step
                    )
                print(
                    f"Deleted {len(dead_memory_tokens)} dead memories: {dead_memory_tokens}"
                )
        else:
            # Do nothing
            print(f"Memory size reaches minimum size, won't clean up dead memory")

        # Do cleanup first, because there's a chance cleanup didn't clean enough
        # memory, therefore current_memory_size still greater than MAX_MEMORY_SIZE.
        # In that case, a force deletion will be triggered
        ####### Delete memory
        current_memory_size = memory_agent.experience_memory.get_n_remaining_memory()
        if current_memory_size > MAX_MEMORY_SIZE:
            # Force deletion
            print(
                f"Memory size exceeds maximum size: {current_memory_size}/{MAX_MEMORY_SIZE}"
            )
            worst_memory_token = memory_agent.experience_memory.find_worst_memory(
                l2_threshold=-1.0,
                deletion_retrieval_n_times=-1,
            )
            assert worst_memory_token is not None, "Worst memory token is None"
            worst_memory_average_l2 = memory_agent.experience_memory.delete_memory(
                worst_memory_token, time=time_step
            )
            print(
                f"Deleted worst memory: {worst_memory_token} with average L2 {worst_memory_average_l2}"
            )
        elif current_memory_size > MIN_MEMORY_SIZE:
            # Regular deletion
            worst_memory_token = memory_agent.experience_memory.find_worst_memory(
                l2_threshold=DELETION_L2_THRESHOLD,
                deletion_retrieval_n_times=DELETION_RETREIVAL_N_TIMES,
            )
            if worst_memory_token is not None:
                worst_memory_average_l2 = memory_agent.experience_memory.delete_memory(
                    worst_memory_token, time=time_step
                )
                print(
                    f"Deleted worst memory: {worst_memory_token} with average L2 {worst_memory_average_l2}"
                )
        else:
            # Do nothing
            print(f"Memory size reaches minimum size, won't delete")

        records.append(
            {
                "token": token,
                "l2_error": l2_error,
                "pred_mem_l2_error": pred_mem_l2_error,
                "mem_gt_l2_error": mem_gt_l2_error,
                "success": is_success,
                "pred_traj": planning_traj,
                "gt_traj": ground_truth,
                "memory_token": memory_token,
                "memory_traj": memory_traj,
                "input_similarity": input_similarity,
                "output_similarity_cos": output_similarity_cos,
                "output_similarity_exp": output_similarity_exp,
            }
        )

        with open(f"./results/{EXPERIMENT_NAME}/records.pkl", "wb") as f:
            pickle.dump(records, f)
        retrieval_records, retrieval_records_reverse = (
            memory_agent.experience_memory.get_retrieval_records()
        )
        with open(f"./results/{EXPERIMENT_NAME}/retrieval_records.pkl", "wb") as f:
            pickle.dump(retrieval_records, f)
        with open(
            f"./results/{EXPERIMENT_NAME}/retrieval_records_reverse.pkl", "wb"
        ) as f:
            pickle.dump(retrieval_records_reverse, f)
        success_stats = memory_agent.experience_memory.get_success_stats()
        with open(f"./results/{EXPERIMENT_NAME}/success_stats.pkl", "wb") as f:
            pickle.dump(success_stats, f)
        deletion_records = memory_agent.experience_memory.get_deletion_records()
        with open(f"./results/{EXPERIMENT_NAME}/deletion_records.pkl", "wb") as f:
            pickle.dump(deletion_records, f)
        forgot_memories = memory_agent.experience_memory.get_deletion_records()
        with open(f"./results/{EXPERIMENT_NAME}/forgotten_memories.pkl", "wb") as f:
            pickle.dump(forgot_memories, f)
        old_exist_time = memory_agent.experience_memory.get_exist_time()
        with open(f"./results/{EXPERIMENT_NAME}/exist_time.pkl", "wb") as f:
            pickle.dump(old_exist_time, f)
        retrieval_timesteps = memory_agent.experience_memory.get_retrieval_timestep()
        with open(f"./results/{EXPERIMENT_NAME}/retrieval_timesteps.pkl", "wb") as f:
            pickle.dump(retrieval_timesteps, f)

        # Save the memory database to a pickle file
        memory_agent.experience_memory.save_db(
            f"./results/{EXPERIMENT_NAME}/memory.pkl"
        )

    # except IndexError as e:
    #     print(f"\nIndexError occurred for token {token}: {e}")
    #     invalid_tokens.append(token)
    #     # raise e
    #     continue
    except Exception as e:
        print(f"An error occurred for token {token}:")
        print(f"{type(e).__name__}: {e}")
        invalid_tokens.append(token)
        # raise e
        continue

print(f"Invalid tokens: {invalid_tokens}")
print(
    f"Success count: {success_count}/{len(token_list)} ({success_count/len(token_list)*100:.2f}%)"
)
