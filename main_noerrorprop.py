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
from agentdriver.main.language_agent import LanguageAgent
from agentdriver.memory.memory_agent import MemoryAgent
from agentdriver.perception.perception_agent import PerceptionAgent
from agentdriver.planning.planning_agent import PlanningAgent
from agentdriver.reasoning.reasoning_agent import ReasoningAgent

EXPERIMENT_NAME = "example_noerrorprop"
RANDOM_SEED = 0
MIN_MEMORY_SIZE = 1
MAX_MEMORY_SIZE = 99999
REFERENCE_RESULT_FILE = "./results/example_addstrict_exp/records.pkl"


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


print(f"Experiment name: {EXPERIMENT_NAME}")
print(f"Experiment time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Minimum memory size: {MIN_MEMORY_SIZE}")
print(f"Maximum memory size: {MAX_MEMORY_SIZE}")
print(f"FINETUNE_PLANNER_NAME: {FINETUNE_PLANNER_NAME}")
print(f"REFERENCE_RESULT_FILE: {REFERENCE_RESULT_FILE}")
os.makedirs(f"./results/{EXPERIMENT_NAME}", exist_ok=True)

# Set the data path and split
data_path = Path("./data/")
split = "val"

# Load the split data
with open(REFERENCE_RESULT_FILE, "rb") as f:
    record_db = pickle.load(f)
token_list = []
retrieve_records = {}
accepted_records = {}
for record in record_db:
    token = record["token"]
    memory_token = record["memory_token"]
    accepted = record["success"]
    token_list.append(token)
    retrieve_records[token] = memory_token
    accepted_records[token] = accepted
print(f"Number of tokens in test: {len(token_list)}")

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
        commonsense_mem, experience_mem = memory_agent.run_with_specified_retrieval(
            working_memory, retrieve_records[token]
        )
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
        memory_agent.experience_memory.record_l2(memory_token, l2_error, time_step)
        pred_mem_l2_error = eval_l2(planning_traj, memory_traj, metric="uniad")
        mem_gt_l2_error = eval_l2(memory_traj, ground_truth, metric="uniad")

        print(f"Pred vs. GT  L2: {l2_error}")
        print(f"Pred vs. mem L2: {pred_mem_l2_error}")
        print(f"Mem  vs. GT  L2: {mem_gt_l2_error}")

        # Adding
        if accepted_records[token]:
            print(f"Token {token} accepted")
            success_count += 1
        else:
            print(f"Token {token} rejected")
        # Add anyway, since we need to make sure it can be retrieved
        # Convert working_memory to memory
        memory_entry = convert_working_memory_to_memory(
            working_memory,
            perception_agent,
            ground_truth,  # This exp is to test if error propagation can be stopped by retrieving GT
        )
        is_memory_full = memory_agent.experience_memory.add_memory(
            new_memory=memory_entry,
            time=time_step,
        )
        print(f"Added memory {token}")

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

        records.append(
            {
                "token": token,
                "l2_error": l2_error,
                "pred_mem_l2_error": pred_mem_l2_error,
                "mem_gt_l2_error": mem_gt_l2_error,
                "success": accepted_records[token],
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
