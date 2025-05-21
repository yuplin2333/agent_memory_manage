# Maintain a long-term memory to retrieve historical driving experiences.
# Written by Jiageng Mao & Junjie Ye
import json
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from agentdriver.functional_tools.detection import (
    get_all_object_detections,
    get_front_object_detections,
    get_leading_object_detection,
    get_object_detections_in_range,
    get_surrounding_object_detections,
)
from agentdriver.functional_tools.ego_state import extract_ego_inputs
from agentdriver.functional_tools.prediction import (
    get_all_future_trajectories,
    get_future_trajectories_for_specific_objects,
    get_future_trajectories_in_range,
    get_future_waypoint_of_specific_objects_at_timestep,
    get_leading_object_future_trajectory,
)
from agentdriver.llm_core.chat import run_one_round_conversation
from agentdriver.memory.memory_prompts import memory_system_prompt


class ExperienceMemory:
    r"""Memory of Past Driving Experiences."""

    def __init__(
        self,
        data_path,
        model_name="",
        # model_name="gpt-3.5-turbo-0125",
        verbose=False,
        compare_perception=False,
    ) -> None:
        self.data_path = data_path / Path("memory") / Path("database.pkl")
        self.num_keys = 3
        self.keys = []
        self.values = []
        self.tokens = []
        self.key_coefs = [1.0, 10.0, 1.0]
        self.k = 1  # Just make sure this is 1 because I skipped gpt retrieval
        self.model_name = model_name
        self.verbose = verbose
        self.compare_perception = compare_perception

        self.training_set_json_path = (
            data_path / Path("finetune_original") / Path("data_samples_train.json")
        )
        # Memory pool capacity
        self.max_capacity = 99999
        # Initial number of memories existing in the memory pool
        self.n_init_memories = 180  # Minimum 1, otherwise code will break
        # Deletion will prioritize those retrieved more than n times to avoid deleting too new memories
        # self.deletion_retrieval_n_times = 3
        # memory token -> list of end-to-end L2 of each retrieval
        self.success_stats = {}
        # query token -> memory token. 1-to-1 mapping, won't delete memory
        self.retrieval_records = {}
        # memory token -> query token. 1-to-many mapping, won't delete memory
        self.retrieval_records_reverse = {}
        # list of {"token", "key", "ego_fut_traj", "L2"}
        self.deleted_records = []
        # entry and exit time of the memory. token -> {entry_time, exit_time}
        self.exist_time = {}
        # clean up patch. Record the timesteps when a retrieval happens
        # used in self.find_dead_memory2()
        self.retrieval_timestep = {}

        self.load_db()

    def gen_vector_keys(self, data_dict):
        vx = data_dict["ego_states"][0] * 0.5
        vy = data_dict["ego_states"][1] * 0.5
        v_yaw = data_dict["ego_states"][4]
        ax = (
            data_dict["ego_hist_traj_diff"][-1, 0]
            - data_dict["ego_hist_traj_diff"][-2, 0]
        )
        ay = (
            data_dict["ego_hist_traj_diff"][-1, 1]
            - data_dict["ego_hist_traj_diff"][-2, 1]
        )
        cx = data_dict["ego_states"][2]
        cy = data_dict["ego_states"][3]
        vhead = data_dict["ego_states"][7] * 0.5
        steeling = data_dict["ego_states"][8]

        return [
            np.array([vx, vy, v_yaw, ax, ay, cx, cy, vhead, steeling]),
            data_dict["goal"],
            data_dict["ego_hist_traj"].flatten(),
        ]

    def load_db(self):
        r"""Load the memory from a file."""
        data = pickle.load(open(self.data_path, "rb"))

        temp_keys = []
        for token in data:
            # Count
            if self.n_init_memories <= 0:
                break
            self.n_init_memories -= 1

            key_arrays = self.gen_vector_keys(data[token])
            if temp_keys == []:
                temp_keys = [[] for _ in range(len(key_arrays))]
            for i, key_array in enumerate(key_arrays):
                temp_keys[i].append(key_array)
            temp_value = data[token].copy()
            temp_value.update({"token": token})
            self.values.append(temp_value)
            self.tokens.append(token)

            # Records
            self.success_stats[token] = []
            self.retrieval_timestep[token] = []
            self.exist_time[token] = {"entry_time": 0, "exit_time": -1}
        for temp_key in temp_keys:
            temp_key = np.stack(temp_key, axis=0)
            self.keys.append(temp_key)

    def compute_input_cosine_similarity(self, query1, query2, epsilon=1e-2):
        vec1 = query1.flatten()
        vec2 = query2.flatten()
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < epsilon or norm2 < epsilon:
            if norm1 < epsilon and norm2 < epsilon:
                return 1.0
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def compute_input_exp_similarity(self, query1, query2):
        # query is a list of 3 np arrays. To computing exp similarity, we will
        # apply self.key_coefs to them, like the original similarity function.
        diffs = []
        for q1, q2, key_coef in zip(query1, query2, self.key_coefs):
            # will do this for each part of query, total 3 times
            squared_diff = np.sum((q1 - q2) ** 2)
            diffs.append(squared_diff * key_coef)
        diffs = sum(diffs)
        similarity = np.exp(-diffs)
        return similarity

    def compute_output_exp_similarity(self, traj1, traj2, gamma=1.0):
        vec1 = traj1.flatten()
        vec2 = traj2.flatten()
        squared_diff = np.sum((vec1 - vec2) ** 2)
        similarity = np.exp(-gamma * squared_diff)
        return similarity

    def compute_output_cosine_similarity(self, traj1, traj2, epsilon=1e-2):
        traj1 = traj1.flatten()
        traj2 = traj2.flatten()
        norm1 = np.linalg.norm(traj1)
        norm2 = np.linalg.norm(traj2)
        if norm1 < epsilon or norm2 < epsilon:
            if norm1 < epsilon and norm2 < epsilon:
                return 1.0
            return 0.0
        return np.dot(traj1, traj2) / (norm1 * norm2)

    def compute_similarity(self, queries, token):
        """Compute the similarity between the current experience and the past experiences in the memory."""
        diffs = []
        for query, key, key_coef in zip(queries, self.keys, self.key_coefs):
            squared_diffs = np.sum((query - key) ** 2, axis=1)
            diffs.append(squared_diffs * key_coef)
        diffs = sum(diffs)

        confidence = np.exp(-diffs)

        if token in self.tokens:
            self_index = self.tokens.index(token)
            confidence[self_index] = 0.0

        sorted_indices = np.argsort(-confidence, kind="mergesort")

        top_k_indices = sorted_indices[: self.k]

        return top_k_indices, confidence[top_k_indices]

    def vector_retrieve(self, working_memory):
        """Step-1 Vectorized Retrieval"""
        querys = self.gen_vector_keys(working_memory["ego_data"])
        top_k_indices, confidence = self.compute_similarity(
            querys, working_memory["token"]
        )

        retrieved_scenes = [self.values[i] for i in top_k_indices].copy()
        retrieved_tokens = [self.tokens[i] for i in top_k_indices].copy()
        return retrieved_tokens, retrieved_scenes, confidence


    def gpt_retrieve(
        self, working_memory, retrieved_tokens, retrieved_scenes, confidence
    ):
        # MY EDIT: Skip the gpt retrieval, randomly select one from vector retrieval results. Since k=1, this is the one.
        ret_index = random.randint(0, len(retrieved_scenes) - 1)

        retrieved_fut_traj = retrieved_scenes[ret_index]["ego_fut_traj"]

        # Manually build the prompt as if it's from GPT
        retrieved_mem_prompt = (
            "*" * 5 + "Past Driving Experience for Reference:" + "*" * 5 + "\n"
        )
        retrieved_mem_prompt += f"Most similar driving experience from memory with confidence score: {confidence[ret_index]:.2f}:\n"
        # retrieved_mem_prompt += retrieve_ego_prompts[ret_index]
        retrieved_mem_prompt += (
            f"The planned trajectory in this experience for your reference:\n"
        )
        fut_waypoints = [
            f"({point[0]:.2f},{point[1]:.2f})" for point in retrieved_fut_traj[1:]
        ]
        traj_prompts = "[" + ", ".join(fut_waypoints) + "]\n"
        retrieved_mem_prompt += traj_prompts

        # Record the retrieval
        self.record_retrieval(working_memory["token"], retrieved_tokens[ret_index])
        return retrieved_mem_prompt

    def retrieve(self, working_memory):
        r"""Retrieve the most similar past driving experiences with current working memory as input."""

        retrieved_tokens, retrieved_scenes, confidence = self.vector_retrieve(
            working_memory
        )
        # this gpt retrieval is skipped by k=1 and directly return the prompt
        retrieved_mem_prompt = self.gpt_retrieve(
            working_memory, retrieved_tokens, retrieved_scenes, confidence
        )

        return retrieved_mem_prompt

    def retrieve_a_specific_memory(self, working_memory, query_token):
        """Retrieve a specific memory given the query token."""
        assert query_token in self.tokens, f"query_token {query_token} not in memory"
        # fake vector retrieve
        retrieved_tokens = [query_token]
        retrieved_scenes = [self.values[self.tokens.index(query_token)]]
        confidence = [1.0]
        # fake gpt retrieve
        retrieved_mem_prompt = self.gpt_retrieve(
            working_memory, retrieved_tokens, retrieved_scenes, confidence
        )

        return retrieved_mem_prompt

    def get_retrieved_memory(self, query_token):
        assert (
            query_token in self.retrieval_records
        ), f"query_token: {query_token} not in retrieval_records"
        return self.retrieval_records[query_token]

    def record_retrieval(self, query_token, memory_token):
        assert (
            query_token not in self.retrieval_records
        ), f"query_token: {query_token} already in retrieval_records"
        self.retrieval_records[query_token] = memory_token
        if memory_token not in self.retrieval_records_reverse:
            self.retrieval_records_reverse[memory_token] = []
        self.retrieval_records_reverse[memory_token].append(query_token)
        print(f"RETRIEVE: {query_token} -> {memory_token}")

    def record_l2(self, memory_token, L2, timestep):
        assert (
            memory_token in self.success_stats
        ), f"memory_token: {memory_token} not in success_stats"
        self.success_stats[memory_token].append(L2)
        self.retrieval_timestep[memory_token].append(timestep)

    def find_worst_memory(
        self,
        l2_threshold: float,
        deletion_retrieval_n_times: int,
    ):
        r"""Find the worst memory to delete.

        Policy (priority from high to low):
            1. Retrieved >= `deletion_retrieval_n_times` times, with the highest average L2 that is > `l2_threshold`.

        This is the only policy to follow. If no memory meets the policy, return None.

        Args:
            l2_threshold (float): The threshold of the average L2 to delete a memory.
            deletion_retrieval_n_times (int): The threshold of the number of retrievals to delete a memory.

        Returns:
            worst_memory_token (str): The token of the worst memory to delete. If no memory meets the policy, return None.
        """
        NOT_RETREIVED_L2 = -1
        average_L2s = {
            token: np.mean(L2s) if len(L2s) > 0 else NOT_RETREIVED_L2
            for token, L2s in self.success_stats.items()
        }

        policy_1_list = [
            token
            for token, average_L2 in average_L2s.items()
            if len(self.success_stats[token]) >= deletion_retrieval_n_times
            and average_L2 > l2_threshold
        ]
        if len(policy_1_list) == 0:
            return None
        policy_1_list = sorted(
            policy_1_list, key=lambda token: average_L2s[token], reverse=True
        )
        worst_memory_token = policy_1_list[0]

        # report how many zeros in average_L2
        num_zeros = sum(
            [1 for value in average_L2s.values() if value == NOT_RETREIVED_L2]
        )
        print(f"Number of zero L2s: {num_zeros}/{len(average_L2s)}")
        return worst_memory_token

    def add_memory(self, new_memory, time):
        r"""Add a new memory to the memory database.

        Return True if the memory pool is full, otherwise False.
        """
        token = new_memory["token"]
        if token in self.tokens:
            raise ValueError(f"Memory with token {token} already exists.")

        query_embedding = self.gen_vector_keys(new_memory["ego_data"])
        self.tokens.append(token)
        self.values.append(new_memory["ego_data"])
        self.keys[0] = np.vstack([self.keys[0], query_embedding[0]])
        self.keys[1] = np.vstack([self.keys[1], query_embedding[1]])
        self.keys[2] = np.vstack([self.keys[2], query_embedding[2]])
        self.success_stats[token] = []
        self.retrieval_timestep[token] = []
        print(
            f"ADD: Added {token}. Memory pool capacity {len(self.tokens)}/{self.max_capacity}"
        )
        self.update_entry_time(token, time)

        if len(self.tokens) >= self.max_capacity:
            # Auto trigger a memory deletion
            print(f"ADD: Memory pool is full.")
            return True
        return False

    def delete_memory(self, token, time):
        r"""Delete a memory from the memory database.

        Return the average L2 of the deleted memory.
        """
        if token not in self.tokens:
            raise ValueError(f"Memory with token {token} does not exist.")

        index = self.tokens.index(token)
        deleted_key = [
            self.keys[0][index].copy(),
            self.keys[1][index].copy(),
            self.keys[2][index].copy(),
        ]

        self.deleted_records.append(
            {
                "token": token,
                "key": deleted_key,
                "ego_fut_traj": self.values[index]["ego_fut_traj"].copy(),
                "L2": self.success_stats[token].copy(),
            }
        )

        index = self.tokens.index(token)
        self.tokens.pop(index)
        self.keys[0] = np.delete(self.keys[0], index, axis=0)
        self.keys[1] = np.delete(self.keys[1], index, axis=0)
        self.keys[2] = np.delete(self.keys[2], index, axis=0)
        self.values.pop(index)
        l2_mean = (
            np.mean(self.success_stats[token])
            if len(self.success_stats[token]) > 0
            else "N/A"
        )
        print(
            f"DELETE: Deleted {token}, \n\tL2: {self.success_stats[token]}, \n\t Retrieved {len(self.success_stats[token])} times, L2 average {l2_mean}"
        )
        # Must delete success_stats, otherwise find_worst_memory will still find it
        del self.success_stats[token]
        # del self.memory_prompts[token]
        self.update_exit_time(token, time)

        return l2_mean if l2_mean != "N/A" else None

    def find_dead_memory(
        self, current_time_step: int, exist_time_threshold: int, retrieval_n_times: int
    ):
        r"""Find the dead memories that have existed for too long and not retrieved much for further clean up.

        Args:
            current_time_step (int): The current time step.
            exist_time_threshold (int): The threshold of the exist time to delete a memory.
            retrieval_n_times (int): The threshold of the number of retrievals to delete a memory. If a memory is retrieved less than (<) this number, it is considered dead.

        Returns:
            dead_memory_tokens (list): The tokens of the dead memories. If no memory meets the policy, return an empty list.
        """
        raise NotImplementedError("find_dead_memory should never be used in our experiment setting! Use find_dead_memory2 instead.")
        # self.exist_time contains already deleted memories, so we won't use it as reference
        # We find exist memory from self.success_stats
        dead_memory_tokens = [
            token
            for token in self.success_stats.keys()
            if current_time_step - self.exist_time[token]["entry_time"]
            >= exist_time_threshold
            and len(self.success_stats[token]) < retrieval_n_times
        ]
        # print(f"CLEANUP: Found {len(dead_memory_tokens)} dead memories: {dead_memory_tokens}")

        return dead_memory_tokens

    def find_dead_memory2(
        self, current_time_step: int, past_time_window: int, retrieval_n_times: int
    ):
        r"""Find the dead memories that have existed for too long and not retrieved much for further clean up.

        Args:
            current_time_step (int): The current time step.
            past_time_window (int): The time window to look back.
            retrieval_n_times (int): The threshold of the number of retrievals to delete a memory. If a memory is retrieved less than (<) this number, it is considered dead.

        Returns:
            dead_memory_tokens (list): The tokens of the dead memories. If no memory meets the policy, return an empty list.
        """
        dead_memory_tokens = []
        for token in self.success_stats.keys():
            enter_time = self.exist_time[token]["entry_time"]
            if current_time_step - enter_time >= past_time_window:
                time_window_start = current_time_step - past_time_window
                retrieval_times_in_window = [
                    retrieval_time
                    for retrieval_time in self.retrieval_timestep[token]
                    if retrieval_time >= time_window_start
                ]
                if len(retrieval_times_in_window) < retrieval_n_times:
                    dead_memory_tokens.append(token)
        # print(f"CLEANUP: Found {len(dead_memory_tokens)} dead memories: {dead_memory_tokens}")

        return dead_memory_tokens

    def find_lru_memory(self, current_time_step: int):
        r"""Find the least recently used memory.

        Args:
            current_time_step (int): The current time step.

        Returns:
            lru_memory_token (str): The token of the least recently used memory.
        """
        # Find the least recently retrieved memory
        def get_last_retrieval_time(token):
            retrieval_times = self.retrieval_timestep[token]
            if not retrieval_times:  # If never retrieved
                return self.exist_time[token]["entry_time"]  # Use entry time as last retrieval time
            return retrieval_times[-1]

        lru_memory_token = max(
            self.success_stats.keys(),
            key=lambda token: current_time_step - get_last_retrieval_time(token),
        )
        return lru_memory_token

    def find_oldest_memory(self):
        r"""Find the oldest memory in the memory database.

        Returns:
            oldest_memory_token (str): The token of the oldest memory.
        """
        # Find the memory with the earliest entry time
        oldest_memory_token = min(
            self.success_stats.keys(),
            key=lambda token: self.exist_time[token]["entry_time"],
        )
        return oldest_memory_token

    def save_db(self, filename):
        r"""Save the memory to a file."""
        data = {token: value for token, value in zip(self.tokens, self.values)}
        pickle.dump(data, open(filename, "wb"))

    # Statistics
    def get_input_cosine_similarity_of_query(self, query_token, data_dict):
        assert (
            query_token in self.retrieval_records
        ), f"query_token: {query_token} not in retrieval_records"
        memory_token = self.retrieval_records[query_token]
        query_embedding = self.gen_vector_keys(data_dict)
        memory_index = self.tokens.index(memory_token)
        memory_embedding = [
            self.keys[0][memory_index],
            self.keys[1][memory_index],
            self.keys[2][memory_index],
        ]
        similarity = self.compute_input_cosine_similarity(
            query_embedding, memory_embedding
        )
        return similarity

    def get_input_exp_similarity_of_query(self, query_token, data_dict):
        assert (
            query_token in self.retrieval_records
        ), f"query_token: {query_token} not in retrieval_records"
        memory_token = self.retrieval_records[query_token]
        query_embedding = self.gen_vector_keys(data_dict)
        memory_index = self.tokens.index(memory_token)
        memory_embedding = [
            self.keys[0][memory_index],
            self.keys[1][memory_index],
            self.keys[2][memory_index],
        ]
        similarity = self.compute_input_exp_similarity(
            query_embedding, memory_embedding
        )
        return similarity

    def get_output_cosine_similarity_of_query(self, query_token, query_traj):
        assert (
            query_token in self.retrieval_records
        ), f"query_token: {query_token} not in retrieval_records"
        memory_token = self.retrieval_records[query_token]
        memory_traj = self.values[self.tokens.index(memory_token)]["ego_fut_traj"][1:]
        similarity = self.compute_output_cosine_similarity(query_traj, memory_traj)
        return similarity

    def get_output_exp_similarity_of_query(self, query_token, query_traj):
        assert (
            query_token in self.retrieval_records
        ), f"query_token: {query_token} not in retrieval_records"
        memory_token = self.retrieval_records[query_token]
        memory_traj = self.values[self.tokens.index(memory_token)]["ego_fut_traj"][1:]
        similarity = self.compute_output_exp_similarity(query_traj, memory_traj)
        return similarity

    def get_success_stats(self):
        return self.success_stats

    def get_retrieval_records(self):
        return self.retrieval_records, self.retrieval_records_reverse

    def get_deletion_records(self):
        return self.deleted_records

    def get_exist_time(self):
        return self.exist_time

    def get_retrieval_timestep(self):
        return self.retrieval_timestep

    def lookup_memory_fut_traj(self, token):
        assert token in self.tokens, f"Memory with token {token} not found."
        memory = self.values[self.tokens.index(token)]
        return memory["ego_fut_traj"][1:]

    def update_entry_time(self, token, time):
        assert token not in self.exist_time, f"token {token} already in exist_time"
        self.exist_time[token] = {"entry_time": time, "exit_time": -1}

    def update_exit_time(self, token, time):
        assert token in self.exist_time, f"token {token} not in exist_time"
        self.exist_time[token]["exit_time"] = time

    def get_n_remaining_memory(self):
        return len(self.tokens)
