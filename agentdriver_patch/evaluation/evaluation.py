import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import os
import argparse
import pandas as pd

def easy_eval_l2(pred_traj, gt_traj, metric="uniad"):
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

def planning_evaluation(pred_trajs_dict, config, success_threshold):
    future_second = 3
    ts = future_second * 2
    device = torch.device('cpu')

    # define the success threshold
    # success_threshold = 7.5

    if config.metric=="uniad":
        from agentdriver.evaluation.metric_uniad import PlanningMetric
        with open(Path(os.path.join(config.gt_folder, 'uniad_gt_seg.pkl')),'rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
    elif config.metric=="stp3":
        from agentdriver.evaluation.metric_stp3 import PlanningMetric
        with open(Path(os.path.join(config.gt_folder, 'stp3_gt_seg.pkl')),'rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
            gt_occ_map[token] = torch.flip(gt_occ_map[token], [-1])
            gt_occ_map[token] = torch.flip(gt_occ_map[token], [-2])
    else:
        raise ValueError(f"Invalid metric: {config.metric}")

    metric_planning_val = PlanningMetric(ts).to(device)

    with open(Path(os.path.join(config.gt_folder, 'gt_traj.pkl')),'rb') as f:
        gt_trajs_dict = pickle.load(f)

    with open(Path(os.path.join(config.gt_folder, 'gt_traj_mask.pkl')),'rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    successful_trajs = 0

    # DataFrame to store all evaluation details
    columns = ['token', 'L2_list', 'Success_list', 'token_success', 'retrieved_tokens', 'gpt_retrieved_tokens']
    evaluation_records = []

    for index, token in enumerate(tqdm(pred_trajs_dict.keys())):
        gt_trajectory = torch.tensor(gt_trajs_dict[token])
        gt_trajectory = gt_trajectory.to(device)

        gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
        gt_traj_mask = gt_traj_mask.to(device)
        if gt_traj_mask.shape[0] != 1 and gt_traj_mask.shape[1] != 6 and gt_traj_mask.shape[1] != 2:
            print(f"Invalid gt_traj_mask shape: {gt_traj_mask.shape}")
            continue

        output_trajs = torch.tensor(pred_trajs_dict[token])
        try:
            output_trajs = output_trajs.reshape(gt_traj_mask.shape)
        except Exception as e:
            print(f"Invalid output_trajs shape: {output_trajs.shape}")
            continue
        output_trajs = output_trajs.to(device)

        occupancy: Tensor = gt_occ_map[token]
        occupancy = occupancy.to(device)

        if output_trajs.shape[1] % 2:  # in case the current timestep is included
            output_trajs = output_trajs[:, 1:]

        if occupancy.shape[1] % 2:  # in case the current timestep is included
            occupancy = occupancy[:, 1:]

        if gt_trajectory.shape[1] % 2:  # in case the current timestep is included
            gt_trajectory = gt_trajectory[:, 1:]

        if gt_traj_mask.shape[1] % 2:  # in case the current timestep is included
            gt_traj_mask = gt_traj_mask[:, 1:]

        metric_planning_val(output_trajs[:, :ts], gt_trajectory[:, :ts], occupancy[:, :ts], token, gt_traj_mask)

        ### Use easy_eval_traj to calculate uniad avg l2. More standard metric
        for i in range(len(output_trajs)):
            output_traj = output_trajs[i, :ts].cpu().numpy()
            gt_traj = gt_trajectory[i, :ts].cpu().numpy()
            avg_l2 = easy_eval_l2(output_traj, gt_traj, config.metric)
            success = avg_l2 < success_threshold
            successful_trajs += success

    success_rate = successful_trajs / len(pred_trajs_dict) * 100
    print(f"Success rate: {success_rate:.2f}% ({successful_trajs}/{len(pred_trajs_dict)})")

    # Save evaluation records to CSV
    df = pd.DataFrame(evaluation_records)
    output_path = Path("evaluation_details.csv")
    df.to_csv(output_path, index=False)

    # Continue with the previous output code
    results = {}
    scores = metric_planning_val.compute()
    for i in range(future_second):
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)] = value[:(i+1)*2].mean()

    headers = ["Method", "L2 (m)", "Collision (%)", "Success Rate (%)"]
    sub_headers = ["1s", "2s", "3s", "Avg."]
    if config.metric == "uniad":
        method = (config.method, "{:.2f}".format(scores["L2"][1]), "{:.2f}".format(scores["L2"][3]), "{:.2f}".format(scores["L2"][5]), \
                  "{:.2f}".format((scores["L2"][1]+ scores["L2"][3]+ scores["L2"][5]) / 3.), \
                  "{:.2f}".format(scores["obj_box_col"][1]*100), \
                  "{:.2f}".format(scores["obj_box_col"][3]*100), \
                  "{:.2f}".format(scores["obj_box_col"][5]*100), \
                  "{:.2f}".format(100*(scores["obj_box_col"][1]+ scores["obj_box_col"][3]+ scores["obj_box_col"][5]) / 3.), \
                  "{:.2f}".format(success_rate))
        print("{:<15} {:<20} {:<20} {:<20}".format(*headers))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers, ""))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))

    elif config.metric == "stp3":
        method = (config.method, "{:.2f}".format(results["plan_L2_1s"]), "{:.2f}".format(results["plan_L2_2s"]), "{:.2f}".format(results["plan_L2_3s"]), \
                  "{:.2f}".format((results["plan_L2_1s"]+results["plan_L2_2s"]+results["plan_L2_3s"])/3.), \
                  "{:.2f}".format(results["plan_obj_box_col_1s"]*100), "{:.2f}".format(results["plan_obj_box_col_2s"]*100), "{:.2f}".format(results["plan_obj_box_col_3s"]*100), \
                  "{:.2f}".format(((results["plan_obj_box_col_1s"] + results["plan_obj_box_col_2s"] + results["plan_obj_box_col_3s"])/3)*100), \
                  "{:.2f}".format(success_rate))
        print("{:<15} {:<20} {:<20} {:<20}".format(*headers))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers, ""))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))


def load_pred_trajs_from_file(path):
    with open(path, "rb") as f:
        pred_trajs_dict = pickle.load(f)
    return pred_trajs_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of planning')
    parser.add_argument('--method', type=str, help='name of the method being evaluated, used for table print', default='Agent-Driver')
    parser.add_argument('--result_file', type=str, help='path to the result file', default='temp_results/refined_trajs_dict_0.0_5.0_1.265_7.89.pkl')
    parser.add_argument('--metric', type=str, default='uniad', help='metric to evaluate, either uniad or stp3')
    parser.add_argument('--gt_folder', type=str, default='data/metrics')
    parser.add_argument("--success_threshold", type=float, required=True)
    config = parser.parse_args()

    result_file = Path(config.result_file)
    pred_trajs_dict = load_pred_trajs_from_file(result_file)
    planning_evaluation(pred_trajs_dict, config, config.success_threshold)
