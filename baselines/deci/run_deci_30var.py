import argparse
import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from causica.datasets.causica_dataset_format import Variable
from causica.datasets.tensordict_utils import expand_td_with_batch_shape
from causica.distributions import ContinuousNoiseDist
from causica.lightning.data_modules.basic_data_module import \
    BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.training.auglag import AugLagLRConfig
from pytorch_lightning.callbacks import TQDMProgressBar
from scipy.special import logsumexp
from tensordict import TensorDict
from tqdm import tqdm

from CITNP.utils.processing import normalise_variable

test_run = bool(
    os.environ.get("TEST_RUN", False)
)  # used by testing to run the notebook as a script


def load_data(data_path: Path):
    with h5py.File(data_path, "r") as f:
        obs_data = f["obs_data"][:]
        int_data = f["int_data"][:]
        causal_graphs = f["causal_graphs"][:]
        intvn_indices = f["intvn_indices"][:]
        variable_counts = f["variable_counts"][:]

    data_dict = {
        "obs_data": obs_data,
        "int_data": int_data,
        "causal_graphs": causal_graphs,
        "intvn_indices": intvn_indices,
        "variable_counts": variable_counts,
    }
    return data_dict


def convert_data_to_pd(data: np.ndarray):
    """
    Converts the data to a pandas dataframe.
    """
    # Set header
    header = [f"X{i}" for i in range(data.shape[1])]
    obs_data = pd.DataFrame(
        data,
        columns=header,
    )
    return obs_data


def per_variable_log_prob(value: TensorDict, do_sem: DistributionParametersSEM):
    """
    Computes the log probability of each variable in the SEM.
    """
    expanded_value = expand_td_with_batch_shape(value, do_sem.batch_shape)
    noise_object = do_sem.noise_dist(do_sem.func(expanded_value, do_sem.graph))
    log_probs = [
        noise_dist.log_prob(value[name])
        for name, noise_dist in noise_object._independent_noise_dists.items()
    ]
    return torch.stack(log_probs, dim=0)


def calculate_intvn_dist(
    data_module: BasicDECIDataModule,
    sem_list: List[DistributionParametersSEM],
    intvn_indices: int,
    int_data: np.ndarray,
    int_data_dict: TensorDict,
):
    int_data = torch.from_numpy(int_data).to(torch.float32)
    int_values = int_data[:, intvn_indices]
    outcome_idx = 1 if intvn_indices == 0 else 0
    outcome_node = int_data[:, outcome_idx]

    all_scm_log_prob = np.zeros((len(sem_list), outcome_node.shape[0]))
    for sem_idx, sem in tqdm(enumerate(sem_list), total=len(sem_list)):
        outcome_log_prob = np.zeros_like(outcome_node)
        for idx, curr_int in enumerate(int_values):
            int_dict = TensorDict(
                {f"X{intvn_indices}": curr_int[None]}, batch_size=tuple()
            )
            do_sem = sem.do(
                interventions=int_dict,
            )

            # Gives out log prob of var in order of 0, 1, 2
            # but skips the intventioned variable
            # So we can safely take the first one
            var_log_prob = (
                per_variable_log_prob(int_data_dict[idx : idx + 1], do_sem)[0]
                # per_variable_log_prob(int_dict, do_sem)[0]
                .detach()
                .cpu()
                .numpy()
            )

            outcome_log_prob[idx] = var_log_prob

        all_scm_log_prob[sem_idx] = outcome_log_prob

    graph_sample_size = all_scm_log_prob.shape[0]
    all_sample = all_scm_log_prob.sum(1)
    log_sum_exp = logsumexp(
        all_sample,
        axis=0
    ) - np.log(graph_sample_size)
    return -1 * log_sum_exp


def main(
    data_path: Path,
    data_path_name: str,
    sample_size: int = 500,
    data_start: int = 0,
    data_end: int = 100,
):

    result_folder = (
        Path("/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines")
        / "deci"
        / "results"
        / data_path_name
    )
    result_folder.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)

    all_obs_data = data["obs_data"]
    all_int_data = data["int_data"]
    all_intvn_indices = data["intvn_indices"]

    # synthetic data should already be normalised
    all_obs_data, mean_obs, std_obs = normalise_variable(
        all_obs_data, axis=0, return_stats=True
    )
    all_int_data = normalise_variable(all_int_data, axis=0, mean=mean_obs, std=std_obs)

    all_obs_data = all_obs_data[:, :sample_size]
    all_int_data = all_int_data[:, :sample_size]

    for i in range(data_start, data_end):
        pl.seed_everything(seed=1)  # set the random seed

        obs_data = all_obs_data[i]
        int_data = all_int_data[i]
        intvn_indices = all_intvn_indices[i]

        data_pd = convert_data_to_pd(obs_data)
        data_module = BasicDECIDataModule(
            dataframe=data_pd,
            normalize=False,
            batch_size=128,
            variables=[
                Variable(group_name=f"X{i}", name=f"X{i}")
                for i in range(obs_data.shape[1])
            ],
        )

        int_data_pd = convert_data_to_pd(int_data)
        int_data_module = BasicDECIDataModule(
            dataframe=int_data_pd,
            normalize=False,
            batch_size=128,
            variables=[
                Variable(group_name=f"X{i}", name=f"X{i}")
                for i in range(int_data.shape[1])
            ],
        )

        num_nodes = len(data_module.dataset_train.keys())

        lightning_module = DECIModule(
            noise_dist=ContinuousNoiseDist.SPLINE,
            prior_sparsity_lambda=43.0,
            init_rho=30.0,
            init_alpha=0.20,
            auglag_config=AugLagLRConfig(
                max_inner_steps=3400,
                max_outer_steps=8,
                lr_init_dict={
                    "icgnn": 0.00076,
                    "vardist": 0.0098,
                    "functional_relationships": 3e-4,
                    "noise_dist": 0.0070,
                },
            ),
        )

        trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=2000,
            # max_epochs=1,
            fast_dev_run=test_run,
            callbacks=[TQDMProgressBar(refresh_rate=19)],
            enable_checkpointing=False,
        )
        trainer.fit(lightning_module, datamodule=data_module)

        sem_module = lightning_module.sem_module
        sem_dist = sem_module()
        sem_samples = sem_dist.sample(sample_shape=torch.Size([100]))

        average_neg_log_prob = calculate_intvn_dist(
            data_module=data_module,
            sem_list=sem_samples,
            intvn_indices=intvn_indices,
            int_data=int_data,
            int_data_dict=int_data_module.dataset_train,
        )

        # Save the results
        save_name = data_path_name + f"_deci_nlpd_data{i}.csv"
        save_path = result_folder / save_name
        df = pd.DataFrame([average_neg_log_prob], columns=["NLL"])
        df.to_csv(save_path, index=False)

        print(f"Average log prob for dataset {i}: {average_neg_log_prob}")


if __name__ == "__main__":
    data_path_name = "30var_ER4_neuralgplvm_1000"
    data_name = "data_idx30"

    data_path = Path(
        f"/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/datasets/synth_training_data/{data_path_name}/test/{data_name}.hdf5"
    )

    sample_size = 500

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_start",
        type=int,
        default=0,
        help="data_start",
    )
    parser.add_argument(
        "--data_end",
        type=int,
        default=100,
        help="data_end",
    )
    args = parser.parse_args()

    main(
        data_path=data_path,
        data_path_name=data_path_name,
        sample_size=sample_size,
        data_start=args.data_start,
        data_end=args.data_end,
    )
