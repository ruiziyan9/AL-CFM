# Estimating Interventional Distributions with Uncertain Causal Graphs through Meta-Learning

This repository is the official implementation of the Neurips 2025 paper [Estimating Interventional Distributions with Uncertain Causal Graphs through Meta-Learning
](https://arxiv.org/abs/2507.05526). 

We tackle the problem of estimating intervetional distributions (useful for causal effects), when the causal graph is unkown. Given an observational dataset, this requires first estimating the causal graph, and then estimating the interventional distribution based on that causal graph.
As this problem is underspecified (multiple causal graphs might explain a dataset equally well), it is possible to maintain uncertainty over causal graphs by following a Bayesian approach. 
While traditional methods explicitly represent the posterior over causal graphs and then compute the interventional distribution for each sampled causal graph, we amortise the entire causal discovery/inference process with a Neural process/prior fitted network.

<img width="1086" height="441" alt="mace_tnp_approach" src="https://github.com/user-attachments/assets/fecbad90-7c6b-4393-a5c6-db351443cbdc" />

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
pip install -e .
```

## Training

To train the model, set the args work_dir and results_path run the bash file:

```train
bash CITNP/experiments/run.sh
```

This runs the training script in `CITNP/experiments/iterable_train.py` which generates data while the model is training.

It is also possible to generate data and save it using the script `CITNP/datasets/create_save_synth_data.py` with the appropriate data generation settings, then train using the training file `CITNP/experiments/file_train.py`.

Data created by `CITNP/datasets/create_save_synth_data.py` is saved in `CITNP/datasets/synth_training_data`, while trained models are saved in `CITNP/experiments/results`.

To evaluate a trained model, create a test set using `CITNP/datasets/create_save_synth_data.py`, input the saved dataset name into the arg data_file, saved model name into the run_name arg and run the bash file:

```evaluate
bash CITNP/experiments/test.sh
```

## Results

### Does our model behave as expected?

We show that when the graph is identifiable from data (left), our model finds the correct posterior (blue) and the correct interventional distribution (red). However, when the graph is not identifiable from data (right), our model finds the correct posterior (blue), but is unable to find the correct interventional distribution (red) as expected. Conditioning on interventional points then allows for our model to find the correct interventional distribution (green).

<img width="983" height="384" alt="Screenshot 2025-12-21 at 14 29 54" src="https://github.com/user-attachments/assets/6a95573c-7ed0-45c7-a3fe-674e5e1a071f" />


### Can our model automatically adjust?

To test if our model can implicitly do the required adjustment to find an interventional distribution, we train on data from two (identifiable) causal graphs, a mediator and a confounder graph. The required adjustment to find the interventional distribution for each graph is different. At test time, when given a confounder or a mediator graph (right), our model is able to find the correct interventional distribution thus implicitly doing the required adjustment. If we hide the confounder, our model again cannot find the correct interventional distribution.  

<img width="863" height="384" alt="Screenshot 2025-12-21 at 14 29 29" src="https://github.com/user-attachments/assets/8a0cd8ca-022b-4fbe-8b6a-2770c81bc57a" />


### How do we generate the training data?

As our model is doing Bayesian inference, the training data effectively forms the prior. Misspecifying this can have a detrimental effect on the performance. However, an advantage of our approach the ability to very easily incorporate different model priors, without having to change the inference scheme. We show below that training on both NN and GP data retains the performance than on training on one of them alone.

<img width="786" height="209" alt="Screenshot 2025-12-21 at 14 36 28" src="https://github.com/user-attachments/assets/c42c71ee-7090-44c7-8b83-00151a54ab9a" />


### How does our model scale?

A challenge in causal discovery/inference is scaling to a large number of variables. We show that our model performs better than competing approaches as we scale the number of variables.

<img width="896" height="230" alt="Screenshot 2025-12-21 at 14 39 13" src="https://github.com/user-attachments/assets/0a444822-6d44-463a-bab6-265abe3aaf43" />

