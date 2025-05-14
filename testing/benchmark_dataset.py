import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from CITNP.datasets.dataset_generator import QueuedInterventionDatasetGenerator


def get_datagenerator():
    datagenerator = QueuedInterventionDatasetGenerator(
        function_generator="resnet",
        num_variables=[15],
        sample_size=1000,
        batch_size=16,
        graph_type=["ER"],
        graph_degrees=[15 * 4],
        iterations_per_epoch=1000,
    )
    return datagenerator


def measure_dataset_time(datagenerator, num_batches=20):
    # Measure single batch generation time
    start_time = time.time()
    batch = next(iter(datagenerator))
    first_batch_time = time.time() - start_time

    # Measure multiple batches
    start_time = time.time()
    batches = []
    for i, batch in tqdm(enumerate(datagenerator), total=num_batches):
        if i >= num_batches:  # Measure 10 batches
            break
        batches.append(batch)
    avg_batch_time = (time.time() - start_time) / num_batches

    print(f"First batch time: {first_batch_time:.4f}s")
    print(f"Average batch time: {avg_batch_time:.4f}s")


def _benchmark_dataloader(dataset, num_workers_list, num_batches=20):
    results = {}

    for num_workers in num_workers_list:
        # Create DataLoader with specified workers
        loader = DataLoader(
            dataset=dataset,
            batch_size=None,  # Your generator already creates batches
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Warmup
        start_time = time.time()
        for _ in trange(5, desc="warmup", leave=False):
            next(iter(loader))
        elapsed = time.time() - start_time
        print(f"Warmup time: {elapsed:.4f}s")

        # Timing
        batch_times = []
        iterator = iter(loader)

        for i in trange(num_batches):
            start_time = time.time()
            try:
                _ = next(iterator)
                end_time = time.time()
                time_for_batch = end_time - start_time
                batch_times.append(time_for_batch)
            except StopIteration:
                print(f"Reached end of dataset at batch {i}")
                break

        if batch_times:
            avg_time = sum(batch_times) / len(batch_times)
            results[num_workers] = avg_time
            print(f"Average batch time with {num_workers} workers: {avg_time:.4f}s")
        else:
            results[num_workers] = None
            print(f"No batches processed with {num_workers} workers")

    return results


def measure_dataloader_time(datagenerator):

    # Test with different worker counts
    dataset = datagenerator
    worker_counts = [2, 4, 8, 10, 16]  # Adjust based on your CPU
    results = _benchmark_dataloader(dataset, worker_counts)

    for workers, avg_time in results.items():
        print(f"Workers: {workers}, Avg time per batch: {avg_time:.4f}s")


def main():
    datagenerator = get_datagenerator()
    # measure_dataset_time(datagenerator)
    measure_dataloader_time(datagenerator)


if __name__ == "__main__":
    main()
