import time

import torch
from torch.utils.data import DataLoader

from CITNP.datasets.dataset_generator import InterventionDatasetGenerator


def get_datagenerator():
    datagenerator = InterventionDatasetGenerator(
        function_generator="resnet",
        num_variables=[15],
        sample_size=1000,
        batch_size=16,
        graph_type=["ER"],
        graph_degrees=[15 * 4],
        iterations_per_epoch=1,
    )
    return datagenerator


def measure_dataset_time(datagenerator):
    # Measure single batch generation time
    start_time = time.time()
    batch = next(iter(datagenerator))
    first_batch_time = time.time() - start_time

    # Measure multiple batches
    start_time = time.time()
    batches = []
    for i, batch in enumerate(datagenerator):
        if i >= 100:  # Measure 10 batches
            break
        batches.append(batch)
    avg_batch_time = (time.time() - start_time) / 100

    print(f"First batch time: {first_batch_time:.4f}s")
    print(f"Average batch time: {avg_batch_time:.4f}s")


def _benchmark_dataloader(dataset, num_workers_list, num_batches=100):
    results = {}

    for num_workers in num_workers_list:
        # Create DataLoader with specified workers
        loader = DataLoader(
            dataset=dataset,
            batch_size=None,  # Your generator already creates batches
            num_workers=num_workers,
            pin_memory=True,
        )

        # Warmup
        for _ in range(2):
            next(iter(loader))

        # Timing
        start_time = time.time()
        for i, _ in enumerate(loader):
            if i >= num_batches:
                break

        elapsed = time.time() - start_time
        results[num_workers] = elapsed / num_batches

    return results


def measure_dataloader_time(datagenerator):

    # Test with different worker counts
    dataset = datagenerator
    worker_counts = [0, 1, 2, 4, 8, 16]  # Adjust based on your CPU
    results = _benchmark_dataloader(dataset, worker_counts)

    for workers, avg_time in results.items():
        print(f"Workers: {workers}, Avg time per batch: {avg_time:.4f}s")


def main():
    datagenerator = get_datagenerator()
    measure_dataset_time(datagenerator)
    measure_dataloader_time(datagenerator)


if __name__ == "__main__":
    main()
