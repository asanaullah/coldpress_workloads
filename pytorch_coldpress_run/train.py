# Assisted by: Claude Sonnet 4.5
"""PyTorch DDP training example - based on demo."""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms


class DynamicMLP(nn.Module):
    """Simple MLP for classification."""

    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--train-test-split", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    # Initialize DDP - detect GPU/CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend)
    rank = dist.get_rank()

    if rank == 0:
        print(f"Using device: {device} (backend: {backend})", flush=True)

    # Dataset loading
    # Use local /tmp for dataset downloads (each pod downloads its own copy)
    # /tmp is pod-local, so only local_rank 0 within each pod should download
    dataset_root = "/tmp/data"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
    )

    # Only local_rank 0 downloads to avoid race conditions within the pod
    if local_rank == 0:
        if args.dataset.lower() == "mnist":
            torchvision.datasets.MNIST(
                root=dataset_root, train=True, download=True, transform=transform
            )
        else:
            raise ValueError("Unsupported dataset")

    # Wait for local_rank 0 to finish downloading
    if dist.is_initialized():
        dist.barrier()

    # All ranks load the dataset
    if args.dataset.lower() == "mnist":
        full_dataset = torchvision.datasets.MNIST(
            root=dataset_root, train=True, download=False, transform=transform
        )
    else:
        raise ValueError("Unsupported dataset")

    input_dim = full_dataset[0][0].numel()
    output_dim = len(full_dataset.classes) if hasattr(full_dataset, "classes") else 10

    # Train-test split
    train_size = int(args.train_test_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,  # Avoid multiprocessing issues with DDP
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=use_cuda,
    )

    # Model
    model = DynamicMLP(input_dim, args.hidden_size, output_dim).to(device)
    if use_cuda:
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop - print every epoch
    # Ensure all processes are ready before training
    dist.barrier()

    start = time.time()
    accuracy = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate and print every epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        if rank == 0:
            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%",
                flush=True,
            )

    end = time.time()

    # Save results (rank 0 only)
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())

        results = {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "train_test_split": args.train_test_split,
            "device": str(device),
            "backend": backend,
            "world_size": dist.get_world_size(),
            "time_seconds": end - start,
            "final_loss": float(loss.detach()),
            "accuracy": accuracy,
            "model_params": num_params,
            "input_dim": input_dim,
            "output_dim": output_dim,
        }

        os.makedirs(args.output_dir, exist_ok=True)

        # Save stats
        with open(f"{args.output_dir}/training_stats.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save model weights
        torch.save(model.module.state_dict(), f"{args.output_dir}/model_weights.pth")
        print(f"Saved model weights to {args.output_dir}/model_weights.pth", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
