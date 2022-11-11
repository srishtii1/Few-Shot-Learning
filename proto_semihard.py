import os

path = "models/"
resnet_ft = os.path.join(path, "model_ft")
resnet50_ft = os.path.join(path, "org_transfer.h5")
print(resnet50_ft)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.models import resnet18, resnet50
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

batch_size = 128
n_workers = 12
image_size = 224
train_set = Flowers102(
    root="./data",
    split="train",
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=n_workers,
    pin_memory=True,
    shuffle=True,
)

n_way = 5
n_shot = 6
n_query = 4
n_validation_tasks = 100

val_set = Flowers102(
    root="./data",
    split="val",
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)
val_set.get_labels = lambda: [instance[1] for instance in val_set]
val_sampler = TaskSampler(
    val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
)

val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=2,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

n_way = 5
n_shot = 4
n_query = 6
n_test_tasks = 200

test_set = Flowers102(
    root="./data",
    split="test",
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]
    ),
    download=True,
)
test_set.get_labels = lambda: [instance[1] for instance in test_set]
test_sampler = TaskSampler(
    test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=2,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)


def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
            .detach()
            .data,
            1,
        )[1]
        == query_labels.cuda()
    ).sum().item(), len(query_labels)


def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )


"""# Proto Model Classes"""


class PrototypicalNetworksNormal(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworksNormal, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


import torch
from torch import Tensor
from easyfsl.utils import compute_prototypes
import random

DEVICE = "cuda"
k = 10


class PrototypicalNetworksKNN(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworksKNN, self).__init__()
        self.backbone = backbone
        self.power = random.randint(1, 5)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Performing
        dists = torch.cdist(z_query, z_support)

        dists, indices = torch.sort(dists, dim=1)

        dists = dists[:, :k]
        indices = indices[:, :k]
        mode_map = []

        for row in indices:
            mode_map.append([0 for i in range(n_way)])
            for j in row:
                mode_map[-1][support_labels[j]] += 1
        mode_map = torch.tensor(mode_map)

        x = (mode_map.float()).to(DEVICE)

        dists = torch.cdist(z_query, z_proto).to(DEVICE)
        delta = 1
        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -(dists / ((torch.pow(x, self.power)) + delta))

        return scores


"""# Define Triplet Loss"""

import torch
import torch.nn as nn


def pairwise_distance_torch(embeddings, device):

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(
        pairwise_distances_squared, torch.tensor([0.0]).to(device)
    )
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.0
    error_mask[error_mask <= 0.0] = 0.0

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones(
        (pairwise_distances.shape[0], pairwise_distances.shape[1])
    ) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(
        pairwise_distances.to(device), mask_offdiagonals.to(device)
    )
    return pairwise_distances


def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = (
        torch.min(
            torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True
        )[0]
        + axis_maximums[0]
    )
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = (
        torch.max(
            torch.mul(pdist_matrix - axis_minimums[0], adjacency_not),
            dim=1,
            keepdim=True,
        )[0]
        + axis_minimums[0]
    )
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(
        torch.ones(batch_size)
    ).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (
        torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.0]).to(device))
    ).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input, target, **kwargs):
        return TripletSemiHardLoss(target, input, self.device)


import warnings

warnings.filterwarnings("ignore")

N_TRAINING_EPISODES = 1000
N_VALIDATION_TASKS = 100
learning_rates = [0.01, 0.001, 0.0001]
d = {
    1: "Normal Proto - Resnet 50 srshti pretrained",
    2: "Normal Proto - Resnet 50 package pretrained",
    3: "KNN Proto - Resnet 50 srshti pretrained",
    4: "KNN Proto - Resnet 50 package pretrained",
}
for model_no in range(4):
    for opt in range(2):
        for learning_rate in learning_rates:
            x = "ADAM" if opt == 0 else "SGD"
            print(f"Running for {d[model_no+1]} opt = {x} lr = {learning_rate}")
            if model_no == 0:
                convolutional_network = resnet50(pretrained=False)
                convolutional_network.fc.requires_grad = False
                convolutional_network.load_state_dict(torch.load(resnet50_ft))
                convolutional_network.fc = nn.Flatten()

                model = PrototypicalNetworksNormal(convolutional_network).cuda()
            elif model_no == 1:
                convolutional_network = resnet50(pretrained=True)
                convolutional_network.fc = nn.Flatten()

                model = PrototypicalNetworksNormal(convolutional_network).cuda()
            elif model_no == 2:
                convolutional_network = resnet50(pretrained=False)
                convolutional_network.fc.requires_grad = False
                convolutional_network.load_state_dict(torch.load(resnet50_ft))
                convolutional_network.fc = nn.Flatten()

                model = PrototypicalNetworksKNN(convolutional_network).cuda()
            elif model_no == 3:
                convolutional_network = resnet50(pretrained=True)
                convolutional_network.fc = nn.Flatten()

                model = PrototypicalNetworksKNN(convolutional_network).cuda()

            train_set.get_labels = lambda: [instance[1] for instance in train_set]
            train_sampler = TaskSampler(
                train_set,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                n_tasks=N_TRAINING_EPISODES,
            )
            train_loader = DataLoader(
                train_set,
                batch_sampler=train_sampler,
                num_workers=12,
                pin_memory=True,
                collate_fn=train_sampler.episodic_collate_fn,
            )

            criterion = TripletLoss(0)
            optimizer = (
                optim.Adam(model.parameters(), lr=learning_rate)
                if opt == 0
                else optim.SGD(model.parameters(), lr=learning_rate)
            )

            def fit(
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor,
                query_labels: torch.Tensor,
            ) -> float:
                optimizer.zero_grad()
                classification_scores = model(
                    support_images.cuda(), support_labels.cuda(), query_images.cuda()
                )

                loss = criterion
                a = loss.forward(classification_scores, query_labels.cuda())
                # loss.backward()
                optimizer.step()
                return a.item()

            log_update_frequency = 10

            all_loss = []
            model.train()
            with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
                for episode_index, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
                ) in tqdm_train:
                    loss_value = fit(
                        support_images, support_labels, query_images, query_labels
                    )
                    all_loss.append(loss_value)

                    if episode_index % log_update_frequency == 0:
                        tqdm_train.set_postfix(
                            loss=sliding_average(all_loss, log_update_frequency)
                        )

            print("\n testing on val")
            evaluate(val_loader)
            print("testing on test")
            evaluate(test_loader)
