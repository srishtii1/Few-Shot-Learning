import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.models import resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR

from easyfsl.samplers import TaskSampler
from easyfsl.utils import sliding_average

from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision import transforms

batch_size = 128
n_workers = 12
image_size = 128
train_set = Flowers102(
    root="./data",
    split = 'train',
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)

val_set = Flowers102(
    root="./data",
    split = 'val',
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)



test_set = Flowers102(
    root="./data",
    split = 'test',
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)

import torch
from torch import nn, Tensor
from easyfsl.methods import FewShotClassifier
from easyfsl.modules.predesigned_modules import (
    default_matching_networks_support_encoder,
    default_matching_networks_query_encoder,
)


class MatchingNetworks(FewShotClassifier):
    """
    Matching networks extract feature vectors for both support and query images. Then they refine
    these feature by using the context of the whole support set, using LSTMs. Finally they compute
    query labels using their cosine similarity to support images.

    Matching Networks output log-probabilities, so we want to use Negative Log Likelihood Loss.
    """

    def __init__(
        self,
        *args,
        support_encoder: nn.Module = None,
        query_encoder: nn.Module = None,
        **kwargs
    ):
        """
        Build Matching Networks by calling the constructor of FewShotClassifier.
        Args:
            support_encoder: module encoding support features. If none is specific, we use
                the default encoder from the original paper.
            query_encoder: module encoding query features. If none is specific, we use
                the default encoder from the original paper.

        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(*args, **kwargs)

        if len(self.backbone_output_shape) != 1:
            raise ValueError(
                "Illegal backbone for Matching Networks. "
                "Expected output for an image is a 1-dim tensor."
            )

        # These modules refine support and query feature vectors
        # using information from the whole support set
        self.support_features_encoder = (
            support_encoder
            if support_encoder
            else default_matching_networks_support_encoder(self.feature_dimension)
        )
        self.query_features_encoding_cell = (
            query_encoder
            if query_encoder
            else default_matching_networks_query_encoder(self.feature_dimension)
        )

        self.softmax = nn.Softmax(dim=1)

        # Here we create the fields so that the model can store
        # the computed information from one support set
        self.contextualized_support_features = None
        self.one_hot_support_labels = None

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract features from the support set with full context embedding.
        Store contextualized feature vectors, as well as support labels in the one hot format.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        support_features = self.backbone(support_images)
        self.contextualized_support_features = self.encode_support_features(
            support_features
        )

        self.one_hot_support_labels = nn.functional.one_hot(support_labels).float()

    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict query labels based on their cosine similarity to support set features.
        Classification scores are log-probabilities.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """

        # Refine query features using the context of the whole support set
        contextualized_query_features = self.encode_query_features(
            self.backbone(query_images)
        )

        # Compute the matrix of cosine similarities between all query images
        # and normalized support images
        # Following the original implementation, we don't normalize query features to keep
        # "sharp" vectors after softmax (if normalized, all values tend to be the same)
        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                nn.functional.normalize(self.contextualized_support_features).T
            )
        )

        # Compute query log probabilities based on cosine similarity to support instances
        # and support labels
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    def encode_support_features(
        self,
        support_features: Tensor,
    ) -> Tensor:
        """
        Refine support set features by putting them in the context of the whole support set,
        using a bidirectional LSTM.
        Args:
            support_features: output of the backbone

        Returns:
            contextualised support features, with the same shape as input features
        """

        # Since the LSTM is bidirectional, hidden_state is of the shape
        # [number_of_support_images, 2 * feature_dimension]
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[
            0
        ].squeeze(0)

        # Following the paper, contextualized features are computed by adding original features, and
        # hidden state of both directions of the bidirectional LSTM.
        contextualized_support_features = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )

        return contextualized_support_features

    def encode_query_features(self, query_features: Tensor) -> Tensor:
        """
        Refine query set features by putting them in the context of the whole support set,
        using attention over support set features.
        Args:
            query_features: output of the backbone

        Returns:
            contextualized query features, with the same shape as input features
        """

        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        # We do as many iterations through the LSTM cell as there are query instances
        # Check out the paper for more details about this!
        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1)

            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features

        return hidden_state

    @staticmethod
    def is_transductive() -> bool:
        return False

convolutional_network = resnet50(pretrained=True)
convolutional_network.fc = nn.Flatten()
print(convolutional_network)

model = MatchingNetworks(convolutional_network).cuda()

"""Defining a 5 way 4 shot model"""

N_WAY = 5  # Number of classes in a task
N_SHOT = 4  # Number of images per class in the support set
N_QUERY = 5  # Number of images per class in the query set
N_EVALUATION_TASKS = 500

# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
test_set.get_labels = lambda: [instance[1] for instance in test_set]
test_sampler = TaskSampler(
    test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
):
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    model.process_support_set(support_images.cuda(), support_labels.cuda())
    return (
        torch.max(
            model(query_images.cuda())
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
    return correct_predictions / total_predictions

N_TRAINING_EPISODES_PER_EPOCH = 100
N_VALIDATION_TASKS = 200

train_set.get_labels = lambda: [instance[1] for instance in train_set]
train_sampler = TaskSampler(
    train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES_PER_EPOCH
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

val_set.get_labels = lambda: [instance[1] for instance in val_set]
val_sampler = TaskSampler(
    val_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_VALIDATION_TASKS
)
val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

"""
Here we define our loss and our optimizer (cross entropy and Adam, pretty standard), and a `fit` method.
This method takes a classification task as input (support set and query set). It predicts the labels of the query set
based on the information from the support set; then it compares the predicted labels to ground truth query labels,
and this gives us a loss value. Then it uses this loss to update the parameters of the model. This is a *meta-training loop*.
"""

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
n_epochs = 20
scheduler_milestones = [120, 160]
scheduler_gamma = 0.1
learning_rate = 1e-2
train_scheduler = MultiStepLR(
    optimizer,
    milestones=scheduler_milestones,
    gamma=scheduler_gamma,
)

def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    model.process_support_set(support_images.cuda(), support_labels.cuda())
    classification_scores = model(
        query_images.cuda()
    )

    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()
    #loss item is a float value
    return loss.item()

"""To train the model, we are just going to iterate over a large number of randomly generated few-shot classification tasks,
and let the `fit` method update our model after each task. This is called **episodic training**.
"""

# Train the model yourself with this cell
def training_epoch(
    model: FewShotClassifier, data_loader: DataLoader):
    log_update_frequency = 10

    all_loss = []
    model.train()
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
            loss_value = fit(support_images, support_labels, query_images, query_labels)
            #loss_value is a float
            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))
    return np.mean(all_loss)

def validate_epoch(
    model: FewShotClassifier, data_loader: DataLoader):
    log_update_frequency = 10
    val_loss=0.0  
    val_correct=0.0  
    val_loss_history=[]  
    val_correct_history=[]  
    model.eval()                                              
    
    with torch.no_grad(): 
        with tqdm(enumerate(data_loader), total=len(data_loader), desc="Validating") as tqdm_train:
            for episode_index, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
                ) in tqdm_train:
                model.process_support_set(support_images.cuda(), support_labels.cuda())
                classification_scores = model(
                    query_images.cuda()
                )

                loss_value = criterion(classification_scores, query_labels.cuda())
                
                _,val_preds=torch.max(classification_scores ,1)  
                val_loss+=loss_value.item()  
                val_correct+=torch.sum(val_preds==query_labels.cuda().data) 
                val_epoch_loss=val_loss/len(data_loader)  
                val_epoch_acc=val_correct.float()/len(data_loader)  
                val_loss_history.append(val_epoch_loss)  
                val_correct_history.append(val_epoch_acc)  
                #loss_value is a float
            # all_loss.append(loss_value.to('cpu'))
    return val_epoch_loss


"""testing and Validation!"""
val_results = []
train_loss_results = []
val_loss_results = []
train_results = []

best_state = model.state_dict()
best_validation_accuracy = 0.0
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    average_loss = training_epoch(model, train_loader)
    train_loss_results.append(average_loss)
    average_val_loss = validate_epoch(model, val_loader)
    val_loss_results.append(average_val_loss)

    validation_accuracy = evaluate(val_loader)
    val_results.append(validation_accuracy)
    train_accuracy = evaluate(train_loader)
    train_results.append(train_accuracy)

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_state = model.state_dict()
        print(f"We found a new best model! with val accuracy {best_validation_accuracy}")

    # Warn the scheduler that we did an epoch
    # so it knows when to decrease the learning rate
    train_scheduler.step()


"""Testing"""

evaluate(test_loader)

"""Plot Accuracy"""
plt.figure()
plt.plot(val_results, color='red', label='Validation')
plt.plot(train_results, color='blue', label='Train')
plt.title('Accuracy over epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks([i for i in range(1, n_epochs + 1,5)])
plt.legend(['Validation', 'Train'])
plt.grid(True)
plt.savefig(f'/home/UG/srishti003/pretrained_accuracy_5.png')

"""Plot Loss"""
plt.figure()
plt.plot(val_loss_results, color='red', label='Validation')
plt.plot(train_loss_results, color='blue', label='Validation')
plt.title('Negative Log Likelihood Loss over epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks([i for i in range(1, n_epochs + 1,5)])
plt.legend(['Validation', 'Train'])
plt.grid(True)
plt.savefig(f'/home/UG/srishti003/pretrained_loss_5.png')

