import lightning as L
import os
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, TensorDataset

from FantAIno.models.fantaino_base import FantAInoFitter

class NeuralNetworkModel(FantAInoFitter, L.LightningModule):
    """Dense Neural Network model for FantAIno."""

    def __init__(
        self,
        mode: str = "regression",
        hidden_dims: list[int] = [128, 128, 128],
        output_dim: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(),
        final_activation: torch.nn.Module | None = None,
        dropout: list[float] | float = 0.2,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        optimizer: callable = torch.optim.SGD,
        model_run_name: str = "master",
    ):
        FantAInoFitter.__init__(self)
        L.LightningModule.__init__(self)
        self.save_hyperparameters()

        self.mode = mode
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.final_activation = final_activation
        self.dropout = dropout
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model_run_name = model_run_name

        # validate/fix input parameters
        if self.mode == "regression":
            assert self.output_dim == 1, "Output dimension must be 1 for regression."
            self.model_name = "NN Regressor"
        elif self.mode == "classification":
            if self.final_activation is None:
                self.final_activation = torch.nn.Softmax(dim=1)
            self.model_name = "NN Classifier"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        if isinstance(self.dropout, list) and len(self.dropout) != len(self.hidden_dims):
            self.dropout = [0.2] * len(self.hidden_dims)
        elif isinstance(self.dropout, float):
            self.dropout = [self.dropout] * len(self.hidden_dims)

        # build NN
        if len(self.hidden_dims) == 0:
            self.model = torch.nn.LazyLinear(self.output_dim)
        else:
            # handle input
            self.model = torch.nn.Sequential(
                torch.nn.LazyLinear(self.hidden_dims[0]),
                self.activation,
            )
            # handle hidden layers
            for i in range(len(self.hidden_dims)-2):
                self.model.append(torch.nn.Dropout(self.dropout[i]))
                self.model.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                self.model.append(self.activation)
            # handle output layer
            self.model.append(torch.nn.Dropout(self.dropout[-1]))
            self.model.append(torch.nn.Linear(self.hidden_dims[-1], self.output_dim))
            if self.final_activation:
                self.model.append(self.final_activation)

    def train(self, X_train, y_train):
        train_data = DataLoader(TensorDataset(X_train, y_train))
        trainer = L.Trainer()
        trainer.fit(self, train_data)

    def predict(self, X_test):
        test_data = DataLoader(TensorDataset(X_test))
        trainer = L.Trainer()
        return trainer.predict(self, test_data)

    def evaluate(self, X_test, y_test, loss_fn: callable = None):
        test_data = DataLoader(TensorDataset(X_test, y_test))
        trainer = L.Trainer()
        trainer.test(self, test_data)
        return 5

    def save_esimtator(self, model_run_name: str):
        """Save the current estimator the way Pytorch Lightning expects."""
        print("NOTE: Saving of the estimator is handled by the trainer.")

    def load_estimator(self, model_run_name: str):
        checkpoint_path = os.path.join(self.root_dir, "results", self.model_name, f"{model_run_name}.ckpt")
        self.model = self.load_from_checkpoint(checkpoint_path)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("validation_loss", loss)
        return loss

    def predict_step(self, pred_batch, pred_batch_idx):
        x = pred_batch
        output = self(x)
        return output

    def test_step(self, test_batch, test_batch_idx):
        x, y = test_batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("test_loss", loss)
        return loss

    