import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, TensorDataset

from FantAIno.models.fantaino_base import FantAInoFitter
from FantAIno.utils.metrics import rounded_regression_accuracy

class NeuralNetworkModel(FantAInoFitter,L.LightningModule):
    """Dense Neural Network model for FantAIno."""

    def __init__(
        self,
        mode: str = "regression",
        hidden_dims: list[int] = [128, 128, 128],
        output_dim: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU,
        final_activation: torch.nn.Module | None = None,
        dropout: list[float] | float = 0.2,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        optimizer: callable = torch.optim.SGD,
        trainer_dict: dict = {},
        early_stopping_dict: dict = {},
        model_run_name: str = "master",
        overwrite_previous_ckpt: bool = False,
    ):
        FantAInoFitter.__init__(self)
        L.LightningModule.__init__(self)

        self.mode = mode
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.final_activation = final_activation
        self.dropout = dropout
        self.loss_fn = loss_fn
        self.optimizer = optimizer

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

        self.model_run_name = model_run_name
        self.checkpoint_path = os.path.join(self.root_dir, "results", self.model_name)
        ckpt_file = os.path.join(self.checkpoint_path, f"{model_run_name}.ckpt")
        if overwrite_previous_ckpt and os.path.exists(ckpt_file):
            os.remove(ckpt_file)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_path,
            filename=f"{self.model_run_name}"
        )
        if early_stopping_dict:
            early_stopping_callback = EarlyStopping(**early_stopping_dict)
            self.trainer = L.Trainer(
                **trainer_dict,
                default_root_dir=self.checkpoint_path,
                callbacks=[checkpoint_callback, early_stopping_callback]
            )
        else:
            self.trainer = L.Trainer(
                **trainer_dict,
                default_root_dir=self.checkpoint_path,
                callbacks=[checkpoint_callback]
            )
        
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
                self.activation(),
            )
            # handle hidden layers
            for i in range(len(self.hidden_dims)-1):
                self.model.append(torch.nn.Dropout(self.dropout[i]))
                self.model.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                self.model.append(self.activation())
            # handle output layer
            self.model.append(torch.nn.Dropout(self.dropout[-1]))
            self.model.append(torch.nn.Linear(self.hidden_dims[-1], self.output_dim))
            if self.final_activation:
                self.model.append(self.final_activation)

    def train_model(self, X_train, y_train, val_proportion: float = 0.15):
        all_data = TensorDataset(X_train, y_train)
        train_dataset, validation_dataset = torch.utils.data.random_split(all_data, [1-val_proportion, val_proportion])
        train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)
        validation_data = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True)
        self.trainer.fit(self, train_data, validation_data)

    def predict_from_model(self, X_test):
        test_data = DataLoader(TensorDataset(X_test), batch_size=64, shuffle=False, num_workers=8, persistent_workers=True)
        predictions_list = self.trainer.predict(self, test_data)
        predictions = torch.cat(predictions_list, axis=0)
        return predictions

    def evaluate_model(self, X_test, y_test, loss_fn: callable = None):
        test_data = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False, num_workers=8, persistent_workers=True)
        test_loss_dict = self.trainer.test(self, test_data)
        return test_loss_dict[0]["test_loss"]

    def save_estimator(self, model_run_name: str):
        """Save the current estimator the way Pytorch Lightning expects."""
        print("NOTE: Saving of the estimator is handled by the trainer.")
    
    def load_estimator(self, model_run_name: str):
        loaded_model = NeuralNetworkModel.load_from_checkpoint(
            checkpoint_path=os.path.join(self.checkpoint_path, f"{model_run_name}.ckpt"),
            mode=self.mode,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
            final_activation=self.final_activation,
            dropout=self.dropout,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            model_run_name=model_run_name,
        )
        self.model = loaded_model.model

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", rounded_regression_accuracy(y.detach().cpu().numpy(), output.detach().cpu().numpy()), prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("validation_loss", loss)
        self.log("validation_accuracy", rounded_regression_accuracy(y.detach().cpu().numpy(), output.detach().cpu().numpy()), prog_bar=True, logger=True)
        return loss

    def predict_step(self, pred_batch, pred_batch_idx):
        x, = pred_batch
        output = self(x)
        return output

    def test_step(self, test_batch, test_batch_idx):
        x, y = test_batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", rounded_regression_accuracy(y.detach().cpu().numpy(), output.detach().cpu().numpy()), prog_bar=True, logger=True)
        return loss

    