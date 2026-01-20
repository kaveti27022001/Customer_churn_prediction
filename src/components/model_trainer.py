import os
import sys
from typing import Tuple

import dill
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.entity.estimator import MyModel


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """Load numpy array from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> object:
    """Load a serialized object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """Serialize and persist an object to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise MyException(e, sys) from e

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def _encode_labels(y: np.ndarray) -> np.ndarray:
        """Convert churn labels to binary 0/1, tolerant to string inputs."""
        if y.dtype == object or y.dtype.kind in {"U", "S", "O"}:
            return np.where(np.char.lower(y.astype(str)) == "yes", 1, 0).astype(int)
        return y.astype(int)

    def _train_xgboost(self, x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        model = XGBClassifier(
            n_estimators=self.model_trainer_config._n_estimators,
            max_depth=self.model_trainer_config._max_depth,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.model_trainer_config._random_state,
            eval_metric="logloss",
        )
        model.fit(x_train, y_train)
        return model

    def _train_random_forest(self, x_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=self.model_trainer_config._n_estimators,
            min_samples_split=self.model_trainer_config._min_samples_split,
            min_samples_leaf=self.model_trainer_config._min_samples_leaf,
            max_depth=self.model_trainer_config._max_depth,
            criterion=self.model_trainer_config._criterion,
            random_state=self.model_trainer_config._random_state,
        )
        model.fit(x_train, y_train)
        return model

    def _train_ann(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 20, batch_size: int = 256):
        input_dim = x_train.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(
            torch.from_numpy(x_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int64)),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        class TorchModelWrapper:
            def __init__(self, net: nn.Module):
                self.net = net.eval()
                self.device = device

            def predict(self, X):
                with torch.no_grad():
                    xb = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
                    logits = self.net(xb)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                    return preds

        return TorchModelWrapper(model)

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Train XGBoost, RandomForest, and a small ANN; pick the best f1 on the test split.
        """
        try:
            # Split
            x_train, y_train_raw = train[:, :-1], train[:, -1]
            x_test, y_test_raw = test[:, :-1], test[:, -1]
            y_train = self._encode_labels(y_train_raw)
            y_test = self._encode_labels(y_test_raw)

            # Train candidates
            logging.info("Training XGBoost classifier")
            xgb_model = self._train_xgboost(x_train, y_train)

            logging.info("Training RandomForest classifier")
            rf_model = self._train_random_forest(x_train, y_train)

            logging.info("Training ANN classifier")
            ann_model = self._train_ann(x_train, y_train)

            # Evaluate
            def _eval(model):
                preds = model.predict(x_test)
                f1 = f1_score(y_test, preds, pos_label=1)
                precision = precision_score(y_test, preds, pos_label=1)
                recall = recall_score(y_test, preds, pos_label=1)
                acc = accuracy_score(y_test, preds)
                return acc, ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)

            xgb_acc, xgb_metrics = _eval(xgb_model)
            rf_acc, rf_metrics = _eval(rf_model)
            ann_acc, ann_metrics = _eval(ann_model)

            # Pick best by f1, break ties by precision, then recall
            candidates = [
                ("xgboost", xgb_model, xgb_metrics),
                ("random_forest", rf_model, rf_metrics),
                ("ann", ann_model, ann_metrics),
            ]
            candidates.sort(key=lambda t: (t[2].f1_score, t[2].precision_score, t[2].recall_score), reverse=True)
            best_name, best_model, best_metrics = candidates[0]
            logging.info(f"Selected {best_name} based on validation f1: {best_metrics.f1_score:.4f}")
            return best_model, best_metrics

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold (on encoded labels)
            train_y_encoded = self._encode_labels(train_arr[:, -1])
            train_preds = trained_model.predict(train_arr[:, :-1])
            if accuracy_score(train_y_encoded, train_preds) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e