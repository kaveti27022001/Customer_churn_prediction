import sys
import pandas as pd
from src.entity.config_entity import ChurnPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class ChurnData:
    def __init__(self,
                gender,
                SeniorCitizen,
                Partner,
                Dependents,
                tenure,
                PhoneService,
                MultipleLines,
                InternetService,
                OnlineSecurity,
                OnlineBackup,
                DeviceProtection,
                TechSupport,
                StreamingTV,
                StreamingMovies,
                Contract,
                PaperlessBilling,
                PaymentMethod,
                MonthlyCharges,
                TotalCharges
                ):
        """
        Churn Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.gender = gender
            self.SeniorCitizen = SeniorCitizen
            self.Partner = Partner
            self.Dependents = Dependents
            self.tenure = tenure
            self.PhoneService = PhoneService
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.OnlineBackup = OnlineBackup
            self.DeviceProtection = DeviceProtection
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges = TotalCharges

        except Exception as e:
            raise MyException(e, sys) from e

    def get_churn_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from ChurnData class input
        """
        try:
            churn_input_dict = self.get_churn_data_as_dict()
            return DataFrame(churn_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e

    def get_churn_data_as_dict(self):
        """
        This function returns a dictionary from ChurnData class input
        """
        logging.info("Entered get_churn_data_as_dict method of ChurnData class")

        try:
            input_data = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }

            logging.info("Created churn data dict")
            logging.info("Exited get_churn_data_as_dict method of ChurnData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class ChurnDataClassifier:
    def __init__(self, prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def _preprocess_data(self, df: DataFrame) -> DataFrame:
        """
        Preprocess the input data to match the training data format.
        """
        try:
            logging.info("Preprocessing input data for prediction")
            
            # Map gender to binary
            df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).astype(int)
            
            # Define categorical columns to encode
            categorical_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                               'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                               'DeviceProtection', 'TechSupport', 'StreamingTV', 
                               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
            
            # Create dummy variables
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            
            # Expected columns after transformation (based on training)
            expected_columns = [
                'gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 
                'MultipleLines_No phone service', 'MultipleLines_Yes',
                'InternetService_Fiber optic', 'InternetService_No',
                'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                'TechSupport_No internet service', 'TechSupport_Yes',
                'StreamingTV_No internet service', 'StreamingTV_Yes',
                'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                'Contract_One year', 'Contract_Two year',
                'PaperlessBilling_Yes',
                'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
                'PaymentMethod_Mailed check'
            ]
            
            # Add missing columns with 0
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training data
            df = df[expected_columns]
            
            logging.info("Data preprocessing completed")
            return df
            
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe) -> str:
        """
        This is the method of ChurnDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of ChurnDataClassifier class")
            
            # Preprocess the input data
            processed_df = self._preprocess_data(dataframe)
            
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(processed_df)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)
