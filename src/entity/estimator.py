class MyModel:
    """
    Custom model wrapper class that combines preprocessing pipeline and trained model.
    """
    def __init__(self, preprocessing_object, trained_model_object):
        """
        :param preprocessing_object: Fitted preprocessing pipeline object
        :param trained_model_object: Trained model object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        Make predictions using the preprocessing pipeline and trained model.
        
        :param X: Input features
        :return: Predictions
        """
        # Transform the input data using the preprocessing pipeline
        transformed_feature = self.preprocessing_object.transform(X)
        
        # Make predictions using the trained model
        return self.trained_model_object.predict(transformed_feature)
    
    def __repr__(self):
        return f"MyModel(preprocessing={type(self.preprocessing_object).__name__}, model={type(self.trained_model_object).__name__})"
