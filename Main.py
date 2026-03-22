import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class IncomeAnalyst:
    """
    A class to handle data loading, preprocessing, and 
    Random Forest classification for the Adult Income dataset.
    """
    def __init__(self, filename="adult_data_cleaned.csv"):
        self.filename = filename
        self.df = None
        # Model configuration: 100 trees with a max depth of 10 to prevent overfitting
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.le = LabelEncoder()

    def get_data(self):
        """Fetch the dataset from a local file or download it from the UCI repository."""
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename)
            print("--- Status: Dataset loaded from local storage.")
        else:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            columns = ["age", "workclass", "fnlwgt", "education", "education_num",
                       "marital_status", "occupation", "relationship", "race", "gender",
                       "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
            # Downloading data using pandas
            self.df = pd.read_csv(url, names=columns, skipinitialspace=True)
            # Caching the data locally for future use
            self.df.to_csv(self.filename, index=False)
            print("--- Status: Dataset downloaded and cached successfully.")

    def preprocess(self):
        """Convert categorical text features into numerical values using Label Encoding."""
        categorical_cols = ["workclass", "education", "marital_status", "occupation", 
                            "relationship", "race", "gender", "native_country", "income"]
        for col in categorical_cols:
            self.df[col] = self.le.fit_transform(self.df[col])
        print("--- Status: Preprocessing complete.")

    def run_model(self):
        """Train the Random Forest model and evaluate its performance."""
        X = self.df.drop(columns=['income'])
        y = self.df['income']
        
        # Split data using stratify=y to maintain class balance in training/testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print(f"--- Results:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    # Entry point of the script
    analyst = IncomeAnalyst()
    analyst.get_data()
    analyst.preprocess()
    analyst.run_model()