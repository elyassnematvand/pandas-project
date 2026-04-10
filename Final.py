import os
import sqlite3
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class IncomeAnalyst:
    def __init__(self, filename="adult_data_cleaned.csv"):
        """
        Initialize paths, model, and runtime state.
        """
        # Safe base directory for script / notebook
        if "__file__" in globals():
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = os.getcwd()

        # Resolve file path
        if os.path.isabs(filename):
            self.filename = filename
        else:
            self.filename = os.path.join(self.base_dir, filename)

        # Model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Encoders / state
        self.label_encoder_y = LabelEncoder()
        self.df = None
        self.trained_pipeline = None
        self.last_result_df = None

    def get_data(self):
        """
        Load CSV safely.
        """
        if not os.path.exists(self.filename):
            print(f"❌ File not found: {self.filename}")
            return False

        try:
            self.df = pd.read_csv(self.filename)
            print(f"✅ Data loaded successfully from: {self.filename}")
            return True

        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False

    def run_model(self):
        """
        Split -> preprocess -> train -> evaluate
        """
        if self.df is None:
            print("❌ No data available. Run get_data() first.")
            return False

        if "income" not in self.df.columns:
            print("❌ 'income' column not found.")
            return False

        X = self.df.drop(columns=["income"]).copy()
        y_raw = self.df["income"].copy()

        # Remove missing target
        valid_target_mask = y_raw.notna()
        X = X.loc[valid_target_mask]
        y_raw = y_raw.loc[valid_target_mask]

        if X.empty:
            print("❌ No valid rows found.")
            return False

        y = self.label_encoder_y.fit_transform(y_raw)

        # Safe stratify
        class_counts = pd.Series(y).value_counts()
        use_stratify = class_counts.min() >= 2

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if use_stratify else None
        )

        # Column detection
        numeric_cols = X_train.select_dtypes(include=["number"]).columns
        categorical_cols = X_train.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        # Preprocessing
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )

        # Full pipeline
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        print("⏳ Training model...")
        model_pipeline.fit(X_train, y_train)

        self.trained_pipeline = model_pipeline

        predictions = model_pipeline.predict(X_test)

        print("\n" + "=" * 30)
        print("MODEL PERFORMANCE REPORT")
        print("=" * 30)

        print(
            classification_report(
                y_test,
                predictions,
                target_names=self.label_encoder_y.classes_,
                zero_division=0
            )
        )

        # Save results
        result_df = X_test.copy()
        result_df["actual_income"] = self.label_encoder_y.inverse_transform(y_test)
        result_df["predicted_income"] = self.label_encoder_y.inverse_transform(predictions)

        self.last_result_df = result_df

        return True

    def save_to_sql(self, db_name="income_results.db"):
        """
        Save data + predictions to SQLite
        """
        if self.df is None:
            print("❌ No data to save.")
            return False

        if not os.path.isabs(db_name):
            db_name = os.path.join(self.base_dir, db_name)

        try:
            with sqlite3.connect(db_name) as conn:
                self.df.to_sql(
                    "processed_data",
                    conn,
                    if_exists="replace",
                    index=False
                )

                if self.last_result_df is not None:
                    self.last_result_df.to_sql(
                        "model_predictions",
                        conn,
                        if_exists="replace",
                        index=False
                    )

            print(f"✅ Data saved to SQL: {db_name}")
            return True

        except sqlite3.Error as e:
            print(f"❌ SQLite error: {e}")
            return False

    def save_model(self, model_name="income_model.pkl"):
        """
        Save trained pipeline + label encoder
        """
        if self.trained_pipeline is None:
            print("❌ No trained model to save.")
            return False

        if not os.path.isabs(model_name):
            model_name = os.path.join(self.base_dir, model_name)

        model_bundle = {
            "pipeline": self.trained_pipeline,
            "label_encoder": self.label_encoder_y
        }

        joblib.dump(model_bundle, model_name)

        print(f"✅ Model saved: {model_name}")
        return True

    def load_model(self, model_name="income_model.pkl"):
        """
        Load trained model
        """
        if not os.path.isabs(model_name):
            model_name = os.path.join(self.base_dir, model_name)

        if not os.path.exists(model_name):
            print(f"❌ Model file not found: {model_name}")
            return False

        model_bundle = joblib.load(model_name)

        self.trained_pipeline = model_bundle["pipeline"]
        self.label_encoder_y = model_bundle["label_encoder"]

        print(f"✅ Model loaded: {model_name}")
        return True

    def predict_income(self, new_data: pd.DataFrame):
        """
        Predict new income values
        """
        if self.trained_pipeline is None:
            print("❌ No trained model loaded.")
            return None

        preds = self.trained_pipeline.predict(new_data)

        return self.label_encoder_y.inverse_transform(preds)


if __name__ == "__main__":
    analyst = IncomeAnalyst()

    if analyst.get_data():
        if analyst.run_model():
            analyst.save_to_sql()
            analyst.save_model()