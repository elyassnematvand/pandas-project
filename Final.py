import os
import sqlite3
import joblib
import pandas as pd

from sklearn.datasets import fetch_openml
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

        # Safe path handling for script / notebook / VS Code
        try:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.base_dir = os.getcwd()

        # Keep original file name
        self.filename = os.path.abspath(
            os.path.join(self.base_dir, filename)
        )

        print(f"📁 Working directory: {self.base_dir}")
        print(f"📂 Expected CSV path: {self.filename}")

        # Model configuration
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.label_encoder_y = LabelEncoder()
        self.df = None
        self.trained_pipeline = None
        self.last_result_df = None

    def download_data_if_missing(self):
        """
        Download dataset once from OpenML if missing
        """
        if os.path.isfile(self.filename):
            print("✅ CSV already exists. No download needed.")
            return True

        print("⬇️ CSV not found. Downloading from OpenML...")

        try:
            adult = fetch_openml(
                name="adult",
                version=2,
                as_frame=True
            )

            self.df = adult.frame

            # Save locally with same filename
            self.df.to_csv(self.filename, index=False)

            print(f"✅ Dataset downloaded successfully: {self.filename}")
            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def get_data(self):
        """
        Download once if missing, then load CSV
        """
        print("\n📥 Preparing dataset...")

        if not os.path.isfile(self.filename):
            if not self.download_data_if_missing():
                return False

        try:
            self.df = pd.read_csv(self.filename)

            if self.df.empty:
                print("❌ CSV file is empty.")
                return False

            print("✅ Data loaded successfully.")
            print(f"📊 Shape: {self.df.shape}")
            print(f"📌 Columns: {list(self.df.columns)}")

            return True

        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False

    def run_model(self):
        """
        Preprocess -> Train -> Evaluate
        """
        print("\n🚀 Running model...")

        if self.df is None:
            print("❌ No data loaded. Run get_data() first.")
            return False

        target_column = None

        if "income" in self.df.columns:
            target_column = "income"
        elif "Income" in self.df.columns:
            target_column = "Income"
        elif "class" in self.df.columns:
            target_column = "class"

        if target_column is None:
            print("❌ Target column not found.")
            return False

        try:
            X = self.df.drop(columns=[target_column]).copy()
            y_raw = self.df[target_column].copy()

            valid_mask = y_raw.notna()
            X = X.loc[valid_mask]
            y_raw = y_raw.loc[valid_mask]

            if X.empty:
                print("❌ No valid rows available.")
                return False

            y = self.label_encoder_y.fit_transform(y_raw)

            class_counts = pd.Series(y).value_counts()
            use_stratify = class_counts.min() >= 2

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if use_stratify else None
            )

            numeric_cols = X_train.select_dtypes(
                include=["number"]
            ).columns

            categorical_cols = X_train.select_dtypes(
                include=["object", "category", "bool"]
            ).columns

            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])

            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ])

            model_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", self.model)
            ])

            print("⏳ Training model...")
            model_pipeline.fit(X_train, y_train)

            self.trained_pipeline = model_pipeline

            predictions = model_pipeline.predict(X_test)

            print("\n" + "=" * 40)
            print("📈 MODEL PERFORMANCE REPORT")
            print("=" * 40)

            print(
                classification_report(
                    y_test,
                    predictions,
                    target_names=self.label_encoder_y.classes_,
                    zero_division=0
                )
            )

            result_df = X_test.copy()
            result_df["actual_income"] = \
                self.label_encoder_y.inverse_transform(y_test)

            result_df["predicted_income"] = \
                self.label_encoder_y.inverse_transform(predictions)

            self.last_result_df = result_df

            return True

        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False

    def save_to_sql(self, db_name="income_results.db"):
        """
        Save original data + predictions into SQLite
        """
        print("\n💾 Saving to SQLite...")

        if self.df is None:
            print("❌ No data to save.")
            return False

        db_path = os.path.abspath(
            os.path.join(self.base_dir, db_name)
        )

        try:
            with sqlite3.connect(db_path) as conn:
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

            print(f"✅ Saved to SQLite: {db_path}")
            return True

        except Exception as e:
            print(f"❌ SQL save failed: {e}")
            return False

    def save_model(self, model_name="income_model.pkl"):
        """
        Save trained pipeline
        """
        print("\n💾 Saving model...")

        if self.trained_pipeline is None:
            print("❌ No trained model available.")
            return False

        model_path = os.path.abspath(
            os.path.join(self.base_dir, model_name)
        )

        try:
            model_bundle = {
                "pipeline": self.trained_pipeline,
                "label_encoder": self.label_encoder_y
            }

            joblib.dump(model_bundle, model_path)

            print(f"✅ Model saved: {model_path}")
            return True

        except Exception as e:
            print(f"❌ Model save failed: {e}")
            return False

    def load_model(self, model_name="income_model.pkl"):
        """
        Load saved model
        """
        print("\n📤 Loading model...")

        model_path = os.path.abspath(
            os.path.join(self.base_dir, model_name)
        )

        if not os.path.isfile(model_path):
            print("❌ Model file not found.")
            return False

        try:
            model_bundle = joblib.load(model_path)

            self.trained_pipeline = model_bundle["pipeline"]
            self.label_encoder_y = model_bundle["label_encoder"]

            print(f"✅ Model loaded: {model_path}")
            return True

        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False

    def predict_income(self, new_data: pd.DataFrame):
        """
        Predict new income values
        """
        print("\n🔮 Predicting...")

        if self.trained_pipeline is None:
            print("❌ No trained model loaded.")
            return None

        try:
            preds = self.trained_pipeline.predict(new_data)
            return self.label_encoder_y.inverse_transform(preds)

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None


if __name__ == "__main__":
    analyst = IncomeAnalyst()

    if analyst.get_data():
        if analyst.run_model():
            analyst.save_to_sql()
            analyst.save_model()
