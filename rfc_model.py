import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class RFC_Model:
    def __init__(self, torii_path : str, 
                 cc_path : str, 
                 normal_path : str):
        """
        Load the data from the csv files
        """
        self.df_torii = pd.read_csv(torii_path)
        self.df_cc = pd.read_csv(cc_path)
        self.df_normal = pd.read_csv(normal_path)

    def preprocess_data(self):
        """
        Preprocess and cleans the data, assign labels 1 for malicious and 0 for normal traffic, encode the protocol
        """
        self.df_torii["label"] = 1
        self.df_cc["label"] = 1
        self.df_normal["label"] = 0
        self.df_anon = pd.concat([self.df_torii, self.df_cc], ignore_index=True)
        self.df_all = self.balance_dataset(self.df_anon, self.df_normal)
        self.df_all.drop(columns=["Flow_ID", "Src_IP", "Dst_IP", "Src_Port", 
                                  "Dst_Port", "Timestamp", "Label", "Sub_Cat"], inplace=True, errors='ignore')
        self.df_all.fillna(0, inplace=True)
        self.df_all = self.df_all[~self.df_all.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)

        
    def balance_dataset(self, malicious_df : pd.DataFrame, normal_df : pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset by sampling the minority class to match the number of samples in the majority class
        """
        rows_malicious = malicious_df.shape[0]
        rows_normal = normal_df.shape[0]
        if rows_malicious > rows_normal:
            malicious_df = malicious_df.sample(n=rows_normal)
        else:
            normal_df = normal_df.sample(n=rows_malicious)
        return pd.concat([malicious_df, normal_df], ignore_index=True)

    def create_and_evaluate_model(self) -> tuple[str, np.ndarray, float]:
        """
        Create and evaluate the model, returns the classification report, confusion matrix and roc auc score
        """
        y = self.df_all['label']
        X = self.df_all.drop(columns=['label']) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_probs = self.model.predict_proba(X_test)[:, 1]
        return classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred), roc_auc_score(y_test, y_probs)

    def save_model(self) -> None:
        with open("Column_Names.txt", "w") as file:
            for col in self.df_all.columns:
                if col != "label":
                    file.write(f"{col}\n")
        joblib.dump(self.model, "rfc_model.pkl")


if __name__ == "__main__":
    rfc_model = RFC_Model(normal_path='data/IoT-23_Normal.csv',
                          torii_path='data/IoT-23_Torii.csv',
                          cc_path='data/IoT-23_C&C.csv')
    rfc_model.preprocess_data()
    classification_report, confusion_matrix, roc_auc_score = rfc_model.create_and_evaluate_model()
    print("Classification Report :\n", classification_report)
    print("Confusion Matrix :\n", confusion_matrix)
    print("ROC AUC Score :", roc_auc_score)
    rfc_model.save_model() if roc_auc_score > 0.999 else print("Model is not good enough")
