import argparse
import pandas as pd
import mlflow
import re
import ast
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. The Raw Log Data

def parse_logs_to_df(log_text):
    """
    Parses log lines to extract the dictionary containing features and predictions.
    """
    data_list = []
    

    pattern = r"Prediction made: (\{.*\})"
    
    for line in log_text.strip().split('\n'):
        match = re.search(pattern, line)
        if match:
  
            record = ast.literal_eval(match.group(1))
            data_list.append(record)
            
    return pd.DataFrame(data_list)



if __name__ == "__main__":
    
    argparse = argparse.ArgumentParser(description="Evaluate model predictions from log data.")
    argparse.add_argument("--log_file", type=str, required=True, help="Path to the log file")
    args = argparse.parse_args()
    

    
    with open(args.log_file, "r") as file:
        log_content = file.read()

    df = parse_logs_to_df(log_content)

    np.random.seed(42)
    df_raw = df.copy()
    df['ground_truth'] = np.random.randint(0, 3, df.shape[0])
    
    df_clean = df.drop(columns=['prediction'])

    print("--- Parsed Data for Evaluation ---")
    print(df.head())
    

    mlflow.set_experiment("Iris_Log_Evaluation")

    with mlflow.start_run(run_name="Log_Batch_Analysis"):
        

        
        result = mlflow.evaluate(
            data=df_clean,
            model="models:/irs/1",
            model_type="classifier", 
            targets="ground_truth", 
            evaluators=["default"],
            evaluator_config={
                "default": {
                    "log_confusion_matrix": True,
                    "log_roc_pr_curves": False
                }
            },

        )
        

        print("\n--- Evaluation Metrics Logged ---")
        print(result.metrics)
        
        # if result.metrics.get("score") < 0.7:
            
        #     print("retraining the model as accuracy is below threshold...")
            
        #     print('sending alert to team: Model accuracy below acceptable threshold!')
            
        #     df_clean.to_csv("iris_retrain_data.csv", index=False)
            
        #     mlflow.run(
        #         "https://github.com/Orianne-B/mlops-with-mlflow.git",
        #         parameters={
        #             "input_data": "/home/kfondio/Documents/mlflow/mlops-with-mlflow/iris_retrain_data.csv",
        #             "processed_data_folder": "/home/kfondio/Documents/mlflow/mlops-with-mlflow/"
        #         },
        #         env_manager="local"
        #     )
        
        # Evidently AI
        #https://github.com/evidentlyai/evidently
        train_df = pd.read_csv("data/raw/iris.csv")
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=train_df, current_data=df_raw)
        result.save_html("evidently_report.html")
        mlflow.log_artifact("evidently_report.html")
        
        # 4. Sauvegarder ou Alerter
        drift_score = result.json()['metrics'][0]['result']['dataset_drift']
        if drift_score > 0.5:
            print("Drift Majeur Détecté !")
        
        
            
            
        print("\nCheck your MLflow UI (usually http://localhost:8080) to see the Confusion Matrix.")