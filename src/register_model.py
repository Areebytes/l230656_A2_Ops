"""
VisionGuard - Task 3.6: register the two best manual runs as model versions,
give the best one the alias 'champion', and store serving settings as version tags.
"""
import mlflow
from mlflow import MlflowClient

from tracking import EXPERIMENT_NAME, REGISTERED_MODEL, setup_mlflow


def main():
    setup_mlflow()
    client = MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED' and tags.tracking_method = 'manual'",
        order_by=["metrics.final_val_roc_auc DESC"])
    runs = runs[runs["tags.model_uri"].notna()]
    if len(runs) < 2:
        raise SystemExit("Need at least 2 finished manual runs with a logged model.")

    best, runner_up = runs.iloc[0], runs.iloc[1]
    versions = {}
    # Register runner-up first, best last -> best gets the higher version number
    for row in (runner_up, best):
        mv = mlflow.register_model(row["tags.model_uri"], REGISTERED_MODEL)
        # Serving settings travel WITH the model version
        for key, value in {"run_name": row["tags.mlflow.runName"],
                           "threshold": row["metrics.decision_threshold"],
                           "img_size": row["params.img_size"],
                           "resize_mode": row["params.resize_mode"],
                           "val_auc": round(row["metrics.final_val_roc_auc"], 4)}.items():
            client.set_model_version_tag(REGISTERED_MODEL, mv.version, key, str(value))
        versions[row["tags.mlflow.runName"]] = mv.version
        print(f"Registered {row['tags.mlflow.runName']} as version {mv.version}")

    champion_version = versions[best["tags.mlflow.runName"]]
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", champion_version)
    client.update_registered_model(
        REGISTERED_MODEL, description="VisionGuard AI-vs-real classifier. "
                                      "Output = P(AI); compare with the version's 'threshold' tag.")
    print(f"\nAlias 'champion' -> version {champion_version} ({best['tags.mlflow.runName']})")


if __name__ == "__main__":
    main()