# promote model

import os
import mlflow
import dagshub
def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")

    print("Token Exists:", bool(dagshub_token))
    print("Token Length:", len(dagshub_token) if dagshub_token else 0)

    dagshub.auth.add_app_token(dagshub_token)

    dagshub.init(
        repo_owner="DataWithPdeep",
        repo_name="ML_Production_pipeline_capston_pro",
        mlflow=True
    )

    print("Tracking URI:", mlflow.get_tracking_uri())

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()