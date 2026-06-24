import os
import mlflow
import dagshub
from mlflow import MlflowClient


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

    client = MlflowClient()
    model_name = "my_model"

    # Get all versions of the model
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise Exception(f"No versions found for model '{model_name}'")

    # Latest version
    latest_version = max(
        versions,
        key=lambda v: int(v.version)
    )

    print(f"Latest Version Found: {latest_version.version}")

    # Archive current Production versions
    for version in versions:
        if getattr(version, "current_stage", None) == "Production":
            print(f"Archiving Version {version.version}")

            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

    # Promote latest version to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production"
    )

    print(
        f"Model version {latest_version.version} "
        f"promoted to Production"
    )


if __name__ == "__main__":
    promote_model()