from huggingface_hub import HfApi


MODEL_PATH = "./qwen3-imdb-final"
REPO_ID = "worldboss/qwen3-0.6b-imdb"


api = HfApi()


api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,
)


api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
)


print(f"Model uploaded to: {REPO_ID}")
