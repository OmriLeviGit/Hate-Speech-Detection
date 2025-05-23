from huggingface_hub import create_repo, upload_folder, delete_repo
from huggingface_hub.errors import RepositoryNotFoundError

try:
    delete_repo(repo_id="olmervii/hatespeech-bert-uncased")
except RepositoryNotFoundError:
    print("Repo doesn't exist, creating new one")

create_repo(
   repo_id="olmervii/hatespeech-bert-uncased",
   repo_type="model",
   private=False
)

upload_folder(
   folder_path="final_model/",
   repo_id="olmervii/hatespeech-bert-uncased",
)