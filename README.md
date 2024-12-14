# Environment set up
1. Install Python 3.10
2. `pip install poetry`
3. `poetry shell`
4. `poetry install`
5. `gcloud init` / make sure you have gcloud properly configured
6. `gcloud auth application-default login`

# Deploy your database
1. Upload documents to a GCS "folder"
2. Set up your GCP parameters in the `simple_gcp_rag/.env` file
3. Run the `python simple_gcp_rag/db_deploy.py`
4. Update ENDPOINT_ID in the .env file

# Chat
1. Open `chat.ipynb` notebook.
2. Send some prompts ;)
