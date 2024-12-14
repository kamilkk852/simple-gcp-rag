from google.cloud import storage
from tqdm import tqdm


def download_documents_from_gcp(bucket_name: str, documents_folder: str, char_limit: int = None) -> list[dict]:
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(
        bucket_name, prefix=documents_folder, delimiter=None)

    documents: list[dict] = []
    for blob in tqdm(blobs):
        if not blob.name or not blob.name.endswith('.txt'):
            continue

        text = blob.download_as_string().decode('utf-8')
        if char_limit:
            text = text[:char_limit]

        documents.append({"name": blob.name, "text": text})

    return documents


def retrieve_document(path: str, bucket_name: str, char_limit: int = None) -> str:
    storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    text = blob.download_as_string().decode('utf-8')
    if char_limit:
        text = text[:char_limit]

    return text


def upload_file(bucket_name: str,
                source_file_name: str,
                destination_blob_name: str,
                replace: bool = True) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    if replace and blob.exists():
        blob.delete()

    generation_match_precondition = 0
    blob.upload_from_filename(
        source_file_name, if_generation_match=generation_match_precondition)
