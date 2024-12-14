import os
import json
import vertexai
from dotenv import load_dotenv
from tqdm import tqdm
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from simple_gcp_rag.utils.gcp import download_documents_from_gcp, upload_file


class DatabaseDeployer:
    def __init__(self, char_limit: int = 2000, docs_per_batch: int = 10):
        load_dotenv(dotenv_path=os.path.join(
            os.path.dirname(__file__), ".env"))

        self.project_id: str = os.getenv("PROJECT_ID")
        self.region: str = os.getenv("REGION")
        self.bucket_name: str = os.getenv("BUCKET_NAME")
        self.documents_folder: str = os.getenv("DOCUMENTS_FOLDER")
        self.emb_folder: str = os.getenv("EMB_FOLDER")
        self.index_name: str = os.getenv("INDEX_NAME")
        self.endpoint_name: str = os.getenv("ENDPOINT_NAME")
        self.emb_model_name: str = os.getenv("EMB_MODEL_NAME")
        self.emb_size: int = int(os.getenv("EMB_SIZE"))
        self.emb_neighbors: int = int(os.getenv("EMB_NEIGHBORS"))

        self.char_limit = char_limit
        self.docs_per_batch = docs_per_batch

        self.initialize_vertexai()

    @property
    def emb_folder_path(self) -> str:
        return os.path.join(f"gs://{self.bucket_name}", self.emb_folder)

    def initialize_vertexai(self) -> None:
        vertexai.init(project=self.project_id, location=self.region)

    def download_documents(self) -> None:
        print("Downloading documents...")
        self.documents = download_documents_from_gcp(
            self.bucket_name, self.documents_folder, self.char_limit)
        print(f"There are {len(self.documents)} documents in the database.")

    def generate_document_embeddings(self) -> None:
        print("Encoding documents...")
        self.emb_model = TextEmbeddingModel.from_pretrained(
            self.emb_model_name)
        embeddings = []
        for i in tqdm(range(0, len(self.documents), self.docs_per_batch)):
            doc_emb_inputs = [TextEmbeddingInput(
                task_type='', title='', text=doc['text']) for doc in self.documents[i:i+self.docs_per_batch]]
            embeddings += self.emb_model.get_embeddings(doc_emb_inputs)

        with open('emb.json', 'w') as f:
            for doc, emb in zip(self.documents, embeddings):
                f.write(json.dumps(
                    {"id": doc['name'], "embedding": emb.values}) + '\n')

        upload_file(self.bucket_name, "emb.json",
                    f"{self.emb_folder}/emb.json")

    def create_and_deploy_index(self) -> None:
        print("Creating and deploying index...")
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.index_name,
            contents_delta_uri=self.emb_folder_path,
            dimensions=self.emb_size,
            approximate_neighbors_count=self.emb_neighbors,
        )

        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(display_name=self.endpoint_name,
                                                                       public_endpoint_enabled=True)
        index_endpoint.deploy_index(
            index=index, deployed_index_id=self.index_name)

    def deploy(self) -> None:
        self.download_documents()
        self.generate_document_embeddings()
        self.create_and_deploy_index()


if __name__ == "__main__":
    deployer = DatabaseDeployer()
    deployer.deploy()
