import os
import vertexai
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, ChatSession
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from simple_gcp_rag.utils.gcp import retrieve_document


class RagChat:
    def __init__(self, retrieved_doc_char_limit: int = 5000) -> None:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"),
                    override=True)

        self.project_id: str = os.getenv("PROJECT_ID")
        self.region: str = os.getenv("REGION")
        self.bucket_name: str = os.getenv("BUCKET_NAME")
        self.endpoint_id: str = os.getenv("ENDPOINT_ID")
        self.index_name: str = os.getenv("INDEX_NAME")
        self.documents_folder: str = os.getenv("DOCUMENTS_FOLDER")
        self.emb_model_name: str = os.getenv("EMB_MODEL_NAME")
        self.emb_neighbors: int = int(os.getenv("EMB_NEIGHBORS"))
        self.chat_model: str = os.getenv("CHAT_MODEL")

        self.retrieved_doc_char_limit = retrieved_doc_char_limit

        self.init_models()

    def init_models(self) -> None:
        vertexai.init(project=self.project_id, location=self.region)

        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            self.endpoint_id)
        self.emb_model = TextEmbeddingModel.from_pretrained(
            self.emb_model_name)
        self.chat_model = GenerativeModel(self.chat_model)

    def retrieve_best_document(self, prompt: str) -> str:
        query = TextEmbeddingInput(task_type='', title='', text=prompt)
        query_emb = self.emb_model.get_embeddings([query])
        response = self.index_endpoint.find_neighbors(deployed_index_id=self.index_name,
                                                      queries=[
                                                          query_emb[0].values],
                                                      num_neighbors=self.emb_neighbors)

        best_document_path: str = response[0][0].id

        best_document = retrieve_document(path=best_document_path,
                                          bucket_name=self.bucket_name,
                                          char_limit=self.retrieved_doc_char_limit)

        return best_document

    def get_chat_response(self, chat: ChatSession, prompt: str) -> str:
        text_response = []
        responses = chat.send_message(prompt, stream=True)
        for chunk in responses:
            text_response.append(chunk.text)
        return "".join(text_response)

    def send_prompt(self, prompt: str) -> str:
        best_document = self.retrieve_best_document(prompt)
        chat_session = self.chat_model.start_chat()
        prompt = "Assuming following context is true, answer the question in the question's language:" +  \
                 f"\n\n<context>\n{best_document}\n</context>\n\n" + \
                 f"QUESTION: {prompt}"
        response = self.get_chat_response(chat_session, prompt)

        return response
