import uuid
import requests
from opensearchpy import OpenSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
import os


class VectorStoreManager:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        index_name: str = "rag_docs"
    ):
        load_dotenv() 

        self.host = host or os.getenv("OPENSEARCH_HOST", "localhost")
        self.port = port or int(os.getenv("OPENSEARCH_PORT", 9200))
        self.user = os.getenv("OPENSEARCH_USER", "admin")
        self.password = os.getenv("OPENSEARCH_PASS", "")

        self.index_name = index_name

        self.client = OpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_auth=(self.user, self.password),
            use_ssl=True,
            verify_certs=True,
            ca_certs="./root-ca.pem"
        )

        try:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(
                    index=index_name,
                    body={
                        "settings": {"index.knn": True},
                        "mappings": {
                            "properties": {
                                "content": {"type": "text"},
                                "vector": {"type": "knn_vector", "dimension": 1024}
                            }
                        },
                    },
                )
        except Exception as e:
            print("OpenSearch connection failed:", e)
            raise SystemExit("Please check OpenSearch and try again.")


        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.ollama_url = "http://localhost:11434/api/embed"
        self.embedding_model = "mxbai-embed-large"

    def _get_embedding(self, text: str) -> List[float]:
        try:
            resp = requests.post(
                self.ollama_url,
                json={"model": self.embedding_model, "input": text},
                timeout=30
            )
            resp.raise_for_status()
            return resp.json().get("embeddings", [[]])[0]
        except Exception as e:
            print("Embedding error:", e)
            return []

    def ingest(self, content: str, metadata: Dict[str, Any]):
        chunks = self.splitter.split_text(content)
        print(f"Splitting into {len(chunks)} chunks…")

        for chunk in tqdm(chunks, desc="Indexing chunks"):
            vector = self._get_embedding(chunk)
            if not vector:
                continue

            doc_id = str(uuid.uuid4())
            body = {
                "content": chunk,
                "vector": vector,
                **metadata
            }
            self.client.index(index=self.index_name, id=doc_id, body=body)

    def rebuild_index(self, docs: List[Dict[str, Any]]):
        self.client.indices.delete(index=self.index_name, ignore_unavailable=True)
        self.client.indices.create(
            index=self.index_name,
            body={
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "vector": {"type": "knn_vector", "dimension": 1024}
                    }
                },
            },
        )
        for doc in docs:
            self.ingest(doc["content"], doc["metadata"])

    def update_delta(self, docs: List[Dict[str, Any]]):
        for doc in docs:
            self.ingest(doc["content"], doc["metadata"])

    def search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_vec = self._get_embedding(query_text)
        if not query_vec:
            return []

        body = {
            "size": limit,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vec,
                        "k": limit
                    }
                }
            }
        }

        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for h in hits:
            src = h["_source"].get("source", "Unknown")
            results.append({
                "content": h["_source"]["content"],
                "score": h["_score"],
                "source": src
            })
        return results

