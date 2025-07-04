import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from app.services.base_vector_db import BaseVectorDB

class ExerciseVectorDB(BaseVectorDB):
    """
    A vector database for storing and searching exercise-related data.
    """
    def __init__(self, model_name='jhgan/ko-sbert-sts',
                 index_path='data/exercise_index.idx',
                 meta_path='data/exercise_meta.pkl'):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata = []
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        self.load_index()


    def _encode(self, texts):
      """
      Encode a list of texts into vectors.
      """
      return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    

    def build_index(self, documents):
      texts = [
        f"{doc['description']} 난이도는 {doc['difficulty']} 부위는 {doc['body_part']}"
        for doc in documents
      ]

      self.metadata = [doc['name'] for doc in documents]
      vectors = self._encode(texts)

      self.index = faiss.IndexFlatIP(self.vector_dim)
      self.index.add(vectors)

      faiss.write_index(self.index, self.index_path)
      with open(self.meta_path, 'wb') as f:
         pickle.dump(self.metadata, f)


    def load_index(self):
       if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
          self.index = faiss.read_index(self.index_path)
          with open(self.meta_path, 'rb') as f:
             self.metadata = pickle.load(f)


    def search(self, query: str, top_k: int = 1):
      vector = self._encode([query])
      D, I = self.index.search(vector, top_k)
      return [self.metadata[i] for i in I[0]]
