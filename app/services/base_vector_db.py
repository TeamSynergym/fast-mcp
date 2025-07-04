from abc import ABC, abstractmethod

class BaseVectorDB(ABC):
    """
    Abstract base class for vector databases.
    """


    @abstractmethod
    def build_index(self, documents):
        pass
    

    @abstractmethod
    def search(self, query: str, top_k: int = 1):
        pass