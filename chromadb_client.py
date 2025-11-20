import chromadb
from chromadb.utils import embedding_functions

class ChromaDBClient:
    def __init__(self, embed_model_name, db_path = "vector_store"):
        self.client = chromadb.PersistentClient(path = db_path)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = embed_model_name
        )
        self.embed_function = sentence_transformer_ef
        
    def create_collection(self, name, metadata={}):
        return self.client.create_collection(name = name, metadata=metadata, embedding_function=self.embed_function)
    
    def add_text(self, collection_name, document_id, text, metadata={}):
        collections = self.client.list_collections()
        collection_names = [c.name for c in collections]
        if collection_name not in collection_names: 
            self.create_collection(collection_name, metadata)
        collection = self.client.get_collection(name = collection_name)
        collection.add(ids = [document_id], documents=[text], metadatas=metadata)
        
    """def add_text(self, collection_name, document_id, text, metadata={}):
        if collection_name not in self.client.list_collections():
            self.create_collection(collection_name, metadata)
        collection = self.client.get_collection(name = collection_name)
        collection.add(ids = [document_id], documents=[text], metadatas=metadata)
    """
    
    def query_text(self, collection_name, query_text, n_results = 5):
        collection = self.client.get_collection(name=collection_name)
        return collection.query(query_texts=[query_text], n_results=n_results)
    
    def get_collection_data(self, collection_name):
        collection = self.client.get_collection(name=collection_name)
        return collection.get(include=["documents", "metadatas"])
