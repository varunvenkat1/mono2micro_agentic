import os
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# from chromadb_client import ChromaDBClient
from chunk_code import extract_code_chunks
# from qwencoderllm import analyse_file


class Embedding:

    def __init__(self):
        self.metadata_store = {}
        # self.db_client = ChromaDBClient(embed_model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize the embedding model
        embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize graph
        self.graph = nx.Graph()

    async def add_dotnet_codebase_embeddings(self, name, directory, language, similarity_threshold=0.75):
        """Walk through a .NET project, chunk and embed code, store nodes in graph, then build edges."""
        # Step 1: Add nodes with embeddings
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith("." + language):
                    file_path = os.path.join(root, file)

                    # Read file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()

                    # If C#, split into chunks (else keep whole file)
                    chunks = [code]
                    if language == "cs":
                        chunks = extract_code_chunks(code)

                    # Encode each chunk and store as a node
                    for idx, chunk in enumerate(chunks):
                        embedding = self.embedding_model.encode(chunk)

                        node_id = f"{file_path}#chunk{idx}"
                        self.graph.add_node(
                            node_id,
                            embedding=embedding,
                            content=chunk,
                            file=file,
                        )

        # Step 2: Build relationships between nodes
        nodes = list(self.graph.nodes(data=True))

        # 1. Connect files in the same directory
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes):
                if i < j:
                    dir1 = os.path.dirname(node1.split("#")[0])
                    dir2 = os.path.dirname(node2.split("#")[0])
                    if dir1 == dir2:  # same folder
                        self.graph.add_edge(node1, node2, relation="same_folder")

        # 2. Connect based on semantic similarity
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes):
                if i < j:
                    sim = cosine_similarity(
                        data1["embedding"].reshape(1, -1),
                        data2["embedding"].reshape(1, -1),
                    )[0][0]
                    if sim > similarity_threshold:
                        self.graph.add_edge(
                            node1, node2, relation="semantic_similarity", weight=sim
                        )

    @staticmethod
    def GetFileContent(file_path):
        """
        Read the content of a file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def GetRelevantContext(self, file_content, top_k=5, hops=1):
        """
        Retrieve relevant context using similarity + graph traversal.
        """
        query_embedding = self.embedding_model.encode(file_content).reshape(1, -1)

        # Initial ranking by similarity
        similarities = {
            node: cosine_similarity(
                query_embedding, data["embedding"].reshape(1, -1)
            )[0][0]
            for node, data in self.graph.nodes(data=True)
        }

        # Take top_k seeds
        top_nodes = sorted(similarities, key=similarities.get, reverse=True)[:top_k]

        # Expand into neighbors (graph traversal)
        expanded_nodes = set(top_nodes)
        for node in top_nodes:
            for neighbor in nx.single_source_shortest_path_length(
                self.graph, node, cutoff=hops
            ).keys():
                expanded_nodes.add(neighbor)

        # Gather content
        context_contents = [self.graph.nodes[n]["content"] for n in expanded_nodes]

        return "\n".join(context_contents)
