from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Milvus
from langchain.schema import Document

# Lade die PDF-Datei und splitte sie in einzelne Seiten
loader = PyPDFLoader("SMA.pdf")
pages = loader.load_and_split()

# Lade vortrainiertes Model für die Embedding-Generierung
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Funktion zum Generieren der Embeddings, die die embed_documents und embed_query Methoden benötigt
class EmbeddingModel:
    def embed_documents(self, texts):
        return embedding_model.encode(texts)
    
    def embed_query(self, query):
        return embedding_model.encode([query])[0]

# Erstelle Dokumente aus den Seiten der PDF
documents = [Document(page_content=page.page_content) for page in pages]

# Erstelle eine Instanz des Embedding-Modells
embedding = EmbeddingModel()

# Erstelle den Vektorstore in Milvus mit den generierten Embeddings
vectorstore = Milvus.from_documents(documents, embedding, connection_args={"host": "localhost", "port": "19530"})

# Definiere die Abfrage, die du durchführen möchtest
query = "Kannst du mir die verschiedenen Bereiche der Systematik nach „What, Why and How Much“ von Kaushik A. nennen und welche Tools in diesen Bereichen verwendet werden?"

# Führe die Ähnlichkeitssuche aus
docs = vectorstore.similarity_search(query, k=5)

# Gib die gefundenen Dokumentinhalte aus
for doc in docs:
   print(doc.page_content)




