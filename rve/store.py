import chromadb
from .embed import OfflineEmbedding
def build_store(docs, path=None):
    client=chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
    col=client.get_or_create_collection('nordwind', metadata={'hnsw:space':'cosine'}, embedding_function=OfflineEmbedding())
    col.upsert(ids=[d['id'] for d in docs], documents=[d['text'] for d in docs]); return col
def query(collection, question, k=3):
    r=collection.query(query_texts=[question], n_results=k)
    return [{'id':i,'text':t,'distance':float(x)} for i,t,x in zip(r['ids'][0],r['documents'][0],r['distances'][0])]
