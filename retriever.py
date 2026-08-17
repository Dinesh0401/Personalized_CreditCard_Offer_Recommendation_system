
import chromadb
from sentence_transformers import SentenceTransformer

class CardRetriever:
    def __init__(self, collection_name='flagship_cards'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)

    def add_cards(self, cards_df):
        documents = []
        metadatas = []
        ids = []
        for idx, row in cards_df.iterrows():
            text = f"Card: {row['card_name']}. Issuer: {row['issuer']}. Fee: {row['annual_fee']}. "                   f"Rewards: {row['primary_reward']} at {row['reward_rate']*100}%. Description: {row['description']}"
            documents.append(text)
            metadatas.append({"card_id": row['card_id'], "fee": int(row['annual_fee'])})
            ids.append(row['card_id'])
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, text, top_k=3):
        return self.collection.query(query_texts=[text], n_results=top_k)
