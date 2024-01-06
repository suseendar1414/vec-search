import pymongo
import requests
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Environment variables
mongo_db_uri = os.getenv('MONGO_DB_URI')
hf_token = os.getenv('HF_TOKEN')

# MongoDB connection
client = pymongo.MongoClient(mongo_db_uri)
db = client.sample_mflix
collection = db.movies

embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:

    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise ValueError(f"request failed with statuscode {response.status_code}: {response.text}")
    
    return response.json()


# # print(generate_embedding("suseendar is awesome"))

# for doc in collection.find({'plot':{"$exists": True }}).limit(50):
#     doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#     collection.replace_one({'_id': doc['_id']},doc)

query = "imaginary character from the outer space"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
]);

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')