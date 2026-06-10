from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(
    url="https://998f6d30-51d1-4174-92b0-e5954fd445b5.eu-west-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MzM0NjdjZjEtNzU3YS00Mjc5LWE3MmQtNjUzYmVjMGMwZjVlIn0.ml0lrkM8aUUt8pbIZiao0Vio-HHPnEAmgVK3M1lP7rU"
)

client.delete_collection("rag_spanish")
client.delete_collection("rag_english")  # clean up old one too

client.create_collection(
    collection_name="rag_spanish",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
print("✅ rag_spanish recreated empty. Ready to ingest.")