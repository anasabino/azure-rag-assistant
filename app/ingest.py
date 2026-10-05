import os
import pathlib

from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration, SearchField, SearchFieldDataType,
    SearchIndex, SearchableField, SimpleField, VectorSearch, VectorSearchProfile,
)

from app.main import get_openai, get_search

INDEX = os.getenv("AZURE_SEARCH_INDEX", "docs")


def create_index():
    client = SearchIndexClient(os.environ["AZURE_SEARCH_ENDPOINT"], DefaultAzureCredential())
    client.create_or_update_index(SearchIndex(
        name=INDEX,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # text-embedding-3-small
                vector_search_profile_name="vp",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="vp", algorithm_configuration_name="hnsw")],
        ),
    ))


def chunks(text: str, size: int = 1000):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def main():
    create_index()
    batch = []
    for path in pathlib.Path("docs").glob("*.md"):
        for n, chunk in enumerate(chunks(path.read_text(encoding="utf-8"))):
            emb = get_openai().embeddings.create(
                model=os.environ["EMBEDDING_DEPLOYMENT"], input=chunk
            ).data[0].embedding
            batch.append({"id": f"{path.stem}-{n}", "content": chunk,
                          "source": path.name, "embedding": emb})
    get_search().upload_documents(batch)
    print(f"{len(batch)} chunks indexados")


if __name__ == "__main__":
    main()