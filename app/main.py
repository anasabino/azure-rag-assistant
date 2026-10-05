import os
from functools import lru_cache

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from fastapi import FastAPI
from openai import AzureOpenAI
from pydantic import BaseModel

app = FastAPI(title="Azure RAG Assistant")


class Question(BaseModel):
    question: str


@lru_cache
def get_openai() -> AzureOpenAI:
    token = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token,
        api_version="2024-10-21",
    )


@lru_cache
def get_search() -> SearchClient:
    return SearchClient(
        os.environ["AZURE_SEARCH_ENDPOINT"],
        os.getenv("AZURE_SEARCH_INDEX", "docs"),
        DefaultAzureCredential(),
    )


def retrieve(query: str, k: int = 4) -> list[dict]:
    emb = get_openai().embeddings.create(
        model=os.environ["EMBEDDING_DEPLOYMENT"], input=query
    ).data[0].embedding
    results = get_search().search(
        search_text=query,  # busca híbrida: keyword + vetor
        vector_queries=[
            VectorizedQuery(vector=emb, k_nearest_neighbors=k, fields="embedding")
        ],
        top=k,
    )
    return [{"content": r["content"], "source": r["source"]} for r in results]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question):
    docs = retrieve(body.question)
    context = "\n\n".join(
        f"[{i + 1}] ({d['source']}) {d['content']}" for i, d in enumerate(docs)
    )
    resp = get_openai().chat.completions.create(
        model=os.environ["CHAT_DEPLOYMENT"],
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "Responda só com base no contexto. Cite as fontes como [1], [2]. "
                "Se não souber, diga que não encontrou.",
            },
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {body.question}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [d["source"] for d in docs],
    }