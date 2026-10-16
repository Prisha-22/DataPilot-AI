from google.genai import types

from src.llm.gemini_client import client


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


def create_document_embedding(text, title=None):
    """
    Create an embedding for a single knowledge-base document.
    """

    if title is None:
        title = "none"

    formatted_text = (
        f"title: {title} | text: {text}"
    )

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=formatted_text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    return response.embeddings[0].values


def create_document_embeddings(texts, titles=None):
    """
    Create separate embeddings for multiple knowledge-base
    documents in one API request.
    """

    if titles is None:
        titles = ["none"] * len(texts)

    if len(texts) != len(titles):
        raise ValueError(
            "texts and titles must have the same length."
        )

    contents = []

    for text, title in zip(texts, titles):

        formatted_text = (
            f"title: {title} | text: {text}"
        )

        contents.append(
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=formatted_text
                    )
                ]
            )
        )

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    if response.embeddings is None:
        raise ValueError(
            "Gemini returned no embeddings."
        )

    return [
        embedding.values
        for embedding in response.embeddings
    ]

def create_query_embedding(query):
    """
    Create an embedding for a user's retrieval query.
    """

    formatted_query = (
        f"task: search result | query: {query}"
    )

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=formatted_query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    return response.embeddings[0].values


def create_embedding(text):
    """
    Backward-compatible helper for document embeddings.
    """

    return create_document_embedding(text)