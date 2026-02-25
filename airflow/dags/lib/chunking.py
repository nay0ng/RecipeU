from typing import List
from langchain_core.documents import Document

MAX_CHUNKING_LENGTH = 5000


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    길이가 긴 문서만 청킹
    """
    chunked = []

    for doc in documents:
        if len(doc.page_content) <= MAX_CHUNKING_LENGTH:
            chunked.append(doc)
            continue

        text = doc.page_content
        size = 1000
        overlap = 200

        start = 0
        idx = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end]

            md = dict(doc.metadata)
            md["chunk_index"] = idx

            chunked.append(Document(page_content=chunk_text, metadata=md))

            start = end - overlap
            idx += 1

    return chunked
