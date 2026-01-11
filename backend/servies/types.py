from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document


@dataclass
class ModalChunks:
    texts: List[Document]
    tables: List[Document]  # tables stored as HTML in page_content
    images: List[Document]  # images stored as base64 in page_content
