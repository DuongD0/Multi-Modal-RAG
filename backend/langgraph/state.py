"""State definitions for the RAG LangGraph agent."""

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """State for the RAG agent graph.

    Attributes:
        messages: Conversation history with add_messages reducer for proper handling.
        retrieved_docs: Documents retrieved from the knowledge base for context.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: Optional[List[dict]]
