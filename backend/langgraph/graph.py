"""Main LangGraph graph definition for the RAG agent."""

import logging
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.core.dependency import get_model, get_settings
from backend.langgraph.state import RAGState
from backend.langgraph.tools import rag_tools, _get_chat_service

logger = logging.getLogger(__name__)

# System prompt for the RAG agent
SYSTEM_PROMPT = """You are a helpful RAG (Retrieval-Augmented Generation) assistant that answers questions using a knowledge base of PDF documents.

Your capabilities:
1. **ingest_document**: Upload and process PDF documents into the knowledge base
2. **search_knowledge_base**: Search for relevant information in uploaded documents

Guidelines:
- When a user asks a question, ALWAYS use the search_knowledge_base tool first to find relevant context
- Base your answers ONLY on the information retrieved from the knowledge base
- If no relevant information is found, clearly state that you don't have that information in the current knowledge base
- Be concise but comprehensive in your answers
- If the user asks to upload or process a document, use the ingest_document tool

Current knowledge base status: Ready to receive queries and documents."""


async def agent_node(state: RAGState, config: RunnableConfig) -> dict:
    """Main agent node that processes messages and decides on actions.

    This node invokes the LLM with the current conversation and available tools.
    The LLM can either respond directly or call tools for RAG operations.
    """
    cfg = get_settings()
    model_service = get_model(cfg)
    llm = model_service.get_chat_model()

    # Bind tools to the model
    llm_with_tools = llm.bind_tools(rag_tools)

    # Prepare messages with system prompt
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(state["messages"])

    # Invoke the model
    response = await llm_with_tools.ainvoke(messages, config)

    return {"messages": [response]}


def should_continue(state: RAGState) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end.

    If the last message has tool calls, route to the tools node.
    Otherwise, end the conversation turn.
    """
    messages = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


def create_rag_graph() -> StateGraph:
    """Create and compile the RAG agent graph.

    The graph has the following structure:
    - agent: Main node that invokes the LLM
    - tools: ToolNode that executes tool calls
    - Conditional edge from agent to either tools or END
    - Edge from tools back to agent
    """
    # Create the graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(rag_tools))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    return workflow


# Create the compiled graph
rag_graph = create_rag_graph()
graph = rag_graph.compile()
