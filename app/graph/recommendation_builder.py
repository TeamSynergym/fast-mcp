from langgraph.graph import StateGraph, END
from .state import RecommendationState
from .recommendation_nodes import (
    summarize_user_node,
    vector_search_node,
    generate_reason_node
)

def create_recommendation_graph():
    builder = StateGraph(RecommendationState)

    builder.add_node("user_summarizer", summarize_user_node)
    builder.add_node("vector_searcher", vector_search_node)
    builder.add_node("reason_generator", generate_reason_node)

    builder.set_entry_point("user_summarizer")
    builder.add_edge("user_summarizer", "vector_searcher")
    builder.add_edge("vector_searcher", "reason_generator")
    builder.add_edge("reason_generator", END)
    
    return builder.compile()