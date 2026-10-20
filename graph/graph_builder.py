from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.broken_promises_agent import broken_promises_node
from agents.missing_results_agent import missing_results_node
from agents.pattern_finder_agent import pattern_finder_node
from agents.side_effect_agent import side_effect_node
from agents.supervisor import supervisor_compile, supervisor_route
from agents.timeline_agent import timeline_node
from agents.track_record_agent import track_record_node
from config.logging_config import setup_logging
from graph.state import MosaicState
from tools.clinical_tools import ALL_CLINICAL_TOOLS
from tools.pubmed_tools import ALL_PUBMED_TOOLS
from tools.search_tools import ALL_SEARCH_TOOLS


logger = setup_logging(__name__)

ALL_TOOLS = (
    ALL_SEARCH_TOOLS
    + ALL_CLINICAL_TOOLS
    + ALL_PUBMED_TOOLS
)


def build_mosaic_graph():
    """Build and compile the MOSAIC agent graph."""

    logger.info("Building MOSAIC agent graph...")

    graph = StateGraph(MosaicState)
    tool_node = ToolNode(ALL_TOOLS)

    graph.add_node("supervisor_route", supervisor_route)
    graph.add_node("broken_promises", broken_promises_node)
    graph.add_node("missing_results", missing_results_node)
    graph.add_node("track_record", track_record_node)
    graph.add_node("pattern_finder", pattern_finder_node)
    graph.add_node("side_effect", side_effect_node)
    graph.add_node("timeline", timeline_node)
    graph.add_node("tools", tool_node)
    graph.add_node("supervisor_compile", supervisor_compile)

    graph.add_edge(START, "supervisor_route")

    # All specialist agents run in parallel.
    specialists = [
        "broken_promises",
        "missing_results",
        "track_record",
        "pattern_finder",
        "side_effect",
        "timeline",
    ]

    for specialist in specialists:
        graph.add_edge("supervisor_route", specialist)
        graph.add_edge(specialist, "supervisor_compile")

    graph.add_edge("supervisor_compile", END)

    compiled = graph.compile()

    logger.info(
        "MOSAIC graph compiled successfully | nodes=8"
    )

    return compiled


mosaic_graph = build_mosaic_graph()