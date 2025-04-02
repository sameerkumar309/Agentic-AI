from agno.agent import Agent 
from agno.models.groq import Groq # (We will import tool classes or create custom ones in later steps)
import os
from dotenv import load_dotenv
load_dotenv()

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

# Initialize the agent with an LLM model and a basic role description
agent = Agent(
    model=Groq(id="qwen-2.5-32b"),  # GPT-4 (you can use "gpt-3.5-turbo" or other models as needed)
    description="You are a helpful logistics assistant.",
    markdown=True  # Enable markdown formatting in responses for readability
)

# Update agent creation with specific instructions and placeholders for tools
agent = Agent(
    model=Groq(id="qwen-2.5-32b"),
    description="You are a knowledgeable logistics assistant.",
    instructions=[
        "If the user asks about a shipment status or provides a tracking ID, use the TrackingTool to retrieve the shipment status.",
        "If the user asks about optimizing a delivery route, use the RouteTool to compute the best route.",
        "Provide clear and concise responses. For routes, list the stop order and total distance. For tracking, give the current status and any relevant details."
    ],
    tools=[],  # tools will be added in the next step
    show_tool_calls=True,  # for development, show when the agent calls a tool (useful for debugging)
    markdown=True
)

# Sample logistics data sources
tracking_data = {
    "TRK12345": "In transit at Toronto distribution center",
    "TRK98765": "Delivered on 2025-03-09 10:24",
    "TRK55555": "Out for delivery - last scanned at Vancouver hub"
}
distance_matrix = {
    "Warehouse": {"A": 10, "B": 15, "C": 20},
    "A": {"Warehouse": 10, "B": 12, "C": 5},
    "B": {"Warehouse": 15, "A": 12, "C": 8},
    "C": {"Warehouse": 20, "A": 5,  "B": 8}
}

import re

class TrackingTool:
    """Tool to fetch shipment status by tracking ID."""
    def __init__(self):
        self.name = "TrackingTool"
        self.description = "Provides shipment status updates given a tracking ID."
    
    def run(self, query: str) -> str:
        # Extract tracking ID from the query (assume IDs follow format like TRK12345)
        match = re.search(r"\bTRK\d+\b", query.upper())
        if not match:
            return "I couldn't find a tracking number in the query."
        tracking_id = match.group(0)
        # Lookup the tracking ID in our data
        status = tracking_data.get(tracking_id)
        if status:
            return f"Status for **{tracking_id}**: {status}"
        else:
            return f"Sorry, I have no information on tracking ID {tracking_id}."


class RouteTool:
    """Tool to calculate optimal route given an origin and multiple destinations."""
    
    def __init__(self):
        self.name = "RouteTool"
        self.description = "Computes the best delivery route given a start and destinations."
    
    def run(self, query: str) -> str:
        # Expect queries in form "from X to Y, Z, ..." (case-insensitive)
        pattern = re.compile(r"from\s+([\w\s]+)\s+to\s+(.+)", re.IGNORECASE)
        match = pattern.search(query)
        if not match:
            return ("Please specify a route in the format 'from <Origin> to <Dest1>, <Dest2>, ...'.")
        origin = match.group(1).strip()
        dests_part = match.group(2)
        # Split multiple destinations by commas or 'and'
        destinations = [d.strip() for d in re.split(r",| and ", dests_part) if d.strip()]
        # Validate locations exist in our distance matrix
        if origin not in distance_matrix:
            return f"Unknown origin location: {origin}."
        for loc in destinations:
            if loc not in distance_matrix:
                return f"Unknown destination: {loc}."
        # Compute the shortest route visiting all destinations (simple brute-force for demo)
        best_route = None
        best_distance = float('inf')
        from itertools import permutations
        for perm in permutations(destinations):
            total = 0
            current = origin
            for nxt in perm:
                total += distance_matrix[current][nxt]
                current = nxt
            # No return to origin in this scenario (one-way route finishing at last destination)
            if total < best_distance:
                best_distance = total
                best_route = perm
        route_list = " -> ".join([origin] + list(best_route)) if best_route else origin
        return f"Optimal route: **{route_list}** (Total distance: {best_distance} km)"
    
# Instantiate the tools
tracking_tool = TrackingTool()
route_tool = RouteTool()
# Attach tools to the agent
agent.tools = [tracking_tool, route_tool]

# Example 1: Tracking query
user_query = "Where is shipment TRK12345 right now?"
agent.print_response(user_query)

# Example 2: Route optimization query
user_query = "What is the best route from Warehouse to A, B and C?"
agent.print_response(user_query)