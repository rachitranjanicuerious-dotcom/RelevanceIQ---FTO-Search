# Has all nodes (6 llm calls)


from langgraph.graph import StateGraph, START, END

from state import RelevanceState

from nodes2 import(
    extract_product_features,
    extract_patent_features,
    compare_features,
    assign_relevance,
    generate_rationale,
    confidence_node
)

# Graph
graph = StateGraph(RelevanceState)

#Nodes
graph.add_node("Extract Product Features" , extract_product_features)
graph.add_node("Extract Patent Features" , extract_patent_features)
graph.add_node("Compare Features" , compare_features)
graph.add_node("Assign Relevance", assign_relevance)
graph.add_node("Rationale" , generate_rationale)
graph.add_node("Confidence", confidence_node)

# Edge
graph.add_edge(START, "Extract Product Features")
graph.add_edge("Extract Product Features", "Extract Patent Features")
graph.add_edge('Extract Patent Features', "Compare Features")
graph.add_edge("Compare Features" ,"Assign Relevance")
graph.add_edge("Assign Relevance" , "Rationale")
graph.add_edge("Rationale", "Confidence")
graph.add_edge("Confidence" , END)



#Define the workflow
workflow = graph.compile()

png = workflow.get_graph().draw_mermaid_png()

# with open("workflow.png", "wb") as f:
#     f.write(png)

# print("Workflow image saved as workflow.png")



