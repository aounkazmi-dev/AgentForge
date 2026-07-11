from backend import chatbot


with open("langgraph_xray.png", "wb") as f:
    f.write(chatbot.get_graph(xray=True).draw_mermaid_png())

print("✅ Graph saved as langgraph_xray.png")