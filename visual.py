from langgraph_database_backend_phase4 import chatbot
import os

png = chatbot.get_graph().draw_mermaid_png()

with open("langgraph.png", "wb") as f:
    f.write(png)

os.startfile("langgraph.png")