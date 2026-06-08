import gradio as gr
import time

def chat(msg, hist):
    yield gr.ChatMessage(role="assistant", content="", metadata={"title": "Thought"})
    
    for i in range(5):
        time.sleep(1)
        yield gr.ChatMessage(role="assistant", content=f"thinking... {i}", metadata={"title": "Thought", "status": "pending"})
        
    yield gr.ChatMessage(role="assistant", content="Final answer")

demo = gr.ChatInterface(chat, type="messages")
demo.launch(server_name="0.0.0.0", server_port=7861)
