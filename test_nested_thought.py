import gradio as gr
import time

def chat(msg, hist):
    chat_list = [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking process", "id": "main", "status": "pending"}}
    ]
    yield chat_list
    
    steps = ["Searching vector DB...", "Extracting entities...", "Synthesizing answer..."]
    
    for i, step in enumerate(steps):
        time.sleep(1.5)
        # Mark previous child as done, if any
        if len(chat_list) > 2:
            prev_idx = len(chat_list) - 1
            chat_list[prev_idx]["metadata"]["status"] = "done"
            chat_list[prev_idx]["metadata"]["duration"] = 1.5
            
        chat_list.append({
            "role": "assistant", 
            "content": "", 
            "metadata": {"title": step, "parent_id": "main", "status": "pending"}
        })
        yield chat_list
        
    time.sleep(1)
    # Mark last child as done
    if len(chat_list) > 2:
        prev_idx = len(chat_list) - 1
        chat_list[prev_idx]["metadata"]["status"] = "done"
        chat_list[prev_idx]["metadata"]["duration"] = 1.0
        
    # Mark main as done
    chat_list[1]["metadata"]["status"] = "done"
    chat_list[1]["metadata"]["duration"] = 1.5 * 3 + 1
    
    # Append final answer
    chat_list.append({"role": "assistant", "content": "Here is the final answer!"})
    yield chat_list

demo = gr.ChatInterface(chat, type="messages")
demo.launch(server_name="0.0.0.0", server_port=7862)
