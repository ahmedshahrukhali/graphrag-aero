import sys
from gradio_client import Client

print("Testing space...")
try:
    client = Client("http://localhost:7860")
    print("Connected to Gradio client!")
    
    # We will submit a query to the chat interface.
    # The signature depends on what the space accepts.
    # Assuming endpoint /chat or similar. Let's just print endpoints:
    print("Endpoints:", client.endpoints)
except Exception as e:
    print(f"Failed: {e}")
