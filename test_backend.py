import httpx
import json
import sys

print("Testing backend generation...")
client = httpx.Client(base_url="http://localhost:8080", timeout=60.0)
body = {"query": "runway excursion", "thread_id": "test_123", "max_hops": 2}

try:
    with client.stream("POST", "/query/stream", json=body) as r:
        for raw in r.iter_lines():
            if raw and raw.startswith("data:"):
                try:
                    data = json.loads(raw.replace("data:", "").strip())
                    if "text" in data:
                        print(data["text"], end="", flush=True)
                    elif "draft" in data:
                        print("\n[DRAFT COMPLETED]")
                except json.JSONDecodeError:
                    pass
    print("\n\nStream finished successfully!")
except Exception as e:
    print(f"\nError: {e}")
