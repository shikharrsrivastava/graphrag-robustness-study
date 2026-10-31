import requests

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "embeddinggemma",
        "input": [
            "Artificial intelligence is transforming healthcare.",
            "Machine learning learns patterns from data."
        ]
    },
    timeout=60
)

response.raise_for_status()

data = response.json()
embeddings = data["embeddings"]

print("Number of embeddings:", len(embeddings))
print("Dimensions:", len(embeddings[0]))
print("First 10 values:", embeddings[0][:10])