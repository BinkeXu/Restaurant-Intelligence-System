import ollama
import sys

def pull_model(name):
    print(f"Pulling model: {name}...")
    try:
        ollama.pull(name)
        print(f"Successfully pulled {name}")
    except Exception as e:
        print(f"Error pulling {name}: {e}")
        # If it fails, maybe the server isn't running or the model name is slightly different
        # but the plan said xbai-embed-large

if __name__ == "__main__":
    pull_model("mxbai-embed-large")
    pull_model("llama3.2")
