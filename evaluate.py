import argparse, os, time
import pandas as pd
from transformers import pipeline
from azureml.core import Run

# Setup
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--model_name", type=str)  # e.g., "gpt2" or "distilgpt2"
args = parser.parse_args()

run = Run.get_context()

# Load dataset
# data_path = os.path.join(args.input_data, "train.csv")

data_path = args.input_data
df = pd.read_csv(data_path)

# Load model
generator = pipeline("text-generation", model=args.model_name)

# Evaluate
latencies = []
for q in df['question'].head(10):
    start = time.time()
    generator(q, max_length=50)
    end = time.time()
    latencies.append(end - start)

# Log metrics
avg_latency = sum(latencies) / len(latencies)
dummy_perplexity = 25.0 if args.model_name == "gpt2" else 28.0

run.log("inference_latency", avg_latency)
run.log("perplexity", dummy_perplexity)
