import argparse, os, time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.eval()  # Set model to evaluation mode

latencies = []
perplexities = []

# Evaluate first 10 questions for demo
for q in df['question'].head(10):
    inputs = tokenizer(q, return_tensors="pt")

    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    end_time = time.time()

    latencies.append(end_time - start_time)
    perplexities.append(perplexity)

# Calculate averages
avg_latency = sum(latencies) / len(latencies)
print(avg_latency)
avg_perplexity = sum(perplexities) / len(perplexities)
print(avg_perplexity)

# Log metrics to AzureML
run.log("inference_latency", avg_latency)
run.log("perplexity", avg_perplexity)


#
# # Load model
# generator = pipeline("text-generation", model=args.model_name)
#
# # Evaluate
# latencies = []
# for q in df['question'].head(10):
#     start = time.time()
#     generator(q, max_length=50)
#     end = time.time()
#     latencies.append(end - start)
#
# # Log metrics
# avg_latency = sum(latencies) / len(latencies)
# dummy_perplexity = 25.0 if args.model_name == "gpt2" else 28.0
#
# run.log("inference_latency", avg_latency)
# run.log("perplexity", dummy_perplexity)
