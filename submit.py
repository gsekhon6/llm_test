from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset, ComputeTarget

# 1. Load workspace
ws = Workspace.from_config()

# 2. Get compute target (GPU cluster)
compute = ComputeTarget(workspace=ws, name="gpt-cluster")

# 3. Get the correct registered dataset
dataset = Dataset.get_by_name(ws, name="squad_train_v1")

# 4. Load environment from your YAML file
gpt_env = Environment.from_conda_specification(
    name="gpt-env",
    file_path="gpt-env.yml"
)

# Optional: register environment (recommended for reuse)
gpt_env.register(workspace=ws)

# 5. Configure the job
config = ScriptRunConfig(
    source_directory=".",          # Make sure evaluate.py is in this folder
    script="evaluate.py",
    compute_target=compute,
    environment=gpt_env,
    arguments=[
        "--input_data", dataset.as_named_input("input_data").as_mount(),
        "--model_name", "gpt2"
    ]
)

# 6. Submit experiment
experiment = Experiment(ws, "gpt-comparison")
run = experiment.submit(config)
run.wait_for_completion(show_output=True)
