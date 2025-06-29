from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# Load Azure ML workspace using config.json in the same folder
ws = Workspace.from_config()

# Cluster names
gpu_cluster_name = "gpt-cluster"
cpu_cluster_name = "gpt-mini-cluster"

# VM sizes
gpu_vm_size = "Standard_DS3_v2"         # GPU-optimized for GPT
cpu_vm_size = "Standard_DS3_v2"      # CPU-focused for GPT-mini

# Cluster settings
min_nodes = 0
max_nodes = 4
idle_seconds_before_scaledown = 120

def create_cluster(name, vm_size):
    try:
        cluster = ComputeTarget(workspace=ws, name=name)
        print(f"✅ Cluster '{name}' already exists.")
    except ComputeTargetException:
        print(f"🔧 Creating cluster '{name}' with VM size '{vm_size}'...")
        config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown
        )
        cluster = ComputeTarget.create(ws, name, config)
        cluster.wait_for_completion(show_output=True)
        print(f"✅ Cluster '{name}' created successfully.")
    return cluster

# Create both clusters
gpu_cluster = create_cluster(gpu_cluster_name, gpu_vm_size)
cpu_cluster = create_cluster(cpu_cluster_name, cpu_vm_size)

print("🏁 All clusters are ready to use.")
