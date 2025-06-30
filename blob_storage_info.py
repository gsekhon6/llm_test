from azureml.core import Workspace

# Load workspace
ws = Workspace.from_config()

# Get default datastore
datastore = ws.get_default_datastore()

print("✅ Default Datastore Info")
print("Name:         ", datastore.name)
print("Account name: ", datastore.account_name)
print("Container:    ", datastore.container_name)
print("Datastore type:", type(datastore))

# If you've uploaded to: datasets/squad/train.csv
blob_path = "datasets/squad/train.csv"
print("\n📂 Expected blob path:", blob_path)
