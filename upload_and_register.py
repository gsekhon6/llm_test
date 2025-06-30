from azureml.core import Workspace, Dataset
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azure.storage.blob import BlobServiceClient
from datetime import datetime, timedelta

# Load Azure ML workspace from config.json
ws = Workspace.from_config()

# Absolute path to your train.csv file
data_path = "/Users/gurkirat/Documents/Expedaite_classifier/expedaite/squad/train.csv"

# Get the default datastore (usually points to Azure Blob Storage)
datastore = ws.get_default_datastore()

# Upload the CSV to Azure Blob Storage
datastore.upload_files(
    files=[data_path],
    target_path="datasets/squad/",  # This is the folder in blob storage
    overwrite=True,
    show_progress=True
)

# Create a Tabular Dataset from the uploaded file
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, "datasets/squad/train.csv"))

# Register the dataset in Azure ML
dataset = dataset.register(
    workspace=ws,
    name="squad_train_v1",  # Registered name (you'll use this later to load it)
    description="Train CSV from SQuAD 2.0 for GPT evaluation",
    create_new_version=True
)

print("TabularDataset 'squad_train_v1' registered successfully.")

# Replace with your storage account info
account_name = "mltest16843092961"
account_key = "tYlRzSX57lS6yEqqVjl8kSg8yxO9J0TsXW/7ZEHlIfR7RDEFS5Sp4Vt0LQkPvV9gBxzuadVzyRwn+ASt+ZUBKw=="  # from Azure Portal → Access keys
container_name = "azureml-blobstore-9a827842-ad93-41e0-988d-94cb363da755"     # usually like 'azureml-blobstore-...'
blob_name = "datasets/squad/train.csv"       # path to your uploaded file

# Generate the SAS token
sas_token = generate_blob_sas(
    account_name=account_name,
    container_name=container_name,
    blob_name=blob_name,
    account_key=account_key,
    permission=BlobSasPermissions(read=True),
    expiry=datetime.utcnow() + timedelta(days=7)
)

# Build the full URL
sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"

print("✅ SAS URL to blob:\n", sas_url)
# sas_url = datastore.generate_shared_access_signature()
# print("SAS URL:", sas_url)
