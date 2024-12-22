import wandb
import os
from tensorflow import keras

# Inisialisasi wandb project
wandb.init(
    project="trash-classification",
    name="take-home-test"
)

# Path ke file model .keras
model_path = "./Deploy/Model/model.keras"

# Memastikan file model ada
if os.path.exists(model_path):
    # Load model
    model = keras.models.load_model(model_path)
    
    # Simpan model ke wandb
    artifact = wandb.Artifact(
        name="trash-classification",
        type="model",
        description="Model ini dilatih pada dataset trashnet dan digunakan untuk mengklasifikasikan sampah"
    )
    
    # Log model sebagai artifact
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    print("Model berhasil diupload ke W&B")
else:
    print(f"Error: File model tidak ditemukan di {model_path}")

wandb.finish()