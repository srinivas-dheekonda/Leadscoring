import tarfile

# Files to combine inside the tarball
files_to_compress = ["res1_model.pkl", "inference.py"]

# Create the compressed tarball
with tarfile.open("model.tar.gz", "w:gz") as tar:
  for file in files_to_compress:
    tar.add(file)
    print(f"Added {file} to model.tar.gz")

print("Successfully created model.tar.gz!")