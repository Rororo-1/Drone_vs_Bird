import kagglehub

# Download latest version
path = kagglehub.dataset_download("romsham/dronevsbird-foryolo")

print("Path to dataset files:", path)