print("STARTING")

from dataset import CloudDataset

print("IMPORT SUCCESS")

dataset = CloudDataset(
    "dataset/cloudy",
    "dataset/clean"
)

print("DATASET CREATED")

print("TOTAL:", len(dataset))

cloudy, clean = dataset[0]

print(cloudy.shape)
print(clean.shape)

print("DONE")