from relea.data.miniimagenet import MiniImageNetDataModule
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datamodule = MiniImageNetDataModule(root=os.getcwd())
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))
    print(batch)
    print(batch[0].dtype)
    img = batch[0].squeeze().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# from datasets import load_dataset

# ds = load_dataset("timm/mini-imagenet", split="train")
# print(ds[0].keys())
# print(ds[0]["image"])