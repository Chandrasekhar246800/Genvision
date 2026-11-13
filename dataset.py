import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


class CloudDataset(Dataset):

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp"
    }

    def __init__(self, cloudy_dir, clean_dir):

        self.cloudy_dir = Path(cloudy_dir)
        self.clean_dir = Path(clean_dir)

        # --------------------------------
        # Validate directories
        # --------------------------------

        if not self.cloudy_dir.is_dir():
            raise FileNotFoundError(
                f"Cloudy image directory not found: "
                f"{self.cloudy_dir}"
            )

        if not self.clean_dir.is_dir():
            raise FileNotFoundError(
                f"Clean image directory not found: "
                f"{self.clean_dir}"
            )

        # --------------------------------
        # Collect supported image filenames
        # --------------------------------

        cloudy_files = {
            path.name
            for path in self.cloudy_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        }

        clean_files = {
            path.name
            for path in self.clean_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        }

        # --------------------------------
        # Validate filename correspondence
        # --------------------------------

        missing_clean = sorted(cloudy_files - clean_files)
        missing_cloudy = sorted(clean_files - cloudy_files)

        if missing_clean:
            raise ValueError(
                "Some cloudy images have no matching clean image.\n"
                f"Examples: {missing_clean[:10]}"
            )

        if missing_cloudy:
            raise ValueError(
                "Some clean images have no matching cloudy image.\n"
                f"Examples: {missing_cloudy[:10]}"
            )

        self.images = sorted(cloudy_files)

        if not self.images:
            raise ValueError(
                "No supported image pairs were found."
            )

        # --------------------------------
        # Image preprocessing
        # --------------------------------

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # --------------------------------
        # Validate image files
        # --------------------------------

        valid_images = []

        for img_name in self.images:

            cloudy_path = self.cloudy_dir / img_name
            clean_path = self.clean_dir / img_name

            try:

                with Image.open(cloudy_path) as img:
                    img.verify()

                with Image.open(clean_path) as img:
                    img.verify()

            except (
                OSError,
                ValueError,
                UnidentifiedImageError
            ) as error:

                raise ValueError(
                    f"Invalid image pair: {img_name}\n"
                    f"Reason: {error}"
                ) from error

            valid_images.append(img_name)

        self.images = valid_images

        print("=" * 50)
        print("CloudDataset Initialized")
        print("=" * 50)
        print(f"Cloudy directory : {self.cloudy_dir}")
        print(f"Clean directory  : {self.clean_dir}")
        print(f"Valid image pairs: {len(self.images)}")
        print("Image size       : 256 x 256")
        print("Image format     : RGB")
        print("Tensor range     : [0, 1]")
        print("=" * 50)

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        cloudy_path = self.cloudy_dir / img_name
        clean_path = self.clean_dir / img_name

        with Image.open(cloudy_path) as image:

            cloudy = image.convert("RGB")

        with Image.open(clean_path) as image:

            clean = image.convert("RGB")

        cloudy = self.transform(cloudy)
        clean = self.transform(clean)

        return cloudy, clean