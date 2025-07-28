"""Dataset generator
====================
* **Selective split generation**: choose any combination of ``train``, ``val``
  and ``test`` via ``--splits``.
* **Per‑split sample limits**: cap how many examples are processed with
  ``--limit_<split>``.
* **Reverse‑mapping metadata**: each CSV row now contains the *scaling* and
  *translation* parameters (``scale_x``, ``scale_y``, ``trans_x``,
  ``trans_y``) needed to map cropped keypoints back to the original frame
  """

import logging
from pathlib import Path
from typing import List, MutableMapping, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

LOGGER = logging.getLogger("cropping_heatmap")


class DatasetGenerator:
    """Generate cropped frames + keypoint CSVs for the requested splits."""
    CSV_COLUMNS: List[str] = (
            ["filepath"]
            + [f"k{i}x" for i in range(8)]
            + [f"k{i}y" for i in range(8)]
            + ["bbox"]
    )

    def __init__(
            self,
            frames_dir: Path,
            out_dir: Path,
            labels_dir: Path,
            dest_size: int = 224,
    ) -> None:
        self.frames_dir = frames_dir.expanduser().resolve()
        self.out_dir = out_dir.expanduser().resolve()
        self.labels_dir = labels_dir.expanduser().resolve()
        self.dest_size = dest_size
        self._splits: MutableMapping[str, pd.DataFrame] = {}

    def load_splits(self, splits: Sequence[str]) -> None:
        """Read the *json* annotation files for the requested ``splits``."""
        self._splits.clear()
        for split in splits:
            json_path = self.labels_dir / f"{split}.json"
            if not json_path.exists():
                LOGGER.error(f"Annotation file not found for split '{split}': {json_path}")
                continue
            self._splits[split] = pd.read_json(json_path)
            LOGGER.debug(f"Loaded {json_path}: {len(self._splits[split])} samples")

    def generate(self, *,
                 splits: Sequence[str] = ("train", "val", "test"),
                 limit_train: Optional[int] = None,
                 limit_val: Optional[int] = None,
                 limit_test: Optional[int] = None,
                 ) -> None:
        """Generate crops/CSV for each requested split with optional limits."""

        self.load_splits(splits)
        if not self._splits:
            LOGGER.error("No splits loaded. Aborting generation.")
            return

        limits = {
            "train": limit_train,
            "val": limit_val,
            "test": limit_test,
        }

        for split in splits:
            if split not in self._splits:
                LOGGER.warning(f"Skipping generation for split '{split}' as it was not loaded.")
                continue
            LOGGER.info(f"Generating {split} split")
            self._generate_one_split(split, limits.get(split))

    def _generate_one_split(self, split: str, limit: Optional[int]) -> None:
        df = self._splits[split]
        if limit is not None:
            df = df.iloc[:limit]

        csv_rows: List[List[float | str]] = []
        split_out_dir = self.out_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split} samples"):
            filename: str = row["filename"]
            keypoints = np.asarray(row["keypoints"]).reshape(-1, 3)[:, :2]
            bbox = np.asarray(row["bbox"])  # [x0, y0, x1, y1]

            frame = self._read_image(filename, row)
            if frame is None:
                continue  # already warned

            # FIX: Unpack all 6 values returned by the corrected _crop_and_resize function.
            cropped_kps, cropped_img = self._crop_and_resize(frame, keypoints, bbox)

            out_path = split_out_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cropped_img)

            # FIX: Correctly structure the CSV row data to match the column order.
            # 1. Separate x and y coordinates.
            kps_x = cropped_kps[:, 0].tolist()
            kps_y = cropped_kps[:, 1].tolist()
            # 2. Convert NumPy array 'bbox' to a list for concatenation.
            bbox_list = bbox.tolist()

            csv_rows.append(
                [str(out_path)] +  # filepath
                kps_x +  # all x-coordinates
                kps_y +  # all y-coordinates
                [str(bbox_list)]  # bbox (as a string for clean CSV)
            )

        # Write CSV
        csv_path = self.out_dir / split / f"keypoints.csv"  # Save CSV in parent dir
        pd.DataFrame(csv_rows, columns=self.CSV_COLUMNS).to_csv(csv_path, index=False)
        LOGGER.info(f"{split}: wrote {len(csv_rows)} samples to {csv_path}")

    def _read_image(self, filename: str, row) -> Optional[np.ndarray]:
        """Load the frame referred to by *filename* returning *None* if missing."""
        try:
            _, traj = filename.replace(".png", "").split("_", 1)
        except ValueError:
            LOGGER.warning(f"Could not parse trajectory from filename '{filename}'. Skipping.")
            return None

        img_path = self.frames_dir / traj / filename
        img = cv2.imread(str(img_path))
        if img is None:
            LOGGER.warning(f"Missing image {img_path} (ann row id {row.name})")
        return img

    def _crop_and_resize(self,
                         frame: np.ndarray,
                         keypoints: np.ndarray,
                         bbox: np.ndarray,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop *frame* to *bbox* and rescale.

        Returns
        -------
        cropped_kps
            Keypoints in the crop space (shape 8×2).
        cropped_img
            The resized crop (dest_size × dest_size × 3).
        """
        x0, y0, x1, y1 = bbox

        crop = frame[int(y0): int(y1), int(x0): int(x1)]

        # Add epsilon to prevent division by zero for zero-sized bboxes
        height, width, _ = crop.shape
        if width == 0 or height == 0:
            LOGGER.warning(f"Invalid crop size (width={width}, height={height}) for bbox {bbox}. Skipping.")
            # Return dummy values to avoid crashing
            return (np.zeros_like(keypoints),
                    np.zeros((self.dest_size, self.dest_size, 3), dtype=np.uint8))

        cropped_img = cv2.resize(crop, (self.dest_size, self.dest_size), interpolation=cv2.INTER_LINEAR)
        # Transform keypoints: 1. Translate to crop origin.
        cropped_kps = (keypoints - np.array([x0, y0]))

        return cropped_kps, cropped_img


def main(config_path: str | Path,
         *,
         splits: Sequence[str] = ("train", "val", "test"),
         limit_train: Optional[int] = None,
         limit_val: Optional[int] = None,
         limit_test: Optional[int] = None,
         loglevel: str | int = "INFO",
         ) -> None:
    # ... (main function body remains the same)
    logging.basicConfig(level=loglevel, format="[%(levelname)s] %(message)s")
    try:
        cfg = OmegaConf.load(str(config_path))
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found at: {config_path}")
        return

    # Using OmegaConf.select to safely access nested keys
    frames_dir_str = OmegaConf.select(cfg, "root.data_out")
    labels_dir_str = OmegaConf.select(cfg, "root.labels_dir")
    transformation = OmegaConf.select(cfg, "data.transformation")

    if not all([frames_dir_str, labels_dir_str, transformation]):
        LOGGER.error("Config is missing required keys: root.data_out, root.labels_dir, or data.transformation")
        return

    frames_dir = Path(frames_dir_str) / transformation
    trans_frames_dir = Path(frames_dir_str) / f"{transformation}_cropped"
    labels_dir = Path(labels_dir_str)
    trans_frames_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Frames dir: %s", frames_dir)
    LOGGER.info("Output dir: %s", trans_frames_dir)
    LOGGER.info("Labels dir: %s", labels_dir)

    gen = DatasetGenerator(frames_dir, trans_frames_dir, labels_dir)
    gen.generate(
        splits=splits,
        limit_train=limit_train,
        limit_val=limit_val,
        limit_test=limit_test,
    )


if __name__ == "__main__":
    # It's good practice to use a CLI argument parser like argparse,
    # but for this script, hardcoded values are clear.
    config_path = "configs/mobilenet_heatmap.yaml"
    splits: Sequence[str] = ("train", "val", "test")
    limit_train: Optional[int] = 1000  # Example: Set a limit for faster testing
    limit_val: Optional[int] = 1000
    limit_test: Optional[int] = 1000
    loglevel: str | int = "INFO"

    main(config_path,
         splits=splits,
         limit_train=limit_train,
         limit_val=limit_val,
         limit_test=limit_test,
         loglevel=loglevel,
         )
