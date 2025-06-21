"""Dataset generator
====================
* **Selective split generation**: choose any combination of ``train``, ``val``
  and ``test`` via ``--splits``.
* **Per‑split sample limits**: cap how many examples are processed with
  ``--limit_<split>``.
* **Reverse‑mapping metadata**: each CSV row now contains the *scaling* and
  *translation* parameters (``scale_x``, ``scale_y``, ``trans_x``,
  ``trans_y``) needed to map cropped keypoints back to the original frame.
```
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

class DatasetGenerator:
    """Generate cropped frames + keypoint CSVs for the requested splits."""

    #: CSV columns – image path followed by 8×(x, y) keypoints and the transform
    CSV_COLUMNS: List[str] = (
        ["filepath"]
        + [f"k{i}x" for i in range(8)]
        + [f"k{i}y" for i in range(8)]
        + ["scale_x", "scale_y", "trans_x", "trans_y"]
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
            self._splits[split] = pd.read_json(json_path)
            LOGGER.debug("Loaded %%s: %%d samples", json_path, len(self._splits[split]))

    def generate(
        self,
        *,
        splits: Sequence[str] = ("train", "val", "test"),
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        limit_test: Optional[int] = None,
    ) -> None:
        """Generate crops/CSV for each requested split with optional limits."""

        if not self._splits:
            self.load_splits(splits)

        limits = {
            "train": limit_train,
            "val": limit_val,
            "test": limit_test,
        }

        for split in splits:
            LOGGER.info("Generating %%s split", split)
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
            bbox = np.asarray(row["bbox"]).reshape(-1, 2)  # [[x0, y0], [x1, y1]]

            frame = self._read_image(filename, row)
            if frame is None:
                continue  # already warned

            cropped_kps, cropped_img, scale_x, scale_y, trans_x, trans_y = (
                self._crop_and_rescale(frame, keypoints, bbox)
            )

            out_path = split_out_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cropped_img)

            csv_rows.append(
                [str(out_path)] +  # filepath
                cropped_kps.flatten().tolist() +  # 8×(x, y)
                [scale_x, scale_y, trans_x, trans_y]
            )

        # Write CSV ------------------------------------------------------
        csv_path = split_out_dir / "keypoints.csv"
        pd.DataFrame(csv_rows, columns=self.CSV_COLUMNS).to_csv(csv_path, index=False)
        LOGGER.info("%%s: wrote %%d samples to %%s", split, len(csv_rows), csv_path)

    def _read_image(self, filename: str, row) -> Optional[np.ndarray]:
        """Load the frame referred to by *filename* returning *None* if missing."""
        frame_num, traj = filename.replace(".png", "").split("_")
        img_path = self.frames_dir / traj / filename
        img = cv2.imread(str(img_path))
        if img is None:
            LOGGER.warning("Missing image %%s (ann row id %%s)", img_path, row.name)
        return img

    def _crop_and_rescale(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, int, int]:
        """Crop *frame* to *bbox* and rescale.

        Returns
        -------
        cropped_kps      – keypoints in the crop space (shape 8×2)
        cropped_img      – the resized crop (dest_size × dest_size × 3)
        scale_x, scale_y – factors used during resize (dest/width, dest/height)
        trans_x, trans_y – origin of the bbox in the original frame
        """
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        crop = frame[int(y0) : int(y1), int(x0) : int(x1)]

        width, height = x1 - x0, y1 - y0
        scale_x = self.dest_size / width
        scale_y = self.dest_size / height

        cropped_img = cv2.resize(crop, (self.dest_size, self.dest_size), interpolation=cv2.INTER_LINEAR)
        cropped_kps = (keypoints - np.array([x0, y0])) * np.array([scale_x, scale_y])
        return cropped_kps, cropped_img, float(scale_x), float(scale_y), int(x0), int(y0)



def main(
    config_path: str | Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
    loglevel: str | int = "INFO",
) -> None:
    """Generate the requested dataset splits.

    Parameters
    ----------
    config_path
        Path to the YAML config with ``root.dataset`` / ``data.transformation``
        etc. (same schema as before).
    splits
        Iterable of split names (any subset of ``{"train", "val", "test"}``).
    limit_* per split
        Maximum number of samples; ``None`` means no limit.
    loglevel
        Anything accepted by :pyfunc:`logging.basicConfig`.
    """
    logging.basicConfig(level=loglevel, format="[%(levelname)s] %(message)s")

    cfg = OmegaConf.load(str(config_path))

    frames_dir = Path(cfg.root.data_out) / cfg.data.transformation
    trans_frames_dir = Path(cfg.root.data_out) / f"{cfg.data.transformation}_cropped"
    labels_dir = Path(cfg.root.labels_dir)
    trans_frames_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Frames dir: %s", frames_dir)
    LOGGER.info("Output dir: %s", trans_frames_dir)

    gen = DatasetGenerator(frames_dir, trans_frames_dir, labels_dir)
    gen.generate(
        splits=splits,
        limit_train=limit_train,
        limit_val=limit_val,
        limit_test=limit_test,
    )



if __name__ == "__main__":
    config_path = "configs/mobilenet.yaml"
    splits: Sequence[str] = ("train", "val", "test")
    limit_train: Optional[int] = None
    limit_val: Optional[int] = None
    limit_test: Optional[int] = None
    loglevel: str | int = "INFO"

    main(
        config_path,
        splits=splits,
        limit_train=limit_train,
        limit_val=limit_val,
        limit_test=limit_test,
        loglevel=loglevel,
    )