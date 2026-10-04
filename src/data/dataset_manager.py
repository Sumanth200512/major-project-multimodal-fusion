"""
Dataset Manager for downloading, checking, and preparing SOCOFing and Iris datasets.
"""
import glob
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import kagglehub

from src.config import RAW_DATA_DIR, RANDOM_STATE
from src.utils.io import get_logger

logger = get_logger(__name__)

class DatasetManager:
    """
    Manages local storage and discovery of:
    1. SOCOFing dataset (ruizgara/socofing)
    2. GFI Iris dataset (basnamhamedsalih/iris-datasetndgfi)
    """

    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR):
        self.raw_data_dir = Path(raw_data_dir)
        self.socofing_dir = self.raw_data_dir / "socofing"
        self.iris_dir = self.raw_data_dir / "iris"

    def check_socofing_available(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if SOCOFing exists either in project data/raw/ or kagglehub cache.
        """
        candidate_paths = [
            self.socofing_dir / "SOCOFing",
            self.socofing_dir,
            Path.home() / ".cache" / "kagglehub" / "datasets" / "ruizgara" / "socofing" / "versions" / "2" / "SOCOFing",
            Path.home() / ".cache" / "kagglehub" / "datasets" / "ruizgara" / "socofing" / "SOCOFing"
        ]

        for p in candidate_paths:
            if p.exists():
                real_dir = p / "Real"
                alt_dir = p / "Altered"
                if real_dir.exists() and alt_dir.exists():
                    real_count = len(glob.glob(str(real_dir / "*.BMP")))
                    alt_count = len(glob.glob(str(alt_dir / "**" / "*.BMP"), recursive=True))
                    if real_count > 0 and alt_count > 0:
                        return True, {
                            "path": str(p),
                            "real_dir": str(real_dir),
                            "alt_dir": str(alt_dir),
                            "real_count": real_count,
                            "alt_count": alt_count
                        }

        return False, {"path": None, "real_count": 0, "alt_count": 0}

    def check_iris_available(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if Iris GFI exists either in project data/raw/ or kagglehub cache.
        """
        candidate_paths = [
            self.iris_dir / "DataSet" / "GFI",
            self.iris_dir / "GFI",
            Path.home() / ".cache" / "kagglehub" / "datasets" / "basnamhamedsalih" / "iris-datasetndgfi" / "versions" / "1" / "DataSet" / "GFI",
            Path.home() / ".cache" / "kagglehub" / "datasets" / "basnamhamedsalih" / "iris-datasetndgfi" / "DataSet" / "GFI"
        ]

        for p in candidate_paths:
            if p.exists():
                lbl_file = p / "List_left_GFI.txt"
                left_dir = p / "NUND_left" / "NUND_left"
                if not left_dir.exists():
                    left_dir = p / "NUND_left"
                
                if lbl_file.exists() and left_dir.exists():
                    img_count = len(glob.glob(str(left_dir / "*.tiff"))) + len(glob.glob(str(left_dir / "*.BMP")))
                    return True, {
                        "path": str(p),
                        "label_file": str(lbl_file),
                        "images_dir": str(left_dir),
                        "image_count": img_count
                    }

        return False, {"path": None, "image_count": 0}

    def download_datasets(self, force: bool = False) -> Dict[str, str]:
        """
        Download datasets via kagglehub if missing, and link/copy to raw data directory.
        """
        results = {}
        soco_avail, soco_info = self.check_socofing_available()
        if not soco_avail or force:
            logger.info("Downloading SOCOFing via kagglehub...")
            soco_path = kagglehub.dataset_download('ruizgara/socofing')
            results["socofing"] = str(soco_path)
        else:
            results["socofing"] = soco_info["path"]

        iris_avail, iris_info = self.check_iris_available()
        if not iris_avail or force:
            logger.info("Downloading Iris GFI via kagglehub...")
            iris_path = kagglehub.dataset_download('basnamhamedsalih/iris-datasetndgfi')
            results["iris"] = str(iris_path)
        else:
            results["iris"] = iris_info["path"]

        return results

    def get_dataset_status(self) -> Dict[str, Any]:
        """Return comprehensive status for UI display."""
        s_avail, s_info = self.check_socofing_available()
        i_avail, i_info = self.check_iris_available()
        return {
            "socofing_available": s_avail,
            "socofing_info": s_info,
            "iris_available": i_avail,
            "iris_info": i_info
        }
