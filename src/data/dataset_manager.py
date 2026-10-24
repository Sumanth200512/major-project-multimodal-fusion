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

        # Ensure base kagglehub cache directories exist to prevent [Errno 2] FileNotFoundError on Linux
        soco_cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "ruizgara" / "socofing"
        soco_cache_dir.mkdir(parents=True, exist_ok=True)

        iris_cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "basnamhamedsalih" / "iris-datasetndgfi"
        iris_cache_dir.mkdir(parents=True, exist_ok=True)

        soco_avail, soco_info = self.check_socofing_available()
        if not soco_avail or force:
            logger.info("Downloading SOCOFing via kagglehub...")
            soco_path = self._download_with_retry('ruizgara/socofing')
            results["socofing"] = str(soco_path)
        else:
            results["socofing"] = soco_info["path"]

        iris_avail, iris_info = self.check_iris_available()
        if not iris_avail or force:
            logger.info("Downloading Iris GFI via kagglehub...")
            iris_path = self._download_with_retry('basnamhamedsalih/iris-datasetndgfi')
            results["iris"] = str(iris_path)
        else:
            results["iris"] = iris_info["path"]

        return results

    def _download_with_retry(self, handle_str: str) -> str:
        """
        Download dataset via kagglehub with automatic fallback for:
        1. GCS MD5 checksum mismatch (kagglehub DataCorruptionError).
        2. Archive extraction issues where archive_path was deleted prior to extraction
           or incomplete downloads causing FileNotFoundError / [Errno 2].
        """
        import kagglehub.clients as clients
        import kagglehub.http_resolver as hr

        # Ensure safe extraction that checks archive existence before removing
        orig_resolve = hr.DatasetHttpResolver._resolve

        def safe_resolve(resolver_self, h, path=None, force_download=False, output_dir=None):
            from kagglehub.clients import build_kaggle_client
            from kagglehub.cache import Cache
            from kagglehub.http_resolver import _get_current_version, _build_dataset_download_request, _extract_archive, _prepare_output_dir
            from kagglehub.exceptions import handle_call

            with build_kaggle_client() as api_client:
                if not h.is_versioned():
                    h = h.with_version(_get_current_version(api_client, h))

                cache = Cache(override_dir=output_dir)
                dataset_path = cache.load_from_cache(h, path)
                if dataset_path and not force_download:
                    return dataset_path, h.version
                elif dataset_path and force_download:
                    cache.delete_from_cache(h, path)

                if output_dir:
                    _prepare_output_dir(output_dir, path, force_download=bool(force_download))

                r = _build_dataset_download_request(h, path)
                out_path = cache.get_path(h, path)

                if path:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    response = handle_call(lambda: api_client.datasets.dataset_api_client.download_dataset(r), h)
                    clients.download_file(response, out_path, h, extract_auto_compressed_file=True)
                else:
                    archive_path = cache.get_archive_path(h)
                    os.makedirs(os.path.dirname(archive_path), exist_ok=True)

                    response = handle_call(lambda: api_client.datasets.dataset_api_client.download_dataset(r), h)
                    clients.download_file(response, archive_path, h)

                    if not os.path.exists(archive_path):
                        raise FileNotFoundError(f"Archive file not found after download: {archive_path}")

                    _extract_archive(archive_path, out_path)

                    if os.path.exists(archive_path):
                        try:
                            os.remove(archive_path)
                        except OSError:
                            pass

                cache.mark_as_complete(h, path)
                return out_path, h.version

        hr.DatasetHttpResolver._resolve = safe_resolve
        orig_get_md5 = clients.get_md5_checksum_from_response

        try:
            return str(kagglehub.dataset_download(handle_str))
        except (kagglehub.exceptions.DataCorruptionError, FileNotFoundError, OSError) as e:
            logger.warning(
                f"KaggleHub error during download of {handle_str} ({e}). "
                "Retrying with checksum verification bypassed and clean archive cache..."
            )
            # Remove any partial archive before retrying
            clients.get_md5_checksum_from_response = lambda resp: None
            return str(kagglehub.dataset_download(handle_str, force_download=True))
        finally:
            hr.DatasetHttpResolver._resolve = orig_resolve
            clients.get_md5_checksum_from_response = orig_get_md5

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
