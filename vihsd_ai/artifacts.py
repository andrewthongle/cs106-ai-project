"""JSON, checksums and loading previously saved baseline checkpoints."""

import hashlib
import json
from pathlib import Path

import joblib


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


class _LegacyBaselineUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    """Resolve the pre-refactor notebook class without modifying __main__."""

    def find_class(self, module, name):
        if module == "__main__" and name == "SocialPreprocessor":
            from .preprocessing import SocialPreprocessor

            return SocialPreprocessor
        return super().find_class(module, name)


def load_baseline(path):
    """Load a trusted joblib file, including checkpoints from the old notebook.

    New checkpoints use vihsd_ai.preprocessing.SocialPreprocessor. The fallback
    only handles the old __main__ class name, preserving the checkpoint bytes.
    """
    try:
        return joblib.load(path)
    except AttributeError as exc:
        if "SocialPreprocessor" not in str(exc):
            raise
        # joblib's stream reader also supports the compressed files saved here.
        from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap

        with (
            open(path, "rb") as raw,
            _validate_fileobject_and_memmap(raw, str(path), None) as (stream, _),
        ):
            return _LegacyBaselineUnpickler(
                str(path), stream, ensure_native_byte_order=True
            ).load()
