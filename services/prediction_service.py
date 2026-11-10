"""Application-facing prediction service."""

from pathlib import Path

from predict import PredictionError, SUPPORTED_EXTENSIONS, predict_file


def predict_media(file_path: str | Path, modality: str) -> dict:
	"""Run a trained modality model and return its structured result."""
	return predict_file(file_path, modality=modality)
