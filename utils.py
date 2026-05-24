import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UTILS_PATH = PROJECT_ROOT / "utils.py"

_spec = importlib.util.spec_from_file_location("main_project_utils", str(_UTILS_PATH))
_main_project_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_project_utils)

RunningMeanStd = _main_project_utils.RunningMeanStd
Statistics = _main_project_utils.Statistics
global2body = _main_project_utils.global2body
generate_points = _main_project_utils.generate_points



