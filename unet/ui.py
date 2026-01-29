import importlib.util
import os
import sys


def _load_ui_package():
    base_dir = os.path.dirname(__file__)
    pkg_dir = os.path.join(base_dir, "ui")
    init_path = os.path.join(pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "ui",
        init_path,
        submodule_search_locations=[pkg_dir],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load UI package from: {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ui"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    ui_pkg = _load_ui_package()
    raise SystemExit(ui_pkg.main())
