from __future__ import annotations


def main() -> int:
    print(
        "Available UI apps:\n"
        "  - python -m ui.editor_app\n"
        "  - python -m ui.create_data_tools.cropper_app\n"
        "  - streamlit run ui/streamlit_demo/app.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
