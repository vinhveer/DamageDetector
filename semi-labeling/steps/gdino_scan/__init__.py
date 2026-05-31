"""GroundingDINO scan (upstream of resemi step01).

Thin wrapper over object_detection.damage_scan.cli: detects crack/mold/spall
boxes and writes them to damage_scan.sqlite3 (the detections table that the
OpenCLIP step reads). CLI in main.py.
"""
