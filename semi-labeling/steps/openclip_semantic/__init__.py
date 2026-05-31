"""OpenCLIP semantic scoring (upstream of resemi step01).

Crops the GDINO boxes from damage_scan.sqlite3, classifies each with OpenCLIP
(crack/mold/spall + negative anchors), and writes openclip_semantic_results —
the input table resemi step01 reads. CLI in main.py.
"""
