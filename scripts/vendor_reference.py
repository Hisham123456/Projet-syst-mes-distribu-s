"""Extract only model class definitions, never execute the upstream notebook."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "reference/zenodo/FDR26_benchmark.ipynb"
expected = "878779012040649ab145b05d51215d9c"
if hashlib.md5(source.read_bytes()).hexdigest() != expected:
    raise SystemExit("Upstream notebook checksum differs. Review the new notebook before extraction.")
notebook = json.loads(source.read_text(encoding="utf-8"))
classes = {"SEBlock", "ChannelAttention", "SpatialAttention", "CBAM", "AttentionClassifier"}
nodes = []
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    try:
        tree = ast.parse("".join(cell["source"]))
    except SyntaxError:
        continue
    nodes.extend(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in classes)
if {n.name for n in nodes} != classes or len(nodes) != 5:
    raise SystemExit("Expected model classes not found uniquely")
header = '''"""Model definitions from FDR26_benchmark.ipynb, Zenodo 20605487.
Extracted without functional changes using scripts/vendor_reference.py.
Upstream README declares MIT for code; see reference/PROVENANCE.md.
Only AttentionClassifier(convnext_tiny) is used; RETFound is not supported here.
"""
import torch
from torch import nn
import timm

NUM_CLASSES = 5
IMG_SIZE = 384
DROPOUT = 0.5

'''
target = ROOT / "src/dr_project/reference_model.py"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(header + "\n\n".join(ast.unparse(n) for n in nodes) + "\n", encoding="utf-8")
print(target)
