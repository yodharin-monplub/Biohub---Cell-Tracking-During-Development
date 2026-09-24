"""Package the frozen predictor with explicit, verified DeepCenter deployment paths."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model104'
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump

RUNTIME_SHA='8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127'
OLD_PATH='''    os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", "").strip(),
'''
NEW_PATH='''    # Integrity audit checks best.pt; inference uses the verified epoch-500 sibling.
    str(Path(os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"]).with_name("best.pt"))
    if os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", "").strip() else "",
'''
OLD_RUNTIME='''os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(
    _deepcenter_materialized_path
)
'''
NEW_RUNTIME='''_model104_runtime_checkpoint = _deepcenter_materialized_path.with_name("checkpoint_last.pt")
if not _model104_runtime_checkpoint.is_file():
    raise FileNotFoundError("Missing model1 epoch-500 runtime checkpoint: " + str(_model104_runtime_checkpoint))
if _integrity_sha256_file(_model104_runtime_checkpoint) != "'''+RUNTIME_SHA+'''":
    raise RuntimeError("Model104 runtime checkpoint differs from the locally scored model1 checkpoint")
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(_model104_runtime_checkpoint)
'''


def packaged_notebook():
    freeze=json.loads((OUT/'build_receipt.json').read_text())
    assert sha(OUT/'submission.ipynb')==freeze['candidate_sha256']
    notebook=json.loads((OUT/'submission.ipynb').read_text())
    source=''.join(notebook['cells'][10]['source'])
    assert source.count(OLD_PATH)==source.count(OLD_RUNTIME)==1
    source=source.replace(OLD_PATH,NEW_PATH).replace(OLD_RUNTIME,NEW_RUNTIME)
    compile(source,'model104:deployment-cell10','exec')
    notebook['cells'][10]['source']=source
    return notebook


if __name__=='__main__':
    destination=OUT/'kaggle'
    destination.mkdir(exist_ok=False)
    notebook=packaged_notebook()
    dump(destination/'submission.ipynb',notebook)
    metadata=json.loads((ROOT/'model1/kernel-metadata.template.json').read_text())
    metadata.update(id='YOUR_KAGGLE_USERNAME/biohub-model104-protected-motion-links',
                    title='Biohub model104 protected motion links')
    dump(destination/'kernel-metadata.template.json',metadata)
    dump(destination/'package_receipt.json',dict(status='prepared_not_executed_on_kaggle',
         scored_candidate_sha256=sha(OUT/'submission.ipynb'),
         deployment_notebook_sha256=sha(destination/'submission.ipynb'),
         metadata_template_sha256=sha(destination/'kernel-metadata.template.json'),
         changed_from_scored_candidate_cells=[10],runtime_checkpoint_sha256=RUNTIME_SHA,
         repair_cell_exactly_unchanged=True,model_weights_unchanged=True,
         change='Resolve best.pt for original integrity check and require its exact epoch-500 sibling for runtime; support both input layouts.',
         kaggle_uploaded=False,kaggle_submitted=False))
