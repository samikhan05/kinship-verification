"""Gradio demo: upload a father, a mother and a child photo -> kinship probability.

    .venv/Scripts/python app.py          # Windows   (Linux/macOS: .venv/bin/python app.py)
    then open the address it prints (http://127.0.0.1:7860)

Runs on CPU. Needs the trained head and its validation scores, which come from a
Kaggle run: download outputs.zip and unzip it to kaggle_outputs/ (or have them in
outputs/). The logic is in src/kinship/predict.py.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kinship.predict import (DISCLAIMER, KinshipPredictor, find_artifacts, format_result,  # noqa: E402
                             load_calibration)
from kinship.utils import add_common_args, config_from_args, get_logger  # noqa: E402

MODEL = "E2-hardneg"     # best ROC-AUC on the test set (see docs/results/results.md)


def build_predictor(cfg: dict, log):
    from kinship.align import ArcFaceAligner
    from kinship.train import HeadScorer, load_head

    needed = [f"checkpoints/{MODEL}.pt", f"results/{MODEL}.json", f"scores/{MODEL}__val__image_xphoto.csv"]
    roots = [cfg["paths"]["output_dir"], ROOT / "outputs", ROOT / "kaggle_outputs" / "outputs"]
    outputs = find_artifacts(needed, roots)
    if outputs is None:
        sys.exit(f"Could not find {needed} in any of {[str(r) for r in roots]}.\n"
                 "Download outputs.zip from the Kaggle run and unzip it to kaggle_outputs/.")
    log.info(f"Using model {MODEL} from {outputs}")
    calibration = load_calibration(outputs, MODEL)
    head = load_head(outputs / "checkpoints" / f"{MODEL}.pt", "cpu")
    return KinshipPredictor(ArcFaceAligner(cfg, log), HeadScorer(head, "cpu"), calibration)


def main() -> None:
    import gradio as gr

    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--share", action="store_true", help="also create a temporary public Gradio link")
    args = ap.parse_args()
    cfg = config_from_args(args)
    log = get_logger("app")
    predictor = build_predictor(cfg, log)

    def run(father, mother, child):
        res = predictor.predict(father, mother, child)
        crops = res.get("crops", {})
        return crops.get("father"), crops.get("mother"), crops.get("child"), format_result(res)

    with gr.Blocks(title="Tri-subject kinship verification") as demo:
        gr.Markdown("# Is this child their child?\nUpload one face photo of the claimed **father**, "
                    "**mother** and **child**.\n\n**" + DISCLAIMER + "**")
        with gr.Row():
            inputs = [gr.Image(label=f"{r.capitalize()} photo", type="numpy", height=260)
                      for r in ("father", "mother", "child")]
        button = gr.Button("Estimate kinship", variant="primary")
        gr.Markdown("**Aligned faces the model actually sees** (detected, rotated and cropped to 112x112):")
        with gr.Row():
            crops = [gr.Image(label=f"{r} (aligned)", height=140, interactive=False)
                     for r in ("father", "mother", "child")]
        result = gr.Markdown(f"*{DISCLAIMER}*")
        button.click(run, inputs=inputs, outputs=crops + [result])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
