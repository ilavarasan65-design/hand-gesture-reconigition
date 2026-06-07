"""
demo.py — Real-time hand gesture recognition demo using Gradio + webcam.

Usage:
    python scripts/demo.py --model checkpoints/best_model.pt --language ASL
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.extractor import KeypointExtractor
from src.data.graph_utils import keypoints_to_graph
from src.models.fusion_model import GestureClassifier
from torch_geometric.data import Batch


LANGUAGE_IDS = {'ASL': 0, 'BSL': 1, 'ISL': 2, 'CSL': 3, 'ArSL': 4, 'FSL': 5}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    label_to_idx = ckpt.get('label_to_idx', {})
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = GestureClassifier(
        num_classes=len(label_to_idx),
        embed_dim=cfg.get('model', {}).get('embed_dim', 256),
        graph_type=cfg.get('model', {}).get('graph_type', 'gcn'),
        use_image=False,  # demo: graph-only for speed
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, idx_to_label


def predict_frame(frame_rgb, model, extractor, idx_to_label, language, device, top_k=5):
    """
    Run inference on a single RGB frame.

    Returns:
        annotated_frame: numpy array with drawn keypoints
        prediction: dict {gesture: confidence}
        raw_kp: keypoints array or None
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    kp, handedness = extractor.extract_from_image(frame_bgr)

    if kp is None:
        annotated = frame_rgb.copy()
        cv2.putText(annotated, 'No hand detected', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 80, 80), 2)
        return annotated, {}, None

    # Draw keypoints
    annotated = extractor.visualize(frame_bgr, kp)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    cv2.putText(annotated, f'{handedness} hand', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 2)

    # Build graph
    graph = keypoints_to_graph(kp)
    batch = Batch.from_data_list([graph]).to(device)
    lang_id = torch.tensor([LANGUAGE_IDS.get(language, 0)], device=device)

    with torch.no_grad():
        logits, _ = model(batch, None, lang_id)
        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_idxs = probs.topk(min(top_k, len(idx_to_label)))

    results = {
        idx_to_label.get(idx.item(), f'class_{idx.item()}'): round(prob.item() * 100, 1)
        for prob, idx in zip(top_probs, top_idxs)
    }

    # Overlay top prediction
    top_label = list(results.keys())[0]
    top_conf = list(results.values())[0]
    cv2.putText(annotated, f'{top_label}: {top_conf:.1f}%', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 200, 255), 2)

    return annotated, results, kp


def build_demo(model, extractor, idx_to_label, device):
    def run_inference(image, language):
        if image is None:
            return None, {}
        frame = np.array(image)
        annotated, results, _ = predict_frame(
            frame, model, extractor, idx_to_label, language, device)
        return Image.fromarray(annotated), results

    with gr.Blocks(title='Multi-Culture Sign Language Recognition') as demo:
        gr.Markdown('## Hand Gesture Recognition — Multi-Culture Sign Language')
        gr.Markdown('Upload an image or use webcam. Select the sign language variant.')

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    sources=['webcam', 'upload'],
                    label='Input',
                    streaming=True,
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGE_IDS.keys()),
                    value='ASL',
                    label='Sign Language',
                )
                run_btn = gr.Button('Recognize', variant='primary')
            with gr.Column():
                image_output = gr.Image(label='Annotated output')
                pred_output = gr.Label(label='Top predictions', num_top_classes=5)

        run_btn.click(run_inference, inputs=[image_input, language],
                      outputs=[image_output, pred_output])
        image_input.stream(run_inference, inputs=[image_input, language],
                           outputs=[image_output, pred_output])

        gr.Markdown(
            '**Supported languages:** ASL (American), BSL (British), '
            'ISL (Indian), CSL (Chinese), ArSL (Arabic), FSL (French)'
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--language', default='ASL')
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    parser.add_argument('--port', type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model from {args.model} on {device}')

    model, idx_to_label = load_model(args.model, device)
    extractor = KeypointExtractor()

    print(f'Model loaded. {len(idx_to_label)} gesture classes.')
    demo = build_demo(model, extractor, idx_to_label, device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()
