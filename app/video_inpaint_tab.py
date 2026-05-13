"""Video Inpainting tab for the Gradio web UI.

Provides the UI components and backend logic for:
  Upload video -> first-frame click to select object -> SAM + OSTrack + STTN pipeline
"""

import os
import tempfile

import cv2
import gradio as gr
import numpy as np
import torch
import imageio.v2 as iio
from PIL import Image

from ostrack import build_ostrack_model, get_box_using_ostrack
from sttn_video_inpaint import build_sttn_model, inpaint_video_with_builded_sttn
from pytracking.lib.test.evaluation.data import Sequence
from utils import dilate_mask


MIN_BBOX_AREA = 100


def _ensure_video_models(model, args):
    """Lazy-load OSTrack and STTN on first video use."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'ostrack' not in model:
        print("[Video] Loading OSTrack tracker ...")
        model['ostrack'] = build_ostrack_model(args.tracker_ckpt)
    if 'sttn' not in model:
        print("[Video] Loading STTN video inpainter ...")
        model['sttn'] = build_sttn_model(
            ckpt_p=args.vi_ckpt, model_type="sttn", device=device)


def _video_upload(video_path):
    """Extract the first frame from the uploaded video for point selection."""
    if not video_path:
        return None, None, []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None, []
    first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return first_frame, first_frame.copy(), []


def build_video_tab(model, args):
    """Create and wire the Video Inpainting tab. Call inside gr.Tabs().

    Closures capture `model` and `args` so Gradio sees the correct
    function signatures (especially the `evt: gr.SelectData` annotation).
    """

    def _process_click(first_frame_orig, vid_clicked_points, evt: gr.SelectData):
        if first_frame_orig is None:
            return None, vid_clicked_points, None

        x, y = evt.index
        vid_clicked_points.append((x, y, 1))

        frame = np.array(first_frame_orig, dtype=np.uint8)
        H, W = frame.shape[:2]

        model['sam'].set_image(frame)
        points = np.array([(px, py) for px, py, _ in vid_clicked_points])
        labels = np.array([lab for _, _, lab in vid_clicked_points])
        masks, scores, _ = model['sam'].predict(
            point_coords=points, point_labels=labels, multimask_output=True)
        model['sam'].reset_image()

        best_mask = masks[scores.argmax()]

        mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
        mask_vis[best_mask > 0] = [0, 255, 0]

        overlay = frame.copy()
        overlay = cv2.addWeighted(overlay, 1.0, mask_vis, 0.5, 0)
        for px, py, _ in vid_clicked_points:
            cv2.circle(overlay, (px, py), 8, (255, 0, 0), -1)

        return overlay, vid_clicked_points, (best_mask * 255).astype(np.uint8)

    def _remove_object(video_path, first_frame_orig, vid_clicked_points,
                       vid_dilate_kernel):
        if video_path is None or first_frame_orig is None or not vid_clicked_points:
            return None

        _ensure_video_models(model, args)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dk = int(vid_dilate_kernel) if vid_dilate_kernel else 15

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tmp_dir = tempfile.mkdtemp(prefix="ia_video_")
        all_frames, frame_paths = [], []
        idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            all_frames.append(rgb)
            p = os.path.join(tmp_dir, f"{idx:06d}.png")
            iio.imwrite(p, rgb)
            frame_paths.append(p)
            idx += 1
        cap.release()

        if len(all_frames) == 0:
            return None

        # SAM on first frame
        key_frame = all_frames[0]
        model['sam'].set_image(key_frame)
        points = np.array([(px, py) for px, py, _ in vid_clicked_points])
        labels = np.array([lab for _, _, lab in vid_clicked_points])
        masks, scores, _ = model['sam'].predict(
            point_coords=points, point_labels=labels, multimask_output=True)
        model['sam'].reset_image()

        key_mask = masks[scores.argmax()]
        if dk > 0:
            key_mask = dilate_mask(key_mask, dk)

        x, y, w, h = cv2.boundingRect(key_mask.astype(np.uint8))
        if w * h < MIN_BBOX_AREA:
            raise gr.Error(
                "Selected region is too small for tracking. "
                "Please click on a larger, more distinct object.")

        # OSTrack: track bounding box across all frames
        init_box = np.array([x, y, w, h], dtype=np.float32).reshape(-1, 4)
        seq = Sequence("web_video", frame_paths, 'inpaint-anything', init_box)
        print("[Video] Tracking ...")
        all_boxes = get_box_using_ostrack(model['ostrack'], seq)

        # SAM per-frame guided by tracked boxes
        print("[Video] Segmenting per-frame ...")
        all_masks = [key_mask]
        ref_mask = key_mask
        for i in range(1, len(all_frames)):
            frame = all_frames[i]
            bx, by, bw, bh = all_boxes[i]
            sam_box = np.array([bx, by, bx + bw, by + bh])

            model['sam'].set_image(frame)
            m, sc, _ = model['sam'].predict(box=sam_box, multimask_output=True)
            model['sam'].reset_image()

            mse = np.mean(
                (m.astype(np.int32) - ref_mask.astype(np.int32)) ** 2,
                axis=(-2, -1))
            best = m[mse.argmin()]
            if dk > 0:
                best = dilate_mask(best, dk)
            ref_mask = best
            all_masks.append(best)

        # STTN: temporally-coherent video inpainting
        print("[Video] Inpainting ...")
        pil_frames = [Image.fromarray(f) for f in all_frames]
        pil_masks = [Image.fromarray(np.uint8(m * 255)) for m in all_masks]

        with torch.no_grad():
            comp_frames = inpaint_video_with_builded_sttn(
                model['sttn'], pil_frames, pil_masks, device=device)

        out_path = os.path.join(tmp_dir, "result.mp4")
        result_arrays = [np.array(f) for f in comp_frames]
        iio.mimwrite(out_path, result_arrays, fps=fps)
        print(f"[Video] Done -> {out_path}")

        return out_path

    def _reset_video(*_args):
        return None, None, None, [], None

    # ---- UI layout ----

    with gr.Tab("Video Inpainting"):
        vid_clicked_points = gr.State([])
        first_frame_orig = gr.State(None)
        vid_mask_preview = gr.State(None)

        gr.Markdown(
            "Upload a video, click on the first frame to select the object "
            "you want to remove, then press **Remove Object from Video**."
        )

        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("## Upload Video")
                video_input = gr.Video(label="Input Video")
                vid_first_frame = gr.Image(
                    type="numpy",
                    label="First Frame (click to select object)",
                    height=400,
                )
                vid_dilate = gr.Slider(
                    label="Dilate Kernel Size", minimum=0, maximum=30,
                    step=1, value=15,
                )

            with gr.Column(variant="panel"):
                gr.Markdown("## Result")
                remove_video_btn = gr.Button(
                    "Remove Object from Video", variant="primary")
                clear_video_btn = gr.Button(
                    value="Reset", variant="secondary")
                video_output = gr.Video(label="Output Video")

        video_input.change(
            _video_upload,
            inputs=[video_input],
            outputs=[vid_first_frame, first_frame_orig, vid_clicked_points],
        )

        vid_first_frame.select(
            _process_click,
            inputs=[first_frame_orig, vid_clicked_points],
            outputs=[vid_first_frame, vid_clicked_points, vid_mask_preview],
            show_progress=True,
            queue=True,
        )

        remove_video_btn.click(
            _remove_object,
            inputs=[video_input, first_frame_orig,
                    vid_clicked_points, vid_dilate],
            outputs=[video_output],
            show_progress=True,
            queue=True,
        )

        clear_video_btn.click(
            _reset_video,
            inputs=[video_input, vid_first_frame, video_output,
                    vid_clicked_points, first_frame_orig],
            outputs=[video_input, vid_first_frame, video_output,
                     vid_clicked_points, first_frame_orig],
        )
