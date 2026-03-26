import contextlib
import io
from collections import defaultdict, Counter
import os, uuid, shutil
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, UploadFile, File, \
     Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont
from sort import Sort
from fastapi.responses import FileResponse
import json
from PIL import ImageFont, ImageDraw

from car_ocr import *
from motor_ocr import *

detection_model = YOLO('resources/yolo_detection_combined_40ep.pt')
motor_ocr_model = YOLO('resources/yolo_motor_ocr_50ep.pt')
car_ocr_model = YOLO('resources/yolo_ocr_car_v2_(10ep finetuned).pt')

font_path = "resources/BYekan.ttf"

#app = FastAPI(title="Plate Detection API")

from fastapi_standalone_docs import StandaloneDocs

# Disable built-in docs URLs or set them explicitly
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")

# Attach the offline serving of docs assets
StandaloneDocs(app=app)


UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve the outputs folder as static files (downloadable links)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


# 👇 Add this route once
@app.get("/files/{filename}", name="serve_file")
async def serve_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "file not found"}, status_code=404)
    return FileResponse(file_path)


# ---------- IMAGE ENDPOINT ----------
@app.post("/detect_image")
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="Save annotated output image")
):
    print()
    img_id = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_DIR, f"{img_id}_{file.filename}")
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Output file if visualize=True
    output_path = os.path.join(OUTPUT_DIR, f"{img_id}_output.jpg") if visualize else None

    results, annotated_image = detect_plate_image(
        img_or_path=img_path,
        post_process=True,
        visualize=visualize
    )

    if visualize and annotated_image is not None:
        cv2.imwrite(output_path, annotated_image)

    # Build download URL if file exists
    download_url = None
    if output_path and os.path.exists(output_path):
        base_url = str(request.base_url).rstrip("/")
        filename = os.path.basename(output_path)
        # download_url = f"{base_url}/files/{filename}"
        download_url = str(request.url_for("serve_file", filename=filename))

    return JSONResponse(content={
        "image_id": img_id,
        "results": results,
        "download_url": download_url
    })


# ---------- VIDEO ENDPOINT ----------
@app.post("/detect_video")
async def detect_video(
    request: Request,
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="Save annotated output video")
):
    vid_id = str(uuid.uuid4())
    vid_path = os.path.join(UPLOAD_DIR, f"{vid_id}_{file.filename}")
    with open(vid_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Output file if visualize=True
    output_path = os.path.join(OUTPUT_DIR, f"{vid_id}_output.mp4") if visualize else None

    results = process_video_with_tracking(
        video_path=vid_path,
        output_path=output_path,
        visualize=visualize
    )

    # Build download URL if file exists
    download_url = None
    if output_path and os.path.exists(output_path):
        base_url = str(request.base_url).rstrip("/")
        filename = os.path.basename(output_path)
        download_url = f"{base_url}/files/{filename}"

    print(json.dumps({
    "video_id": vid_id,
    "results": results,
    "download_url": download_url
    }, default=str, indent=2))


    return JSONResponse(content={
        "video_id": vid_id,
        "results": results,
        "download_url": download_url
    })


def draw_text(draw, text, box_coords, text_margin, font_path, color="black"):
    font_size = 26
    x, y, w, h = box_coords
    text_margin_x , text_margin_y = text_margin

    while font_size > 5:
        font = ImageFont.truetype(font_path, font_size)

        # LOGIC: If it fits inside the plate image width, BREAK the loop
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # shrink ONLY if it doesn't fit
        if text_w <= w and text_h <= h:
            break

        font_size -= 1

    draw.text((x + text_margin_x, y - text_margin_y), text, font=font, fill=color)

# track with crnn:
def process_video_with_tracking(video_path, output_path, post_process=True, yolo_char_confidence=0.55, visualize=False):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Only initialize writer if saving video
    out = None
    if visualize:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Sort(max_age=15)  # Initialize SORT tracker

    frame_count = 0
    pre = 0
    total_start = time.time()

    # 🔸 For saving best plate per track_id
    track_ocr_history = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(frame_count)

        with contextlib.redirect_stdout(io.StringIO()):
            detection = detection_model.predict(frame, verbose=False, agnostic_nms=True)[0]

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        dets_for_sort = []
        detection_info = []  # (x1, y1, x2, y2, conf, label)

        for box in detection.boxes:
            cls_id = int(box.cls)
            label = detection.names[cls_id]
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            if label in ['car', 'motorcycle']:
                dets_for_sort.append([x1, y1, x2, y2, conf])
                detection_info.append((x1, y1, x2, y2, conf, label))

        dets_for_sort = np.array(dets_for_sort) if dets_for_sort else np.empty((0, 5))
        tracked_objects = tracker.update(dets_for_sort)

        for i, track in enumerate(tracked_objects):
            x1, y1, x2, y2, track_id = map(int, track)

            matching_label = None
            for dx1, dy1, dx2, dy2, _, label in detection_info:
                if abs(dx1 - x1) < 10 and abs(dy1 - y1) < 10:
                    matching_label = label
                    break

            if matching_label is None:
                continue

            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0 or plate_img.shape[0] == 0 or plate_img.shape[1] == 0:
                print(f"[WARNING] Empty crop at frame {frame_count}, box: ({x1},{y1},{x2},{y2}) — skipped")
                continue

            # OCR prediction
            pred_text = "?"
            color = "gray"
            width = 2
            if label == 'car':
                yolo_result = car_ocr_model.predict(plate_img, agnostic_nms=True)
                pred_text = get_yolo_ocr_result_car(yolo_result, conf_thresh=yolo_char_confidence)
                print('car before post proccess: ', pred_text)
                if post_process and len(pred_text) != 8:  # todo it may not be enough to check 8 ...
                    pred_text = car_post_process(yolo_result, car_ocr_model, plate_img.shape[:2])
                    print('car after post proccess: ', pred_text)
                color = "yellow"
                width = 4
                # Prepare the data for inserting the plate image
                plate_w = 150
                plate_h = image_margin = 50
                text_margin_x = 20
                text_margin_y = 35
                overlay = Image.open("./resources/car_plate.jpg").convert("RGBA")
                overlay = overlay.resize((plate_w, plate_h), Image.Resampling.LANCZOS)


            elif label == 'motorcycle':
                with contextlib.redirect_stdout(io.StringIO()):
                    pred_text = get_yolo_ocr_result_motor(plate_img, motor_ocr_model, conf_thresh=0.75)
                print('motor before post proccess: ', pred_text)
                color = "red"
                width = 4
                # Prepare the data for inserting the plate image
                plate_w = 90
                plate_h = image_margin = 60
                text_margin_x = 10
                text_margin_y = 50
                overlay = Image.open("./resources/motor_plate.jpg").convert("RGBA")
                overlay = overlay.resize((plate_w, plate_h), Image.Resampling.LANCZOS)

            else:
                print('matching_label does not match with any!')
                continue

            # 🔸 Save this prediction for the track_id
            if isinstance(pred_text, str) and pred_text.strip():
                track_ocr_history[track_id].append((label, frame_count, pred_text))
                print("added to history = ", (frame_count, pred_text))

            if visualize:
                # 🔸 Compute the max prediction so far for this track
                history_texts = [t for _, _, t in track_ocr_history[track_id]]
                preds = [p.replace("[", "").replace("]", "") for p in history_texts]
                max_pred_text = None
                if history_texts:
                    counts = Counter(preds)
                    max_pred_text, max_count = counts.most_common(1)[0]

                # Draw on frame
                try:
                    if max_pred_text.count('?') != 8:
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                        # insert plate image
                        draw = ImageDraw.Draw(pil_image)
                        # if 
                        pil_image.paste(overlay, (x1, y1 - image_margin), overlay)

                        # pil_image.paste(overlay, (x1, y1 - image_margin), overlay)
                        # draw.text((x1, y1 - 30), f"({track_id}): {pred_text}", font=font, fill="yellow")
                        # Max prediction so far
                        if max_pred_text :
                            # change the format for the plate image
                            if label == 'car':
                                max_pred_text = max_pred_text[:6] + " " + max_pred_text[6:]
                            elif label == 'motorcycle':
                                max_pred_text = "  " + max_pred_text[:3] + "\n" + max_pred_text[4:]

                            print("Writing on the image", max_pred_text)
                            clean_text = to_persian_numbers(max_pred_text)
                            clean_text = fix_persian_text(clean_text)
                            draw_text(draw, clean_text, 
                                      (x1, y1, plate_w - 20, plate_h), 
                                      (text_margin_x, text_margin_y),
                                      font_path)

                except Exception as e:
                    print(f"[WARNING] draw.text() failed at frame {frame_count} for ID {track_id}: {e}")

        # Write video frame if saving
        if visualize:
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            out.write(frame)
            print(f"[DEBUG] Frame {frame_count} written to output.")

        if frame_count != pre + 1:
            print('###################')
            print('ohhhhh nooooo!')
            print('⚠ Frame jump detected')
            print('###################')

        pre = frame_count

        # Real-time processing  
        # frame = cv2.resize(frame, (1280, 640))
        # cv2.imshow("Real-Time", frame)
        # time.sleep(0.03)  # ~30 FPS

        # if cv2.waitKey(1) == 27:
            # break

    cap.release()
    cv2.destroyAllWindows()
    if visualize and out is not None:
        out.release()
    print("all closed!")

    total_time = time.time() - total_start
    effective_fps = frame_count / total_time

    print(f"[INFO] Processed {frame_count} frames")
    print(f"[INFO] Total time: {total_time:.2f} sec")
    print(f"[INFO] Effective FPS: {effective_fps:.2f}")

    results = []
    result_id = 1
    for idx, (track_id, history) in enumerate(track_ocr_history.items(), start=1):
        if not history:  # skip empty histories
            print(f"[WARNING] Track {track_id} has empty history, skipping")
            continue

        try:
            labels, frame_nums, raw_preds = zip(*history)
        except Exception as e:
            print(f"[ERROR] Could not unzip history for track {track_id}: {history}")
            raise e

        # Sanitize predictions: remove "[" and "]"
        preds = [p.replace("[", "").replace("]", "") for p in raw_preds]
        pred_counter = Counter(preds)
        most_common_pred, most_common_count = pred_counter.most_common(1)[0]
        unknown_num = most_common_pred.count('?')
        most_common_pred = make_rtl(most_common_pred)
        total_preds = len(preds)
        confidence = round(most_common_count / total_preds, 2)  # confidence between 0.0 and 1.0
        start_frame = min(frame_nums)
        end_frame = max(frame_nums)

        # invalid prediction
        if unknown_num == 8:
            pass
        else: 
            results.append({
                # "track_id": track_id,
                "label": labels[0],
                "id": result_id,
                "most_common_prediction": most_common_pred,
                "unknown_num": unknown_num,
                "confidence": confidence,
                "start_time_sec": round(start_frame / fps, 2),
                "end_time_sec": round(end_frame / fps, 2)
            })
            result_id += 1

    # Pretty print
    print("id - best plate")
    for row in results:
        print(row["id"], ' - ', row["most_common_prediction"])

    return results


def detect_plate_image(img_or_path, post_process, yolo_char_confidence=0.55, visualize=False):
    if isinstance(img_or_path, str):
        image = cv2.imread(img_or_path)
        if image is None:
            raise ValueError(f"Could not read image from {img_or_path}")
    elif isinstance(img_or_path, np.ndarray):
        image = img_or_path
    else:
        raise TypeError("img_or_path must be str path or np.ndarray image")

    # detection:
    with contextlib.redirect_stdout(io.StringIO()):
        detection = detection_model.predict(image, verbose=False, agnostic_nms=True)[0]

    if visualize:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, 26)

    results = []

    for box in detection.boxes:
        cls_id = int(box.cls)
        label = detection.names[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()

        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0 or plate_img.shape[0] == 0 or plate_img.shape[1] == 0:
            print(f"[WARNING] Empty crop, box: ({x1},{y1},{x2},{y2}) — skipped")
            continue

        # OCR prediction
        pred_text = "?"
        color = "gray"
        width = 2
        pred_text2 = None

        if label == 'car':
            yolo_result = car_ocr_model.predict(plate_img, agnostic_nms=True)
            pred_text = get_yolo_ocr_result_car(yolo_result, conf_thresh=yolo_char_confidence)
            color = "yellow"
            width = 4
            print('car before post proccess: ', pred_text)
            if post_process and len(pred_text) != 8:  # todo it may not be enough to check 8 ...
                pred_text = car_post_process(yolo_result, car_ocr_model, plate_img.shape[:2])
                print('car after post proccess: ', pred_text)


        elif label == 'motorcycle':
            with contextlib.redirect_stdout(io.StringIO()):
                pred_text = get_yolo_ocr_result_motor(plate_img, motor_ocr_model, conf_thresh=0.75)
            # print('motor before post proccess: ', pred_text)
            color = "red"
            width = 3
        else:
            print('matching_label does not match with any!')
            continue

        if visualize:
            
            font = ImageFont.truetype(font_path, 26)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw.text((x1, max(0, y1 - 40)), pred_text, font=font, fill='yellow')

        # Collect result
        results.append({
            "label": label,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "plate_text": pred_text
        })

    if visualize:
        # image = np.array(pil_image)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return results, image

