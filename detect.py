#!/usr/bin/env python3
"""
detect.py - Gender & Age detection

Features:
- Uses OpenCV DNN Caffe models (same filenames as in your project).
- Safe ROI slicing + explicit resize to avoid empty-ROI crashes.
- Optionally run on a single image (--image) or camera (default).
- Skip frames to reduce CPU load (--skip N).
- Choose backend/target for OpenCV DNN (cpu/opencl/cuda) with --target.
- Graceful exit on 'q' or Ctrl+C, prints timings for profiling.
- Optional: save detected face crops (--save-faces) into ./faces/
"""

import cv2
import argparse
import time
import os

def highlightFace(net, frame, conf_threshold=0.7):
    """Run face detector and return frame with boxes + list of boxes [x1,y1,x2,y2]."""
    frame_opencv = frame.copy()
    h, w = frame_opencv.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_opencv, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)

    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # clamp coords to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 - x1 > 0 and y2 - y1 > 0:
                boxes.append([x1, y1, x2, y2])
                thickness = max(1, int(round(h / 150)))
                cv2.rectangle(frame_opencv, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    return frame_opencv, boxes

def parse_args():
    p = argparse.ArgumentParser(description="Gender & Age Detection (robust detect.py)")
    p.add_argument('--image', help='Path to image file (optional). If omitted, use webcam 0.', default=None)
    p.add_argument('--skip', type=int, default=2, help='Process every N-th frame (default 2).')
    p.add_argument('--conf', type=float, default=0.7, help='Face detection confidence threshold.')
    p.add_argument('--save-faces', action='store_true', help='Save detected face crops into ./faces/')
    p.add_argument('--target', choices=['cpu', 'opencl', 'cuda'], default='cpu',
                   help='DNN target: cpu (default), opencl (if supported), or cuda (if built for CUDA).')
    return p.parse_args()

def set_dnn_target(net, target):
    # prefer OpenCV backend, pick target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    if target == 'cpu':
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    elif target == 'opencl':
        # may accelerate on some systems
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    elif target == 'cuda':
        # requires opencv with cuda dnn
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            # fallback
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def main():
    args = parse_args()

    # --- model filenames (keep same names as your repo)
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    # model mean and labels
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # check model files exist (helpful early error)
    needed = [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]
    missing = [f for f in needed if not os.path.isfile(f)]
    if missing:
        print("ERROR: Missing model files:", missing)
        print("Place these files in the same folder as detect.py or update filenames.")
        return

    # load networks
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # set backend/target for faceNet (affects speed)
    set_dnn_target(faceNet, args.target)
    set_dnn_target(ageNet, args.target)
    set_dnn_target(genderNet, args.target)

    # prepare capture
    capture_src = args.image if args.image else 0
    video = cv2.VideoCapture(capture_src)
    if not video.isOpened():
        print("ERROR: Could not open capture:", capture_src)
        return

    if args.save_faces:
        os.makedirs("faces", exist_ok=True)
    padding = 20
    process_every_n = max(1, args.skip)
    frame_count = 0
    last_boxes = []

    try:
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                # If single image, break after first read
                if args.image:
                    break
                # for camera, try again or break
                print("Warning: frame not received; exiting.")
                break

            frame_count += 1
            t_frame_start = time.time()

            # run face detection every Nth frame (to save CPU)
            if frame_count % process_every_n == 0:
                t0 = time.time()
                resultImg, boxes = highlightFace(faceNet, frame, conf_threshold=args.conf)
                t1 = time.time()
                last_boxes = boxes
                detect_time = t1 - t0
            else:
                # reuse previous resultImg if available; otherwise show current frame
                if 'resultImg' in locals():
                    resultImg = resultImg.copy()
                else:
                    resultImg = frame.copy()
                boxes = last_boxes
                detect_time = 0.0

            # for each detected face, run gender+age
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                # apply padding and clamp coords
                y1c = max(0, y1 - padding)
                y2c = min(frame.shape[0] - 1, y2 + padding)
                x1c = max(0, x1 - padding)
                x2c = min(frame.shape[1] - 1, x2 + padding)

                face = frame[y1c:y2c, x1c:x2c]
                if face is None or face.size == 0:
                    print("Skipping empty ROI:", (x1c, y1c, x2c, y2c))
                    continue

                # explicit resize to model's expected input
                try:
                    face_resized = cv2.resize(face, (227, 227))
                except cv2.error as e:
                    print("cv2.resize failed:", e)
                    continue

                blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False)

                # gender
                genderNet.setInput(blob)
                g_preds = genderNet.forward()
                gender = genderList[int(g_preds[0].argmax())]

                # age
                ageNet.setInput(blob)
                a_preds = ageNet.forward()
                age = ageList[int(a_preds[0].argmax())]

                label = f"{gender}, {age}"
                cv2.putText(resultImg, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

                # optionally save face crops
                if args.save_faces:
                    fname = os.path.join("faces", f"face_{frame_count:06d}_{i}.jpg")
                    cv2.imwrite(fname, face_resized)

                # print small summary to console (you can remove to reduce log noise)
                print(f"[Frame {frame_count}] {label} at {x1},{y1},{x2},{y2} (detect {detect_time:.3f}s)")

            cv2.imshow("Detecting age and gender", resultImg)

            # If running on a single image, show the window and wait indefinitely
            if args.image:
                print("Image mode: press any key in the image window to exit...")
                cv2.waitKey(0)  # waits until you press any key
                break

            # Webcam mode: press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting by user request ('q' pressed).")
                break


            # optional: small perf print every few frames
            if frame_count % (10 * process_every_n) == 0:
                total_time = time.time() - t_frame_start
                print(f"[Perf] frame {frame_count}, last detect {detect_time:.3f}s, frame time {total_time:.3f}s")

    except KeyboardInterrupt:
        print("Interrupted by user (KeyboardInterrupt). Exiting...")
    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
