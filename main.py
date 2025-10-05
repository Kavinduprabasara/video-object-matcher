import cv2
from ultralytics import YOLO
import supervision as sv
from color_detection import get_dominant_color, get_color_name

video = cv2.VideoCapture("video.mp4")

model = YOLO("yolov8n.pt")

bbox_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()



while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[(detections.confidence > 0.5)]
        
        labels = []
        for i, class_id in enumerate(detections.class_id):
            class_name = results.names[class_id]
            
            if class_name == "person":

                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                

                person_roi = frame[y1:y2, x1:x2]

                height = y2 - y1
                upper_body = person_roi[int(height*0.2):int(height*0.6), :]
                
                if upper_body.size > 0:
                    dominant_color = get_dominant_color(upper_body)
                    color_name = get_color_name(dominant_color)
                    label = f"{class_name} - {color_name}"
                else:
                    label = class_name
            else:
                label = class_name
            
            labels.append(label)

        frame = bbox_annotator.annotate(
            scene=frame,
            detections=detections
        )
        
        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()