import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)[0]
        annotated_frame = results.plot()
        '''for result in results:
            if result.boxes.conf[0]>0.7:
                cv2.rectangle(frame, (int(result.boxes.xyxy[0][0]), int(result.boxes.xyxy[0][1])), (int(result.boxes.xyxy[0][2]), int(result.boxes.xyxy[0][3])), (255, 0, 0), 3)
                str="{0},{1:.2f}".format(model.names[int(result.boxes.cls)],result.boxes.conf[0])
                cv2.putText(frame,str,(int(result.boxes.xyxy[0][0]), int(result.boxes.xyxy[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
'''




        #print(results.boxes.conf)

        # seg image
        '''gray = results.masks[0].data[0]
        pred_masks_np = gray.detach().cpu().numpy()
        xxx = np.array(pred_masks_np, dtype="uint8") * 255
        cv2.imshow("YOLOv8 seg", xxx)

        #for result in results:## 전체
        for result in results[0]:##처음 ㅇobject 만
            for keypoint_indx, keypoint in enumerate(result.masks.xy[0]):
                cv2.putText(annotated_frame, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
'''
        # pose image
        '''for result in results:
            for keypoint_indx,keypoint in enumerate(result.keypoints.tolist()):
                cv2.putText(annotated_frame,str(keypoint_indx),(int(keypoint[0]),int(keypoint[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
'''
        # Visualize the results on the frame


        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
       # cv2.imshow("YOLOv8 Inference2", frame)
        # Break the loop if 'q' is pressed

    else:
        # Break the loop if the end of the video is reached
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()