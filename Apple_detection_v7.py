'''
Implement  by vaibhav Kurrey
E-mail = vkurrey3.vk@gmail.com
'''

import cv2
from yolov7 import YOLOv7
import glob

# Initialize yolov7 object detector
model_path = "yolov7.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.2, iou_thres=0.3)

class Work_Model:
   def predict(self, imgPath):
      print('work')

      # Read image
      imgPath = cv2.imread(imgPath)

      # Detect Objects
      boxes, scores, class_ids = yolov7_detector(imgPath)
      combined_img = yolov7_detector.draw_detections(imgPath)

      # Return bounding box coordinates for first detected object as a tuple
      if len(boxes) > 0:
         
          bbox_list = []
          for i in range(len(boxes)):
              x_min, y_min, x_max, y_max = map(int, boxes[i])
              bbox = [x_min, y_min, x_max, y_max]
              bbox_list.append(bbox)
          print(f"{len(bbox_list)} objects detected at coordinates: {bbox_list}")
          combined_img = yolov7_detector.draw_detections(imgPath)
          cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
          
          cv2.imshow("Detected Objects", combined_img)
          cv2.imwrite("detected_objects.jpg", combined_img)
          cv2.waitKey(0)
          return bbox_list
      else:
          print("No objects detected.")
          return None



if __name__ == "__main__":
   model = Work_Model()
   path = 'test/*.jpg'
   for imgPath in glob.glob(path):
      print(model.predict(imgPath))













































