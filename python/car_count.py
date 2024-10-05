import cv2
import numpy as np
import torch.cuda
from ultralytics import YOLO


class TraditionModel:
    def __init__(self):
        super().__init__()
        self.min_w = 90
        self.min_h = 90
        self.motion = cv2.createBackgroundSubtractorMOG2()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __call__(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 去噪（高斯）
        blur = cv2.GaussianBlur(frame, (3, 3), 5)
        # cv2.imshow("blur", frame)
        # 去背影
        mask = self.motion.apply(blur)
        # cv2.imshow("motion", mask)
        # 腐蚀， 去掉图中小斑块
        erode = cv2.erode(mask, self.kernel)
        # cv2.imshow("erode", erode)
        # 膨胀， 还原放大
        dilate = cv2.dilate(erode, self.kernel, iterations=3)
        # cv2.imshow("dilate", dilate)
        # 闭操作，去掉物体内部的小块
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        # cv2.imshow("close", close)
        contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cars = []
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            # 对车辆的宽高进行判断
            # 以验证是否是有效的车辆
            is_valid = (w >= self.min_w) and (h >= self.min_h)
            if not is_valid:
                continue
            cars.append((x + w // 2, y + h // 2, w, h))
        return cars


class YOLOv8Model:
    def __init__(self):
        self.model = YOLO("weights/yolov8n.pt")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.origin_imgsz = None
        self.imgsz = 640

    def preprocess(self, im):
        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        im = im[np.newaxis, ...]
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        im = im.to(self.device)
        im = im.float()
        im /= 255
        return im

    def postprocess(self, result):
        ih, iw = self.origin_imgsz
        car_tensor = result.boxes.xywh
        car_tensor = car_tensor.cpu().numpy()
        cars = []
        for x, y, w, h in car_tensor:
            # 将x,y,w,h恢复到原图尺寸
            x, y, w, h = x / self.imgsz * iw, y / self.imgsz * ih, w / self.imgsz * iw, h / self.imgsz * ih
            x, y, w, h = int(x), int(y), int(w), int(h)
            cars.append((x, y, w, h))
        return cars

    def __call__(self, image):
        self.origin_imgsz = image.shape[:2]
        image = self.preprocess(image)
        with torch.no_grad():
            results = self.model.predict(image)
            cars = self.postprocess(results[0])
            return cars


class OpenCVModel:
    def __init__(self):
        # 加载onnx模型推理
        self.net = cv2.dnn.readNetFromONNX("weights/yolov8n.onnx")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.imgsz = 640

    def preprocess(self, im):
        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        im = im[np.newaxis, ...]
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype(float)
        im /= 255
        return im

    def nms(self, outputs, confidence_threshold=0.3, nms_threshold=0.4):
        # 1 84 8400
        outputs = np.transpose(outputs, (0, 2, 1))
        boxes = []
        confidences = []
        class_ids = []
        height, width = self.origin_imgsz
        for output in outputs:
            for detection in output:
                scores = detection[4:]  # 类别得分
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                center_x = int(detection[0] * width / self.imgsz)
                center_y = int(detection[1] * height / self.imgsz)
                w = int(detection[2] * width / self.imgsz)
                h = int(detection[3] * height / self.imgsz)
                # 计算左上角坐标
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # 应用 NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        cars = []
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            x = x + w // 2
            y = y + h // 2
            cars.append((x, y, w, h))
        return cars

    def __call__(self, frame):
        self.origin_imgsz = frame.shape[:2]
        # blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        blob = self.preprocess(frame)
        self.net.setInput(blob)
        outputs = self.net.forward()
        cars = self.nms(outputs)
        return cars


def main():
    line_high = 550  # 检测线的高度
    offset = 10  # 线的偏移
    car_num = 0  # 统计车的数量

    cap = cv2.VideoCapture('videos/video.mp4')

    # model = TraditionModel()
    model = OpenCVModel()

    while True:
        ret, frame = cap.read()
        if ret:
            cars = model(frame)
            # 画一条检测线
            cv2.line(frame, (10, line_high), (1200, line_high), (255, 255, 0), 3)
            for x, y, w, h in cars:
                if (w / frame.shape[1]) > 0.6:
                    continue
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 255), 2)
                if (y > line_high - offset) and (y < line_high + offset):
                    car_num += 1
            cv2.putText(frame, "Cars Count:" + str(car_num), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            cv2.imshow('video', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
