#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

class TraditionModel {
private:
    int min_w;
    int min_h;
    Ptr<BackgroundSubtractorMOG2> motion;
    Mat kernel;

public:
    TraditionModel() {
        min_h = 90;
        min_w = 90;
        motion = createBackgroundSubtractorMOG2();
        kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    }

    vector<Rect> forward(Mat frame) {
        //灰度
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        //去噪
        GaussianBlur(frame, frame, Size(3, 3), 5);
        //去背景
        this->motion->apply(frame, frame);
        //腐蚀
        erode(frame, frame, this->kernel, Point(-1, -1), 1);
        //膨胀
        dilate(frame, frame, this->kernel, Point(-1, -1), 3);
        //闭操作
        morphologyEx(frame, frame, MORPH_CLOSE, this->kernel, Point(-1, -1), 2);
        //查找轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(frame, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        //过滤
        vector<Rect> cars;
        for (auto &c: contours) {
            Rect r = boundingRect(c);
            bool is_valid = (r.width > this->min_w) and (r.height > this->min_h);
            if (is_valid) {
                cars.push_back(r);
            }
        }
        return cars;
    }
};


class YOLOv8Model {
private:
    dnn::Net net;
    int imgsz;
    Size origin_imgsz;
public:
    YOLOv8Model() {
        this->net = dnn::readNetFromONNX("../assets/yolov8n.onnx");
//        this->net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
//        this->net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
        this->imgsz = 640;
    }

    vector<Rect> nms(Mat &outputs, float conf_thresh = 0.3, float iou_thresh = 0.4) {
        outputs = outputs.reshape(1, 84);
        transpose(outputs, outputs);
        vector<Rect> boxes;
        vector<float> confs;
        int width = this->origin_imgsz.width;
        int height = this->origin_imgsz.height;

        for (int i = 0; i < outputs.rows; i++) {
            Mat scores = outputs.row(i).colRange(4, outputs.cols);
            double conf;
            Point class_id;
            minMaxLoc(scores, 0, &conf, 0, &class_id);
            int cx = outputs.at<float>(i, 0) * width / this->imgsz;
            int cy = outputs.at<float>(i, 1) * height / this->imgsz;
            int w = outputs.at<float>(i, 2) * width / this->imgsz;
            int h = outputs.at<float>(i, 3) * height / this->imgsz;
            // 计算左上角坐标
            int x = cx - w/2;
            int y = cy - h/2;
            boxes.emplace_back(Point(x, y), Size(w, h));
            confs.emplace_back(conf);
        }

        vector<int> indices;
        dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);
        vector<Rect> cars;
        for (int i: indices){
            cars.push_back(boxes[i]);
        }
        return cars;
    }

    vector<Rect> forward(Mat &frame) {
        this->origin_imgsz = frame.size();
        Mat blob = dnn::blobFromImage(frame, 1.0 / 255, Size(this->imgsz, this->imgsz), Scalar(0, 0, 0), true, false);
        this->net.setInput(blob);
        Mat outputs = this->net.forward();
        vector<Rect> cars = this->nms(outputs);
        return cars;
    }
};

int main(int argc, char const *argv[]) {
    VideoCapture cap("../assets/videos/video.mp4");
//    TraditionModel model;
    YOLOv8Model model;
    int line_high = 550;
    int offset = 10;
    int car_num = 0;
    while (true) {
        Mat frame;
        bool ret = cap.read(frame);
        if (ret) {
            vector<Rect> cars = model.forward(frame);
            // 绘制检测线
            line(frame, Point(10, line_high), Point(1200, line_high), Scalar(255, 255, 0), 3);
            for (auto &rect: cars) {
                if (rect.width > 0.6 * frame.cols) {
                    continue;
                }
                rectangle(frame, rect, Scalar(0, 0, 255), 2);
                if (rect.y > line_high - offset and rect.y < line_high + offset) {
                    car_num++;
                }
            }
            putText(frame, "Cars Count: " + to_string(car_num), Point(500, 60), FONT_HERSHEY_SIMPLEX, 2,
                    Scalar(255, 0, 0), 5);
            imshow("video", frame);
        }
        int key = waitKey(1);
        if (key == 27) {
            break;
        }
    }
    cap.release();
    destroyAllWindows();

    return 0;
}


