#ifndef YOLO8_HPP
#define YOLO8_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// Detection structure
struct Detection {
    float x, y, w, h;
    float confidence;
    int classId;
    std::string className;
};

class YOLO8Detector {
private:
    Ort::Session* session;
    Ort::Env env;
    std::vector<std::string> classNames;
    const std::vector<int> input_shape_ = {1, 3, 640, 640};
    const float conf_threshold = 0.45f;
    const float iou_threshold = 0.5f;
    
    void loadClassNames(const std::string& labelsPath);
    std::vector<Detection> postprocess(
        const std::vector<float>& outputs,
        const cv::Mat& original,
        const cv::Mat& resized);
    cv::Mat letterbox(const cv::Mat& source, int target_width, int target_height);
    
public:
    YOLO8Detector(const std::string& modelPath, 
                  const std::string& labelsPath,
                  bool useGPU = false);
    ~YOLO8Detector();
    
    std::vector<Detection> detect(const cv::Mat& image);
    void drawBoundingBox(cv::Mat& image, const std::vector<Detection>& detections);
};

#endif // YOLO8_HPP
