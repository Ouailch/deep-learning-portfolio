#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "YOLO8.hpp"

int main(int argc, char* argv[])
{
    // Configuration
    const std::string labelsPath = "../models/classes.names";
    const std::string modelPath = "../models/best_v8_1.onnx";
    
    int cameraId = 0;
    if (argc > 1)
    {
        try {
            cameraId = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid camera ID. Using default (0)" << std::endl;
        }
    }
    
    // Initialize detector
    bool isGPU = false;
    YOLO8Detector detector(modelPath, labelsPath, isGPU);
    
    // Open camera
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera " << cameraId << std::endl;
        return -1;
    }
    
    // Set camera resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    std::cout << "Camera initialized successfully" << std::endl;
    std::cout << "Press 'ESC' to exit" << std::endl;
    std::cout << "Press 'S' to save frame" << std::endl;
    
    cv::Mat frame;
    int frameCount = 0;
    float avgFPS = 0.0f;
    const int FPS_HISTORY = 30;
    
    while (true)
    {
        if (!cap.read(frame) || frame.empty())
        {
            std::cerr << "Error reading frame from camera" << std::endl;
            break;
        }
        
        // Measure detection time
        auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = detector.detect(frame);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        float currentFPS = 1000.0f / duration.count();
        
        // Calculate moving average FPS
        avgFPS = (avgFPS * (FPS_HISTORY - 1) + currentFPS) / FPS_HISTORY;
        
        // Draw detections
        detector.drawBoundingBox(frame, results);
        
        // Display performance metrics
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(avgFPS));
        std::string detectionText = "Detections: " + std::to_string(results.size());
        
        cv::putText(frame, fpsText, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, detectionText, cv::Point(10, 70),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Show frame
        cv::imshow("Real-time Detection", frame);
        
        int key = cv::waitKey(1);
        
        if (key == 27) // ESC
        {
            std::cout << "Exiting..." << std::endl;
            break;
        }
        else if (key == 's' || key == 'S')
        {
            std::string filename = "capture_" + std::to_string(frameCount) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Frame saved to: " << filename << std::endl;
        }
        
        frameCount++;
    }
    
    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "\nSession Statistics:" << std::endl;
    std::cout << "Total frames processed: " << frameCount << std::endl;
    std::cout << "Average FPS: " << avgFPS << std::endl;
    
    return 0;
}
