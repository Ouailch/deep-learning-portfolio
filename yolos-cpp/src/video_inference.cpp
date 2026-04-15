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
    const std::string videoPath = (argc > 1) ? argv[1] : "../data/test_video.mp4";
    const std::string modelPath = "../models/best_v8_1.onnx";
    
    // Initialize detector
    bool isGPU = false;
    YOLO8Detector detector(modelPath, labelsPath, isGPU);
    
    // Open video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
        return -1;
    }
    
    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video Properties:" << std::endl;
    std::cout << "  Resolution: " << frameWidth << "x" << frameHeight << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    
    // Setup video writer for output
    cv::VideoWriter writer("output_video.mp4",
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          fps,
                          cv::Size(frameWidth, frameHeight));
    
    if (!writer.isOpened())
    {
        std::cerr << "Warning: Could not open video writer" << std::endl;
    }
    
    // Process frames
    cv::Mat frame;
    int frameCount = 0;
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nProcessing video..." << std::endl;
    
    while (cap.read(frame))
    {
        if (frame.empty()) break;
        
        // Detect objects
        auto frameStart = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = detector.detect(frame);
        auto frameEnd = std::chrono::high_resolution_clock::now();
        
        auto frameDuration = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart);
        
        // Draw detections
        detector.drawBoundingBox(frame, results);
        
        // Add performance info to frame
        std::string perfInfo = "FPS: " + std::to_string(1000.0 / frameDuration.count());
        cv::putText(frame, perfInfo, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Write frame
        if (writer.isOpened())
        {
            writer.write(frame);
        }
        
        // Display frame
        cv::imshow("Video Detection", frame);
        
        frameCount++;
        if (frameCount % 30 == 0)
        {
            std::cout << "Processed " << frameCount << " frames..." << std::endl;
        }
        
        // Exit on ESC
        if (cv::waitKey(1) == 27) break;
    }
    
    auto totalEndTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(totalEndTime - totalStartTime);
    
    // Cleanup
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    std::cout << "\nVideo processing complete!" << std::endl;
    std::cout << "Total frames processed: " << frameCount << std::endl;
    std::cout << "Total time: " << totalDuration.count() << " seconds" << std::endl;
    std::cout << "Output saved to: output_video.mp4" << std::endl;
    
    return 0;
}
