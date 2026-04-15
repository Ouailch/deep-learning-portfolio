#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include "YOLO8.hpp"

int main(int argc, char* argv[])
{
    // Configuration paths
    const std::string labelsPath = "../models/classes.names";
    const std::string imagePath = (argc > 1) ? argv[1] : "../data/test_image.jpg";
    const std::string modelPath = "../models/best_v8_1.onnx";
    
    // Initialize detector
    bool isGPU = false;
    YOLO8Detector detector(modelPath, labelsPath, isGPU);
    
    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image at: " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "Processing image: " << imagePath << std::endl;
    
    // Perform object detection and measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<Detection> results = detector.detect(image);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
    std::cout << "Objects detected: " << results.size() << std::endl;
    
    // Draw bounding boxes
    detector.drawBoundingBox(image, results);
    
    // Display results
    cv::imshow("Detections", image);
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Optional: Save output image
    std::string outputPath = "output.jpg";
    cv::imwrite(outputPath, image);
    std::cout << "Output image saved to: " << outputPath << std::endl;
    
    return 0;
}
