cmake_minimum_required(VERSION 3.15)
project(yolo-cpp-detector VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime directory
set(ONNXRUNTIME_DIR "onnxruntime-linux-x64-1.20.0" CACHE PATH "ONNX Runtime directory")

# Find required packages
find_package(OpenCV 4.5 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_DIR}/include)

# Source files
set(SOURCES
    src/image_inference.cpp
    src/video_inference.cpp
    src/camera_inference.cpp
)

# Create executables
add_executable(image_inference src/image_inference.cpp)
add_executable(video_inference src/video_inference.cpp)
add_executable(camera_inference src/camera_inference.cpp)

# Link libraries
foreach(TARGET image_inference video_inference camera_inference)
    target_link_libraries(${TARGET} PUBLIC
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
    )
    
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_DIR}/include
    )
endforeach()

# Platform-specific settings
if(UNIX AND NOT APPLE)
    message(STATUS "Building for Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
elseif(APPLE)
    message(STATUS "Building for macOS")
    foreach(TARGET image_inference video_inference camera_inference)
        target_link_libraries(${TARGET} PUBLIC
            ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib
        )
    endforeach()
elseif(WIN32)
    message(STATUS "Building for Windows")
    foreach(TARGET image_inference video_inference camera_inference)
        target_link_libraries(${TARGET} PUBLIC
            ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
        )
    endforeach()
endif()

# Output directories
set_target_properties(image_inference video_inference camera_inference
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")
