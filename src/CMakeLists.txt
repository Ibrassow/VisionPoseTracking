

add_library(tracking SHARED
    camera.cpp
    frame.cpp
    extractor.cpp
)

target_link_libraries(tracking
    Eigen3::Eigen 
    ${OpenCV_LIBS}
)
