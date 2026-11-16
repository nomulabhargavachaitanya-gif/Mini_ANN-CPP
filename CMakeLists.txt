cmake_minimum_required(VERSION 3.14)
project(MiniANN VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(miniann_core
    src/NeuralNetwork.cpp
    src/Layer.cpp
    src/Neuron.cpp
    src/DataLoader.cpp
)
target_include_directories(miniann_core PUBLIC include)

add_executable(miniann examples/main.cpp)
target_link_libraries(miniann PRIVATE miniann_core)

add_executable(predict_miniann examples/predict_main.cpp)
target_link_libraries(predict_miniann PRIVATE miniann_core)

add_executable(xor_miniann examples/xor_main.cpp)
target_link_libraries(xor_miniann PRIVATE miniann_core)
