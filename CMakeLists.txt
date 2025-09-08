cmake_minimum_required(VERSION 3.31.0)
project(llm_project VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED True)

link_directories(/usr/local/lib)

include_directories(
    "include"
    "lib"
)

add_executable(llm_project 
    src/main.cpp 
    src/llm_interface.cpp
    src/embed_interface.cpp
    src/rag_client.cpp
)

target_link_libraries(llm_project 
    PRIVATE 
    llama
    ggml
    ggml-base
)

include(CTest)
enable_testing()

