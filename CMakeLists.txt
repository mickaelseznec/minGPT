# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(gpt)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
find_package(CUDA REQUIRED)


add_library(gpt SHARED gpt.cpp gpt.cu)
set_target_properties(gpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                     EXPORT_COMPILE_COMMANDS ON
                                     CMAKE_CXX_STANDARD 17
                                     PREFIX ""
                                     SUFFIX ".${Python3_SOABI}.so")
target_include_directories(gpt PRIVATE ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(gpt ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${CUDA_LIBRARIES} ${Python3_LIBRARIES})
