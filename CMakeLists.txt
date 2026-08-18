cmake_minimum_required(VERSION 3.0.0)
project(code)
set(CMAKE_CXX_STANDARD 17)

# 设置默认构建类型为 Debug（如果没有指定）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 设置编译器特定的调试选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Werror")

# 根据构建类型设置不同的编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug 模式额外选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
    # 启用更多调试信息
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    message(STATUS "Debug mode enabled with AddressSanitizer")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "Release mode enabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # 带调试信息的发布版本
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -DNDEBUG")
    message(STATUS "Release with Debug Info mode enabled")
endif()

# 显示当前构建类型和编译选项
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
enable_testing()
add_executable(code
main.cpp
)