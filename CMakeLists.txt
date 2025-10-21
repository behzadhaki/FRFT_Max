cmake_minimum_required(VERSION 3.16)

project(FRFT VERSION 1.0.0 LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Build Settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)

# ------------------------------------------------------------------------------
# Fetch CPM
# ------------------------------------------------------------------------------
include(cmake/CPM.cmake)

# ------------------------------------------------------------------------------
# Fetch Min API
# ------------------------------------------------------------------------------
CPMAddPackage(
        NAME MinAPI
        GITHUB_REPOSITORY Cycling74/min-api
        GIT_TAG main
)
if (NOT MinAPI_ADDED)
    message(FATAL_ERROR "❌ Failed to fetch Min API.")
endif()
set(MIN_API_ROOT ${MinAPI_SOURCE_DIR})
message(STATUS "✅ Using Min API from: ${MIN_API_ROOT}")

# ------------------------------------------------------------------------------
# Set Max SDK Root (manual clone)
# ------------------------------------------------------------------------------
set(MAX_SDK_ROOT "${CMAKE_SOURCE_DIR}/lib/max-sdk")
message(STATUS "✅ Using Max SDK from: ${MAX_SDK_ROOT}")

# ------------------------------------------------------------------------------
# Set LibTorch Location
# ------------------------------------------------------------------------------
set(TORCH_DIR "/Library/Application Support/libtorch/libtorch-2.6.0-Release/libtorch")
file(TO_CMAKE_PATH "${TORCH_DIR}" TORCH_CMAKE_PATH)
find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PATH}" NO_DEFAULT_PATH)
message(STATUS "✅ Found LibTorch at: ${TORCH_CMAKE_PATH}")

# ------------------------------------------------------------------------------
# Include BundleTorchLibs helper
# ------------------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/BundleTorchLibs.cmake)

# ------------------------------------------------------------------------------
# Common Includes
# ------------------------------------------------------------------------------
set(COMMON_INCLUDES
        ${MIN_API_ROOT}/include
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes
        ${TORCH_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# Shared Source Directory
# ------------------------------------------------------------------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# ------------------------------------------------------------------------------
# Model Path
# ------------------------------------------------------------------------------
set(MODEL_SOURCE "${CMAKE_SOURCE_DIR}/models/frft_continuous.ts")
set(MODEL_FILENAME "frft_continuous.ts")

# ------------------------------------------------------------------------------
# FRFT Target
# ------------------------------------------------------------------------------
set(FRFT frft)
add_library(${FRFT} MODULE ${SRC_DIR}/frft.cpp)
target_include_directories(${FRFT} PRIVATE ${COMMON_INCLUDES})
target_link_libraries(${FRFT} "${TORCH_LIBRARIES}")

# Define the model filename for C++ code
target_compile_definitions(${FRFT} PRIVATE
        FRFT_MODEL_FILENAME="${MODEL_FILENAME}"
)

set_target_properties(${FRFT} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
        BUNDLE TRUE
        BUNDLE_EXTENSION mxo
        OUTPUT_NAME ${FRFT}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${FRFT}.mxo/Contents/MacOS
        INSTALL_RPATH "@loader_path"
)
configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
        ${CMAKE_BINARY_DIR}/${FRFT}.mxo/Contents/Info.plist
        @ONLY
)
bundle_torch_libs(${FRFT})

# Copy model file into .mxo/Contents/MacOS (same location as torch dylibs)
add_custom_command(TARGET ${FRFT}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MODEL_SOURCE}"
        "${CMAKE_BINARY_DIR}/${FRFT}.mxo/Contents/MacOS/${MODEL_FILENAME}"
        COMMENT "📦 Copied model file into ${FRFT}.mxo/Contents/MacOS"
)

# ------------------------------------------------------------------------------
# Install .mxo Bundle (Optional)
# ------------------------------------------------------------------------------
install(DIRECTORY "${CMAKE_BINARY_DIR}/${FRFT}.mxo"
        DESTINATION "${CMAKE_SOURCE_DIR}/externals"
)

# ------------------------------------------------------------------------------
# Group Target
# ------------------------------------------------------------------------------
add_custom_target(FRFTAll
        DEPENDS ${FRFT}
)

# ------------------------------------------------------------------------------
# Copy to Max Package Path (Optional)
# ------------------------------------------------------------------------------
set(MAX_PACKAGE_PATH "$ENV{HOME}/Documents/Max 9/Packages/FRFT/externals")

# Copy frft to Max Packages
add_custom_command(TARGET ${FRFT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/${FRFT}.mxo"
        "${MAX_PACKAGE_PATH}/${FRFT}.mxo"
        COMMENT "✅ Copied ${FRFT}.mxo to Max Packages folder"
)