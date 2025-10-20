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
# Encoder Target
# ------------------------------------------------------------------------------
set(FRFTIN frftin)
add_library(${FRFTIN} MODULE ${SRC_DIR}/frftin.cpp)
target_include_directories(${FRFTIN} PRIVATE ${COMMON_INCLUDES})
target_link_libraries(${FRFTIN} "${TORCH_LIBRARIES}")
set_target_properties(${FRFTIN} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
        BUNDLE TRUE
        BUNDLE_EXTENSION mxo
        OUTPUT_NAME ${FRFTIN}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${FRFTIN}.mxo/Contents/MacOS
        INSTALL_RPATH "@loader_path"
)
configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
        ${CMAKE_BINARY_DIR}/${FRFTIN}.mxo/Contents/Info.plist
        @ONLY
)
bundle_torch_libs(${FRFTIN})

# Rename Encoder to frftin~.mxo
add_custom_command(TARGET ${FRFTIN}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/${FRFTIN}.mxo"
        "${CMAKE_BINARY_DIR}/${FRFTIN}~.mxo"
        COMMAND ${CMAKE_COMMAND} -E remove_directory
        "${CMAKE_BINARY_DIR}/${FRFTIN}.mxo"
        COMMENT "Renaming ${FRFTIN}.mxo → ${FRFTIN}~.mxo"
)

# ------------------------------------------------------------------------------
# Decoder Target
# ------------------------------------------------------------------------------
set(FRFTOUT frftout)
add_library(${FRFTOUT} MODULE ${SRC_DIR}/frftout.cpp)
target_include_directories(${FRFTOUT} PRIVATE ${COMMON_INCLUDES})
target_link_libraries(${FRFTOUT} "${TORCH_LIBRARIES}")
set_target_properties(${FRFTOUT} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
        BUNDLE TRUE
        BUNDLE_EXTENSION mxo
        OUTPUT_NAME ${FRFTOUT}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${FRFTOUT}.mxo/Contents/MacOS
        INSTALL_RPATH "@loader_path"
)
configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
        ${CMAKE_BINARY_DIR}/${FRFTOUT}.mxo/Contents/Info.plist
        @ONLY
)
bundle_torch_libs(${FRFTOUT})

# Rename Decoder to frftout~.mxo
add_custom_command(TARGET ${FRFTOUT}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/${FRFTOUT}.mxo"
        "${CMAKE_BINARY_DIR}/${FRFTOUT}~.mxo"
        COMMAND ${CMAKE_COMMAND} -E remove_directory
        "${CMAKE_BINARY_DIR}/${FRFTOUT}.mxo"
        COMMENT "Renaming ${FRFTOUT}.mxo → ${FRFTOUT}~.mxo"
)

# ------------------------------------------------------------------------------
# Install Renamed .mxo Bundles (Optional)
# ------------------------------------------------------------------------------
install(DIRECTORY "${CMAKE_BINARY_DIR}/${FRFTIN}~.mxo"
        DESTINATION "${CMAKE_SOURCE_DIR}/externals"
)
install(DIRECTORY "${CMAKE_BINARY_DIR}/${FRFTOUT}~.mxo"
        DESTINATION "${CMAKE_SOURCE_DIR}/externals"
)

# ------------------------------------------------------------------------------
# Group Target
# ------------------------------------------------------------------------------
add_custom_target(FRFTAll
        DEPENDS ${FRFTIN} ${FRFTOUT}
)

# ------------------------------------------------------------------------------
# Copy to Max Package Path (Optional)
# ------------------------------------------------------------------------------
set(MAX_PACKAGE_PATH "$ENV{HOME}/Documents/Max 9/Packages/FRFT/externals")

# Copy frftin~ to Max Packages
add_custom_command(TARGET ${FRFTIN} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/${FRFTIN}~.mxo"
        "${MAX_PACKAGE_PATH}/${FRFTIN}~.mxo"
        COMMENT "✅ Copied ${FRFTIN}~.mxo to Max Packages folder"
)

# Copy frftout~ to Max Packages
add_custom_command(TARGET ${FRFTOUT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/${FRFTOUT}~.mxo"
        "${MAX_PACKAGE_PATH}/${FRFTOUT}~.mxo"
        COMMENT "✅ Copied ${FRFTOUT}~.mxo to Max Packages folder"
)


