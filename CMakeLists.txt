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
# Find FFTW3 - FORCE STATIC LINKING
# ------------------------------------------------------------------------------
message(STATUS "🔍 Looking for FFTW3 (static)...")

# Look for static library first
find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h
        HINTS
        /opt/homebrew/include
        /opt/homebrew/opt/fftw/include
        /usr/local/include
        /usr/local/opt/fftw/include
)

# Force finding the .a static library
find_library(FFTW3_STATIC_LIB NAMES libfftw3.a
        HINTS
        /opt/homebrew/lib
        /opt/homebrew/opt/fftw/lib
        /usr/local/lib
        /usr/local/opt/fftw/lib
)

if(FFTW3_INCLUDE_DIRS AND FFTW3_STATIC_LIB)
    set(FFTW3_LIBRARIES ${FFTW3_STATIC_LIB})
    set(FFTW3_FOUND TRUE)
    message(STATUS "✅ Found FFTW3 static library")
    message(STATUS "   Include: ${FFTW3_INCLUDE_DIRS}")
    message(STATUS "   Library: ${FFTW3_LIBRARIES}")
else()
    message(FATAL_ERROR "❌ FFTW3 static library not found!

Install FFTW with Homebrew (includes static library):
    brew install fftw

The static library should be at:
    /opt/homebrew/opt/fftw/lib/libfftw3.a

If you only have .dylib, reinstall FFTW.")
endif()

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
# Set Max SDK Root
# ------------------------------------------------------------------------------
set(MAX_SDK_ROOT "${CMAKE_SOURCE_DIR}/lib/max-sdk")
if(NOT EXISTS ${MAX_SDK_ROOT})
    message(FATAL_ERROR "❌ Max SDK not found at: ${MAX_SDK_ROOT}")
endif()
message(STATUS "✅ Using Max SDK from: ${MAX_SDK_ROOT}")

# ------------------------------------------------------------------------------
# Common Includes
# ------------------------------------------------------------------------------
set(COMMON_INCLUDES
        ${MIN_API_ROOT}/include
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes
        ${FFTW3_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# Shared Source Directory
# ------------------------------------------------------------------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# ------------------------------------------------------------------------------
# FRFT Engine Library
# ------------------------------------------------------------------------------
add_library(frft_engine STATIC
        ${SRC_DIR}/frft_engine.cpp
        ${SRC_DIR}/frft_engine.h
)
target_include_directories(frft_engine PUBLIC
        ${SRC_DIR}
        ${FFTW3_INCLUDE_DIRS}
)

# Link with static FFTW library
target_link_libraries(frft_engine PUBLIC ${FFTW3_LIBRARIES} m)

message(STATUS "🔗 FRFT Engine linked with static FFTW: ${FFTW3_LIBRARIES}")

# ------------------------------------------------------------------------------
# Helper function to copy .mxo to Max Packages folder
# ------------------------------------------------------------------------------
function(deploy_to_max TARGET_NAME)
    # Code sign the bundle
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND codesign --force --deep --sign - --timestamp=none
            "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>"
            COMMENT "🔐 Code signed ${TARGET_NAME}.mxo bundle"
    )

    # Copy to Max Packages folder
    set(MAX_PACKAGE_PATH_DEFAULT "$ENV{HOME}/Documents/Max 9/Packages/FRFT/externals")
    set(MAX_PACKAGE_PATH ${MAX_PACKAGE_PATH_DEFAULT} CACHE PATH "Path to Max externals folder")

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo"
            "${MAX_PACKAGE_PATH}/${TARGET_NAME}.mxo"
            COMMENT "✅ Copied ${TARGET_NAME}.mxo to Max Packages folder"
    )

    message(STATUS "📦 ${TARGET_NAME} will deploy to: ${MAX_PACKAGE_PATH}")
endfunction()

# ------------------------------------------------------------------------------
# FRFT Target (Max External)
# ------------------------------------------------------------------------------
set(FRFT frft)
add_library(${FRFT} MODULE ${SRC_DIR}/frft.cpp)
target_include_directories(${FRFT} PRIVATE ${COMMON_INCLUDES})
target_link_libraries(${FRFT} frft_engine)

set_target_properties(${FRFT} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
        BUNDLE TRUE
        BUNDLE_EXTENSION mxo
        OUTPUT_NAME ${FRFT}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${FRFT}.mxo/Contents/MacOS
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
)

configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
        ${CMAKE_BINARY_DIR}/${FRFT}.mxo/Contents/Info.plist
        @ONLY
)

# Deploy to Max (no bundling needed with static linking!)
deploy_to_max(${FRFT})

# ------------------------------------------------------------------------------
# Install .mxo Bundle
# ------------------------------------------------------------------------------
install(DIRECTORY "${CMAKE_BINARY_DIR}/${FRFT}.mxo"
        DESTINATION "${CMAKE_SOURCE_DIR}/externals"
)

# ------------------------------------------------------------------------------
# Signal Inspector Target (Optional)
# ------------------------------------------------------------------------------
if(EXISTS ${SRC_DIR}/signalinspector.cpp)
    set(SignalInspector signalinspector)
    add_library(${SignalInspector} MODULE ${SRC_DIR}/signalinspector.cpp)
    target_include_directories(${SignalInspector} PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(${SignalInspector} frft_engine)

    set_target_properties(${SignalInspector} PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
            BUNDLE TRUE
            BUNDLE_EXTENSION mxo
            OUTPUT_NAME ${SignalInspector}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${SignalInspector}.mxo/Contents/MacOS
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
    )

    configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
            ${CMAKE_BINARY_DIR}/${SignalInspector}.mxo/Contents/Info.plist
            @ONLY
    )

    # Deploy to Max
    deploy_to_max(${SignalInspector})

    install(DIRECTORY "${CMAKE_BINARY_DIR}/${SignalInspector}.mxo"
            DESTINATION "${CMAKE_SOURCE_DIR}/externals"
    )

    add_custom_target(FRFTAll DEPENDS ${FRFT} ${SignalInspector})
else()
    message(STATUS "ℹ️  signalinspector.cpp not found, skipping")
    add_custom_target(FRFTAll DEPENDS ${FRFT})
endif()

# ------------------------------------------------------------------------------
# Optimization flags for Release builds
# ------------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(frft_engine PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(${FRFT} PRIVATE -O3 -march=native -ffast-math)
    if(TARGET ${SignalInspector})
        target_compile_options(${SignalInspector} PRIVATE -O3 -march=native -ffast-math)
    endif()
    message(STATUS "✅ Release optimizations enabled")
endif()

# ------------------------------------------------------------------------------
# Print Summary
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════")
message(STATUS "  FRFT Build Configuration Summary")
message(STATUS "═══════════════════════════════════════════════════")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Architecture: ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "  FFTW3 Include: ${FFTW3_INCLUDE_DIRS}")
message(STATUS "  FFTW3 Library: ${FFTW3_LIBRARIES} (STATIC)")
message(STATUS "  Linking: Static (no dylib bundling needed)")
message(STATUS "  Bundle size: ~2MB (everything embedded)")
message(STATUS "  Min API: ${MIN_API_ROOT}")
message(STATUS "  Max SDK: ${MAX_SDK_ROOT}")
message(STATUS "  Source Dir: ${SRC_DIR}")
message(STATUS "  Max Deploy: ~/Documents/Max 9/Packages/FRFT/externals")
message(STATUS "═══════════════════════════════════════════════════")
message(STATUS "")