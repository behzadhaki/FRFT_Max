cmake_minimum_required(VERSION 3.16)

project(FRFT VERSION 1.0.0 LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Build Settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Windows-specific settings
if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_definitions(-D_WIN32_WINNT=0x0601)  # Windows 7 minimum
    add_definitions(-DNOMINMAX)  # Prevent min/max macro conflicts
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

if(MSVC)
    add_definitions(-D_CRT_NONSTDC_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
endif()

# ------------------------------------------------------------------------------
# Platform Detection
# ------------------------------------------------------------------------------
if(WIN32)
    set(IS_WINDOWS TRUE)
    set(EXTERNAL_EXTENSION "mxe64")
elseif(APPLE)
    set(IS_MACOS TRUE)
    set(EXTERNAL_EXTENSION "mxo")
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------------------------
# Fetch CPM
# ------------------------------------------------------------------------------
include(cmake/CPM.cmake)

# ------------------------------------------------------------------------------
# Find FFTW3 - Bundled (Windows) or System (macOS)
# ------------------------------------------------------------------------------
message(STATUS "🔍 Looking for FFTW3...")

if(IS_WINDOWS)
    # Use bundled FFTW3 for Windows
    set(FFTW3_BUNDLED_ROOT "${CMAKE_SOURCE_DIR}/fftw3/fftw3_x64-windows-static")

    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h
            PATHS "${FFTW3_BUNDLED_ROOT}/include"
            NO_DEFAULT_PATH
    )

    find_library(FFTW3_LIBRARIES NAMES fftw3.lib fftw3-3.lib libfftw3-3.lib
            PATHS "${FFTW3_BUNDLED_ROOT}/lib"
            NO_DEFAULT_PATH
    )

    if(FFTW3_INCLUDE_DIRS AND FFTW3_LIBRARIES)
        set(FFTW3_FOUND TRUE)
        message(STATUS "✅ Using bundled FFTW3 (Windows)")
        message(STATUS "   Include: ${FFTW3_INCLUDE_DIRS}")
        message(STATUS "   Library: ${FFTW3_LIBRARIES}")
    else()
        message(FATAL_ERROR "❌ FFTW3 not found in bundled location!

Expected structure:
  ${FFTW3_BUNDLED_ROOT}/include/fftw3.h
  ${FFTW3_BUNDLED_ROOT}/lib/fftw3.lib

Please check that the vcpkg files are properly copied.")
    endif()

elseif(IS_MACOS)
    # Use system FFTW3 for macOS (Homebrew)
    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h
            HINTS
            /opt/homebrew/include
            /opt/homebrew/opt/fftw/include
            /usr/local/include
            /usr/local/opt/fftw/include
    )

    find_library(FFTW3_LIBRARIES NAMES libfftw3.a
            HINTS
            /opt/homebrew/lib
            /opt/homebrew/opt/fftw/lib
            /usr/local/lib
            /usr/local/opt/fftw/lib
    )

    if(FFTW3_INCLUDE_DIRS AND FFTW3_LIBRARIES)
        set(FFTW3_FOUND TRUE)
        message(STATUS "✅ Found FFTW3 (macOS Homebrew)")
        message(STATUS "   Include: ${FFTW3_INCLUDE_DIRS}")
        message(STATUS "   Library: ${FFTW3_LIBRARIES}")
    else()
        message(FATAL_ERROR "❌ FFTW3 not found!

Install FFTW with Homebrew:
    brew install fftw")
    endif()
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

set(MAX_SDK_INCLUDES "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes")

# ------------------------------------------------------------------------------
# Find Max SDK Libraries (Windows only)
# ------------------------------------------------------------------------------
if(IS_WINDOWS)
    # Locate Max SDK libraries
    find_library(MAXAPI_LIB NAMES MaxAPI
            PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes/x64"
            NO_DEFAULT_PATH
    )
    find_library(MAXAUDIO_LIB NAMES MaxAudio
            PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/msp-includes/x64"
            NO_DEFAULT_PATH
    )
    find_library(JITTER_LIB NAMES jitlib
            PATHS "${MAX_SDK_ROOT}/source/max-sdk-base/c74support/jit-includes/x64"
            NO_DEFAULT_PATH
    )

    if(NOT MAXAPI_LIB)
        message(FATAL_ERROR "❌ MaxAPI.lib not found!

Expected location:
  ${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes/x64/MaxAPI.lib

Copy it from your Max installation:
  C:\\Program Files\\Cycling '74\\Max 9\\lib\\max-includes\\x64\\MaxAPI.lib")
    endif()

    message(STATUS "✅ Found MaxAPI.lib: ${MAXAPI_LIB}")
    if(MAXAUDIO_LIB)
        message(STATUS "✅ Found MaxAudio.lib: ${MAXAUDIO_LIB}")
    endif()
    if(JITTER_LIB)
        message(STATUS "✅ Found jitlib.lib: ${JITTER_LIB}")
    endif()
endif()

# ------------------------------------------------------------------------------
# Platform-specific libraries
# ------------------------------------------------------------------------------
if(IS_WINDOWS)
    set(PLATFORM_LIBRARIES
            kernel32
            user32
            gdi32
            winspool
            shell32
            ole32
            oleaut32
            uuid
            comdlg32
            advapi32
            winmm
            ws2_32
    )
else()
    set(PLATFORM_LIBRARIES "")
endif()

# ------------------------------------------------------------------------------
# Common Includes
# ------------------------------------------------------------------------------
set(COMMON_INCLUDES
        ${MIN_API_ROOT}/include
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support/max-includes
        ${MAX_SDK_ROOT}/source/max-sdk-base/c74support/msp-includes
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
if(IS_WINDOWS)
    target_link_libraries(frft_engine PUBLIC ${FFTW3_LIBRARIES})
else()
    target_link_libraries(frft_engine PUBLIC ${FFTW3_LIBRARIES} m)
endif()

message(STATUS "🔗 FRFT Engine linked with static FFTW: ${FFTW3_LIBRARIES}")

# ------------------------------------------------------------------------------
# Helper function to copy external to Max Packages folder
# ------------------------------------------------------------------------------
function(deploy_to_max TARGET_NAME)
    if(IS_MACOS)
        # Code sign the bundle on macOS
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND codesign --force --deep --sign - --timestamp=none
                "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>"
                COMMENT "🔏 Code signed ${TARGET_NAME}.${EXTERNAL_EXTENSION}"
        )
    endif()

    # Set deployment path based on platform
    if(IS_WINDOWS)
        set(MAX_PACKAGE_PATH_DEFAULT "$ENV{USERPROFILE}/Documents/Max 9/Packages/FRFT/externals")
    else()
        set(MAX_PACKAGE_PATH_DEFAULT "$ENV{HOME}/Documents/Max 9/Packages/FRFT/externals")
    endif()

    set(MAX_PACKAGE_PATH ${MAX_PACKAGE_PATH_DEFAULT} CACHE PATH "Path to Max externals folder")

    if(IS_WINDOWS)
        # Windows: copy DLL file
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${TARGET_NAME}>"
                "${MAX_PACKAGE_PATH}/${TARGET_NAME}.${EXTERNAL_EXTENSION}"
                COMMENT "✅ Copied ${TARGET_NAME}.${EXTERNAL_EXTENSION} to Max Packages folder"
        )
    else()
        # macOS: copy bundle
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_BINARY_DIR}/${TARGET_NAME}.${EXTERNAL_EXTENSION}"
                "${MAX_PACKAGE_PATH}/${TARGET_NAME}.${EXTERNAL_EXTENSION}"
                COMMENT "✅ Copied ${TARGET_NAME}.${EXTERNAL_EXTENSION} to Max Packages folder"
        )
    endif()

    message(STATUS "📦 ${TARGET_NAME} will deploy to: ${MAX_PACKAGE_PATH}")
endfunction()

# ------------------------------------------------------------------------------
# FRFT Target (Max External)
# ------------------------------------------------------------------------------
set(FRFT frft)

set(FRFT_SOURCE_FILES
        ${SRC_DIR}/frft.cpp
)

# Add commonsyms.c only on macOS (Windows links against MaxAPI.lib)
if(IS_MACOS)
    list(APPEND FRFT_SOURCE_FILES ${MAX_SDK_INCLUDES}/common/commonsyms.c)
endif()

add_library(${FRFT} MODULE ${FRFT_SOURCE_FILES})
target_include_directories(${FRFT} PRIVATE ${COMMON_INCLUDES})
target_link_libraries(${FRFT} PRIVATE frft_engine)

if(IS_WINDOWS)
    # Windows-specific: Link Max SDK libraries
    target_link_libraries(${FRFT} PRIVATE
            ${PLATFORM_LIBRARIES}
            ${MAXAPI_LIB}
    )

    # Add MaxAudio if available (for MSP objects)
    if(MAXAUDIO_LIB)
        target_link_libraries(${FRFT} PRIVATE ${MAXAUDIO_LIB})
    endif()

    # Add Jitter if available (for Jitter objects)
    if(JITTER_LIB)
        target_link_libraries(${FRFT} PRIVATE ${JITTER_LIB})
    endif()

    # Windows linker options - handle potential symbol conflicts
    target_link_options(${FRFT} PRIVATE
            "/FORCE:MULTIPLE"     # Allow multiple symbol definitions
            "/IGNORE:4006"        # Ignore warnings about multiple definitions
    )

    set_target_properties(${FRFT} PROPERTIES
            OUTPUT_NAME ${FRFT}
            SUFFIX ".${EXTERNAL_EXTENSION}"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

elseif(IS_MACOS)
    # macOS-specific properties
    target_link_libraries(${FRFT} PRIVATE ${PLATFORM_LIBRARIES})

    set_target_properties(${FRFT} PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
            BUNDLE TRUE
            BUNDLE_EXTENSION ${EXTERNAL_EXTENSION}
            OUTPUT_NAME ${FRFT}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${FRFT}.${EXTERNAL_EXTENSION}/Contents/MacOS
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
    )

    configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
            ${CMAKE_BINARY_DIR}/${FRFT}.${EXTERNAL_EXTENSION}/Contents/Info.plist
            @ONLY
    )
endif()

deploy_to_max(${FRFT})

# ------------------------------------------------------------------------------
# Install External
# ------------------------------------------------------------------------------
if(IS_WINDOWS)
    install(FILES "$<TARGET_FILE:${FRFT}>"
            DESTINATION "${CMAKE_SOURCE_DIR}/externals"
            RENAME "${FRFT}.${EXTERNAL_EXTENSION}"
    )
else()
    install(DIRECTORY "${CMAKE_BINARY_DIR}/${FRFT}.${EXTERNAL_EXTENSION}"
            DESTINATION "${CMAKE_SOURCE_DIR}/externals"
    )
endif()



# ------------------------------------------------------------------------------
# FRFT Test Executable (Standalone Testing)
# ------------------------------------------------------------------------------
add_executable(frft_test_reconstruction ${SRC_DIR}/frft_test_reconstruction.cpp)
target_include_directories(frft_test_reconstruction PRIVATE
        ${SRC_DIR}
        ${FFTW3_INCLUDE_DIRS}
)
target_link_libraries(frft_test_reconstruction PRIVATE frft_engine)

if(IS_WINDOWS)
    # Windows-specific: no additional libraries needed for standalone test
else()
    # macOS/Linux: link math library
    target_link_libraries(frft_test_reconstruction PRIVATE m)
endif()

set_target_properties(frft_test_reconstruction PROPERTIES
        OUTPUT_NAME frft_test_reconstruction
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "✅ FRFT Test executable configured")

# Install test executable
install(TARGETS frft_test_reconstruction
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/bin"
)

# ------------------------------------------------------------------------------
# Optimization flags for Release builds
# ------------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(frft_engine PRIVATE /O2 /fp:fast)
        target_compile_options(${FRFT} PRIVATE /O2 /fp:fast)
        target_compile_options(frft_test_reconstruction PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(frft_engine PRIVATE -O3 -march=native -ffast-math)
        target_compile_options(${FRFT} PRIVATE -O3 -march=native -ffast-math)
        target_compile_options(frft_test_reconstruction PRIVATE -O3 -march=native -ffast-math)
    endif()
    message(STATUS "✅ Release optimizations enabled")
endif()

# ------------------------------------------------------------------------------
# Print Summary
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "══════════════════════════════════════════════════")
message(STATUS "  FRFT Build Configuration Summary")
message(STATUS "══════════════════════════════════════════════════")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
if(IS_MACOS)
    message(STATUS "  Architecture: ${CMAKE_OSX_ARCHITECTURES}")
endif()
message(STATUS "  External Type: .${EXTERNAL_EXTENSION}")
message(STATUS "  FFTW3 Include: ${FFTW3_INCLUDE_DIRS}")
message(STATUS "  FFTW3 Library: ${FFTW3_LIBRARIES} (STATIC)")
if(IS_WINDOWS)
    message(STATUS "  MaxAPI Library: ${MAXAPI_LIB}")
    if(MAXAUDIO_LIB)
        message(STATUS "  MaxAudio Library: ${MAXAUDIO_LIB}")
    endif()
    if(JITTER_LIB)
        message(STATUS "  Jitter Library: ${JITTER_LIB}")
    endif()
endif()
message(STATUS "  Linking: Static (no external dependencies)")
message(STATUS "  Min API: ${MIN_API_ROOT}")
message(STATUS "  Max SDK: ${MAX_SDK_ROOT}")
message(STATUS "  Source Dir: ${SRC_DIR}")
if(IS_WINDOWS)
    message(STATUS "  Max Deploy: %USERPROFILE%/Documents/Max 9/Packages/FRFT/externals")
else()
    message(STATUS "  Max Deploy: ~/Documents/Max 9/Packages/FRFT/externals")
endif()
message(STATUS "══════════════════════════════════════════════════")
message(STATUS "")