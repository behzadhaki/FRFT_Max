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
# Find FFTW3 - Use full library paths
# ------------------------------------------------------------------------------
message(STATUS "🔍 Looking for FFTW3...")

find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h
        HINTS
        /opt/homebrew/include
        /opt/homebrew/opt/fftw/include
        /usr/local/include
        /usr/local/opt/fftw/include
)

find_library(FFTW3_LIBRARIES NAMES fftw3 libfftw3
        HINTS
        /opt/homebrew/lib
        /opt/homebrew/opt/fftw/lib
        /usr/local/lib
        /usr/local/opt/fftw/lib
)

if(FFTW3_INCLUDE_DIRS AND FFTW3_LIBRARIES)
    set(FFTW3_FOUND TRUE)
    message(STATUS "✅ Found FFTW3")
    message(STATUS "   Include: ${FFTW3_INCLUDE_DIRS}")
    message(STATUS "   Library: ${FFTW3_LIBRARIES}")
else()
    message(FATAL_ERROR "❌ FFTW3 not found! Install with: brew install fftw")
endif()

# Determine if we're linking statically or dynamically
get_filename_component(FFTW3_LIB_EXT ${FFTW3_LIBRARIES} EXT)
if(FFTW3_LIB_EXT STREQUAL ".a")
    set(FFTW3_STATIC TRUE)
    message(STATUS "   Using static FFTW3 (no bundling needed)")
elseif(FFTW3_LIB_EXT STREQUAL ".dylib")
    set(FFTW3_STATIC FALSE)
    message(STATUS "   Using dynamic FFTW3 (will bundle into .mxo)")
else()
    message(WARNING "   Unknown FFTW3 library type: ${FFTW3_LIB_EXT}")
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
target_link_libraries(frft_engine PUBLIC ${FFTW3_LIBRARIES} m)

# ------------------------------------------------------------------------------
# Helper function to bundle FFTW dylib into .mxo (similar to BundleTorchLibs)
# ------------------------------------------------------------------------------
function(bundle_fftw_lib TARGET_NAME)
    if(NOT FFTW3_STATIC)
        # Get just the filename of the dylib
        get_filename_component(FFTW3_DYLIB_NAME ${FFTW3_LIBRARIES} NAME)

        # Copy FFTW dylib into the bundle
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW3_LIBRARIES}"
                "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo/Contents/MacOS"
                COMMENT "📦 Bundled FFTW3 library into ${TARGET_NAME}.mxo"
        )

        # Make the copied dylib writable (Homebrew libs are read-only)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND chmod u+w "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo/Contents/MacOS/${FFTW3_DYLIB_NAME}"
                COMMENT "🔓 Made bundled FFTW3 writable"
        )

        # Update install_name to use @loader_path
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND install_name_tool -change
                "${FFTW3_LIBRARIES}"
                "@loader_path/${FFTW3_DYLIB_NAME}"
                "$<TARGET_FILE:${TARGET_NAME}>"
                COMMENT "🔧 Updated ${TARGET_NAME} to use bundled FFTW3"
        )

        # Update the id of the copied dylib
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND install_name_tool -id
                "@loader_path/${FFTW3_DYLIB_NAME}"
                "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo/Contents/MacOS/${FFTW3_DYLIB_NAME}"
                COMMENT "🔧 Updated bundled FFTW3 library id"
        )

        # Code sign the bundle
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND codesign --force --deep --sign - --timestamp=none
                "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>"
                COMMENT "🔐 Code signed ${TARGET_NAME}.mxo bundle"
        )

        message(STATUS "📦 ${TARGET_NAME} will bundle: ${FFTW3_DYLIB_NAME}")
    endif()

    # Copy to Max Packages folder (just like BundleTorchLibs.cmake)
    set(MAX_PACKAGE_PATH_DEFAULT "$ENV{HOME}/Documents/Max 9/Packages/FRFT/externals")
    set(MAX_PACKAGE_PATH ${MAX_PACKAGE_PATH_DEFAULT} CACHE PATH "Path to Max externals folder")

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo"
            "${MAX_PACKAGE_PATH}/${TARGET_NAME}.mxo"
            COMMENT "✅ Copied ${TARGET_NAME}.mxo to Max Packages folder"
    )
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

# Bundle FFTW library if needed
bundle_fftw_lib(${FRFT})

# ------------------------------------------------------------------------------
# Install .mxo Bundle (also available via 'make install')
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

    # Bundle FFTW library if needed
    bundle_fftw_lib(${SignalInspector})

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
message(STATUS "  FFTW3 Library: ${FFTW3_LIBRARIES}")
if(FFTW3_STATIC)
    message(STATUS "  FFTW3 Linking: Static (no bundling needed)")
else()
    message(STATUS "  FFTW3 Linking: Dynamic (will bundle into .mxo)")
endif()
message(STATUS "  Min API: ${MIN_API_ROOT}")
message(STATUS "  Max SDK: ${MAX_SDK_ROOT}")
message(STATUS "  Source Dir: ${SRC_DIR}")
message(STATUS "  Externals Dir: ${CMAKE_SOURCE_DIR}/externals")
message(STATUS "═══════════════════════════════════════════════════")
message(STATUS "")