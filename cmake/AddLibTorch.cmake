function(setup_libtorch_and_scripts TARGET_NAME)
    set(TORCH_VERSION 2.6.0)
    set(APP_SUPPORT_PATH "/Library/Application Support")
    set(TORCH_ROOT_PATH "${APP_SUPPORT_PATH}/libtorch/libtorch-${TORCH_VERSION}-${CMAKE_BUILD_TYPE}")

    message(STATUS "TORCH_ROOT_PATH: ${TORCH_ROOT_PATH}")

    if (NOT EXISTS "${TORCH_ROOT_PATH}")
        message(FATAL_ERROR "❌ Torch not found at ${TORCH_ROOT_PATH}. Please run install_libtorch.sh first.")
    endif()

    target_include_directories(${TARGET_NAME}
            PRIVATE
            "${TORCH_ROOT_PATH}/libtorch/include"
            "${TORCH_ROOT_PATH}/libtorch/include/torch/csrc/api/include"
    )

    file(GLOB TORCH_LIBS "${TORCH_ROOT_PATH}/libtorch/lib/*.dylib")
    target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_LIBS})

    # Copy .dylibs into the mxo bundle
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_ROOT_PATH}/libtorch/lib"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMENT "✅ Copied LibTorch dylibs into .mxo"
    )

    # Define TorchScripts location
    set(SOURCE_SCRIPT_DIR "${CMAKE_SOURCE_DIR}/TorchScripts")
    set(BUNDLE_RESOURCE_DIR "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/Contents/Resources")

    # Make base folder
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_RESOURCE_DIR}/TorchScripts"
    )

    # Conditionally copy TorchScripts/Models
    if(EXISTS "${SOURCE_SCRIPT_DIR}/Models")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E remove_directory "${BUNDLE_RESOURCE_DIR}/TorchScripts/Models"
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${SOURCE_SCRIPT_DIR}/Models" "${BUNDLE_RESOURCE_DIR}/TorchScripts/Models"
                COMMENT "📦 Copied Models to .mxo Resources"
        )
    endif()

    # Conditionally copy TorchScripts/Processing_Scripts
    if(EXISTS "${SOURCE_SCRIPT_DIR}/Processing_Scripts")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E remove_directory "${BUNDLE_RESOURCE_DIR}/TorchScripts/Processing_Scripts"
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${SOURCE_SCRIPT_DIR}/Processing_Scripts" "${BUNDLE_RESOURCE_DIR}/TorchScripts/Processing_Scripts"
                COMMENT "📦 Copied Processing_Scripts to .mxo Resources"
        )
    endif()

    # Define preprocessor macros for use in C++
    add_definitions(-DDEFAULT_MODEL_DIR="${BUNDLE_RESOURCE_DIR}/TorchScripts/Models")
    add_definitions(-DDEFAULT_PROCESSING_SCRIPTS_DIR="${BUNDLE_RESOURCE_DIR}/TorchScripts/Processing_Scripts")

    set_target_properties(${TARGET_NAME} PROPERTIES
            INSTALL_RPATH "@loader_path"
    )
endfunction()

