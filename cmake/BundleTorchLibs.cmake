function(bundle_torch_libs TARGET_NAME)
    set(LIBTORCH_LIB_DIR "${TORCH_CMAKE_PATH}/lib")
    file(GLOB TORCH_DYLIBS
            "${LIBTORCH_LIB_DIR}/libc10.dylib"
            "${LIBTORCH_LIB_DIR}/libtorch*.dylib"
    )

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DYLIBS}
            "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo/Contents/MacOS"
            COMMENT "📦 Bundled LibTorch .dylib files into ${TARGET_NAME}.mxo"
    )

    set(LIBOMP_PATH "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LIBOMP_PATH}"
            "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo/Contents/MacOS"
            COMMENT "📦 Bundled libomp.dylib"
    )

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND install_name_tool -delete_rpath "${LIBTORCH_LIB_DIR}"
            "$<TARGET_FILE:${TARGET_NAME}>"
            COMMAND install_name_tool -add_rpath "@loader_path"
            "$<TARGET_FILE:${TARGET_NAME}>"
            COMMAND install_name_tool -change "${LIBOMP_PATH}"
            "@loader_path/libomp.dylib"
            "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo/Contents/MacOS/libtorch_cpu.dylib"
            COMMENT "🔧 Fixed RPATH and libomp linkage for ${TARGET_NAME}"
    )

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND codesign --force --deep --sign - --timestamp=none
            "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>"
            COMMENT "🔐 Code signed ${TARGET_NAME}.mxo bundle"
    )

    set(MAX_PACKAGE_PATH_DEFAULT "$ENV{HOME}/Documents/Max 9/Packages/FRFT/externals")
    set(MAX_PACKAGE_PATH ${MAX_PACKAGE_PATH_DEFAULT} CACHE PATH "Path to Max externals folder")

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_PATH}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/${TARGET_NAME}.mxo"
            "${MAX_PACKAGE_PATH}/${TARGET_NAME}.mxo"
            COMMENT "✅ Copied ${TARGET_NAME}.mxo to Max Packages folder"
    )

    # Copy models to Resources folder (only once per build, not per target)
    set(MAX_PACKAGE_RESOURCES_PATH "$ENV{HOME}/Documents/Max 9/Packages/FRFT/Resources")
    if(EXISTS "${CMAKE_SOURCE_DIR}/models")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_PACKAGE_RESOURCES_PATH}"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/models"
                "${MAX_PACKAGE_RESOURCES_PATH}"
                COMMENT "📦 Copied models to ${MAX_PACKAGE_RESOURCES_PATH}"
        )
    endif()
endfunction()