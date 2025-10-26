#pragma once

#include "c74_min.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>  // For SHGetFolderPath
#include <filesystem>
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#endif

#include "ext.h"  // For Max externals
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <condition_variable>

#ifdef _WIN32
// Forward declare Jitter's round (exported from jitlib.dll).
extern "C" double round(double);

// Prevent accidental use of std::round.
#undef round
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#endif

class BundleResourceLoader {
public:
    static std::string get_resource_path(const std::string& resource_name) {
#ifdef _WIN32
        HMODULE hModule = NULL;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&get_resource_path, &hModule)) {
            wchar_t path[MAX_PATH];
            if (GetModuleFileNameW(hModule, path, MAX_PATH)) {
                std::wstring wpath(path);
                std::string external_path(wpath.begin(), wpath.end());

                // external_path = ".../GrooveTransformer/externals/gt.baseEncoder.mxe64"
                size_t externals_pos = external_path.find("\\externals\\");
                if (externals_pos != std::string::npos) {
                    std::string package_path = external_path.substr(0, externals_pos);
                    std::string resources_path = package_path + "\\resources\\" + resource_name;
                    return resources_path;
                }
            } else {
                return "";
            }

        }
        return "";
#else
        // Get the address of this function to find which bundle we're in
        Dl_info info;
        if (dladdr((void*)get_resource_path, &info) != 0 && info.dli_fname) {
            std::string external_path(info.dli_fname);
            // Path should be something like: .../GrooveTransformer/externals/gt.baseEncoder.mxo/Contents/MacOS/gt.baseEncoder
            // We want: .../GrooveTransformer/resources/BaseVAE/beta_0.2/encoder.onnx

            // Find the package directory by going up from the external
            size_t externals_pos = external_path.find("/externals/");
            if (externals_pos != std::string::npos) {
                // Get the package root directory
                std::string package_path = external_path.substr(0, externals_pos);
                std::string resources_path = package_path + "/resources/" + resource_name;
                return resources_path;
            } else {
                return "";
            }
        }

        // Fallback: try the old bundle resource method for backward compatibility
        CFArrayRef bundles = CFBundleGetAllBundles();
        if (bundles) {
            CFIndex count = CFArrayGetCount(bundles);
            for (CFIndex i = 0; i < count; i++) {
                CFBundleRef bundle = (CFBundleRef)CFArrayGetValueAtIndex(bundles, i);
                if (bundle) {
                    // Check if this bundle has our resource
                    CFStringRef resource_cf = CFStringCreateWithCString(kCFAllocatorDefault,
                                                                      resource_name.c_str(),
                                                                      kCFStringEncodingUTF8);
                    if (resource_cf) {
                        CFURLRef resource_url = CFBundleCopyResourceURL(bundle, resource_cf, nullptr, nullptr);
                        CFRelease(resource_cf);

                        if (resource_url) {
                            char path_buffer[2048];
                            if (CFURLGetFileSystemRepresentation(resource_url, true,
                                                               (UInt8*)path_buffer,
                                                               sizeof(path_buffer))) {
                                CFRelease(resource_url);

                                // Verify the file exists before returning
                                if (file_exists(std::string(path_buffer))) {
                                    return std::string(path_buffer);
                                }
                            }
                            CFRelease(resource_url);
                        }
                    }
                }
            }
        }
#endif
        return "";
    }

    static std::string get_package_resources_path() {
#ifdef _WIN32
        HMODULE hModule = NULL;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&get_package_resources_path, &hModule)) {
            wchar_t path[MAX_PATH];
            if (GetModuleFileNameW(hModule, path, MAX_PATH)) {
                std::wstring wpath(path);
                std::string external_path(wpath.begin(), wpath.end());
                size_t externals_pos = external_path.find("\\externals\\");
                if (externals_pos != std::string::npos) {
                    std::string package_path = external_path.substr(0, externals_pos);
                    return package_path + "\\resources";
                }
            }
        }
        return "";
#else
        Dl_info info;
        if (dladdr((void*)get_resource_path, &info) != 0 && info.dli_fname) {
            std::string external_path(info.dli_fname);
            size_t externals_pos = external_path.find("/externals/");
            if (externals_pos != std::string::npos) {
                std::string package_path = external_path.substr(0, externals_pos);
                return package_path + "/resources";
            }
        }
#endif
    }

    static bool file_exists(const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    }
};