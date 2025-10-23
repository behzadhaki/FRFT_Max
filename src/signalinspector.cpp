// SignalInspector.cpp - Efficient STFT visualizer with smart decimation

#ifdef _WIN32
#include <windows.h>
#endif

#include "c74_min.h"
#include "c74_min_graphics.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>

using namespace c74::min;
using namespace c74::min::ui;

class SignalInspector : public object<SignalInspector>, public ui_operator<400, 300>, public vector_operator<> {
public:
    MIN_DESCRIPTION { "Efficient STFT SignalInspector visualizer with smart temporal decimation" };
    MIN_TAGS        { "ui, graphics, SignalInspector, visualization, msp, fft" };
    MIN_AUTHOR      { "Custom" };

    inlet<>  in { this, "(signal) Input STFT vector" };

    SignalInspector(const atoms& args = {})
        : object<SignalInspector>{}
        , ui_operator<400, 300>{this, args}
        , vector_operator<>{}
        , SignalInspectorSurface(nullptr)
        , surfaceWidth(0)
        , surfaceHeight(0)
        , currentFrame(0)
        , writeFrame(0)
        , numBins(0)
        , displayBins(0)
        , needsFullRedraw(true)
        , lastRenderFrame(0)
        , lastUpdateTime(std::chrono::steady_clock::now())
    {
        // Initialize with default frame count
        SignalInspectorData.resize(200);
    }

    ~SignalInspector() {
        updateTimer.stop();
        if (SignalInspectorSurface) {
            jgraphics_surface_destroy(SignalInspectorSurface);
            SignalInspectorSurface = nullptr;
        }
    }

    // Attributes
    attribute<int> frames {
        this, "frames", 200,
        range{10, 2000},
        description{"Number of time frames to display"}
    };

    attribute<number> updateratehz {
        this, "updateratehz", 20.0,
        range{1.0, 60.0},
        description{"Display update rate in Hz"}
    };

    attribute<bool> showhalfspectrum {
        this, "showhalfspectrum", true,
        description{"Show only lower half of spectrum (DC to Nyquist)"}
    };

    attribute<number> minValue { this, "minval", 0.0, description{"Minimum value (linear)"} };
    attribute<number> maxValue { this, "maxval", 100.0, description{"Maximum value (linear)"} };

    // Color scheme attributes
    attribute<color> lowColor { this, "lowcolor", color{0.0, 0.0, 0.2, 1.0} };
    attribute<color> midColor { this, "midcolor", color{1.0, 0.5, 0.0, 1.0} };
    attribute<color> highColor { this, "highcolor", color{1.0, 1.0, 0.0, 1.0} };

    // Clear the SignalInspector
    message<> clear { this, "clear", MIN_FUNCTION {
        writeFrame.store(0);
        currentFrame.store(0);
        lastRenderFrame = 0;
        numBins = 0;
        displayBins = 0;
        SignalInspectorData.clear();
        accumulatedFrame.clear();
        int maxFrames = static_cast<int>(frames);
        SignalInspectorData.resize(maxFrames);
        needsFullRedraw = true;
        return {};
    }};

    // Vector operator perform method - accumulates STFT frames
    void operator()(audio_bundle input, audio_bundle output) {
        // Get the input vector size
        auto in_vec = input.samples(0);
        int vecsize = static_cast<int>(input.frame_count());

        // Store the most recent frame (overwriting previous accumulated frame)
        // This ensures we always have the latest data when it's time to commit
        accumulatedFrame.clear();
        accumulatedFrame.reserve(vecsize);
        for (int i = 0; i < vecsize; ++i) {
            accumulatedFrame.push_back(static_cast<double>(in_vec[i]));
        }

        // Update bin count if changed
        if (numBins != vecsize) {
            numBins = vecsize;
            displayBins = static_cast<bool>(showhalfspectrum) ? (vecsize / 2) : vecsize;
            needsFullRedraw = true;
        }
    }

private:
    // Data storage
    std::vector<std::vector<double>> SignalInspectorData;
    std::vector<double> accumulatedFrame;  // Most recent STFT frame
    c74::max::t_jsurface* SignalInspectorSurface;
    int surfaceWidth, surfaceHeight;
    std::atomic<int> currentFrame;  // For rendering
    std::atomic<int> writeFrame;    // For writing new data
    int numBins;        // Total bins from input
    int displayBins;    // Bins to actually display (half if showhalfspectrum)
    bool needsFullRedraw;
    int lastRenderFrame;

    // Timing for decimation
    std::chrono::steady_clock::time_point lastUpdateTime;
    std::chrono::milliseconds updateInterval{50}; // 20 Hz default

    // Timer for automatic updates
    timer<> updateTimer { this, MIN_FUNCTION {
        commitFrame();
        currentFrame.store(writeFrame.load());
        redraw();

        // Re-schedule based on current updateratehz
        double rate = static_cast<double>(updateratehz);
        updateTimer.delay(1000.0 / rate);

        return {};
    }};

    // Message called after object construction to start the timer
    message<> start { this, "start", MIN_FUNCTION {
        double rate = static_cast<double>(updateratehz);
        updateTimer.delay(1000.0 / rate);
        return {};
    }};

    // Loadbang to start timer automatically and clear any stale data
    message<> loadbang { this, "loadbang", MIN_FUNCTION {
        // Clear any previous state
        writeFrame.store(0);
        currentFrame.store(0);
        lastRenderFrame = 0;
        numBins = 0;
        displayBins = 0;
        SignalInspectorData.clear();
        accumulatedFrame.clear();
        int maxFrames = static_cast<int>(frames);
        SignalInspectorData.resize(maxFrames);
        needsFullRedraw = true;

        // Start the timer
        double rate = static_cast<double>(updateratehz);
        updateTimer.delay(1000.0 / rate);
        return {};
    }};

    void commitFrame() {
        if (accumulatedFrame.empty()) return;

        // Determine which bins to store
        std::vector<double> frameToStore;
        int binsToStore = static_cast<bool>(showhalfspectrum) ? (numBins / 2) : numBins;

        frameToStore.reserve(binsToStore);
        for (int i = 0; i < binsToStore && i < static_cast<int>(accumulatedFrame.size()); ++i) {
            frameToStore.push_back(accumulatedFrame[i]);
        }

        // Add to SignalInspector buffer
        addFrame(frameToStore);
    }

    void addFrame(const std::vector<double>& frame) {
        if (frame.empty()) return;

        int maxFrames = static_cast<int>(frames);
        int current = writeFrame.load();

        // Check if buffer is full
        if (current >= maxFrames) {
            shiftBuffer();
            current = writeFrame.load();
        }

        // Store the new frame
        if (current < static_cast<int>(SignalInspectorData.size())) {
            SignalInspectorData[current] = frame;
            writeFrame.store(current + 1);
        }
    }

    void shiftBuffer() {
        // Shift by half when full
        int maxFrames = static_cast<int>(frames);
        int halfFrames = maxFrames / 2;
        int keepFrames = maxFrames - halfFrames;

        // Move second half to first half
        for (int i = 0; i < keepFrames; ++i) {
            if (halfFrames + i < static_cast<int>(SignalInspectorData.size())) {
                SignalInspectorData[i] = std::move(SignalInspectorData[halfFrames + i]);
            }
        }

        // Clear the second half
        for (int i = keepFrames; i < maxFrames && i < static_cast<int>(SignalInspectorData.size()); ++i) {
            SignalInspectorData[i].clear();
        }

        writeFrame.store(keepFrames);
        needsFullRedraw = true;
    }

    color interpolateColor(double value) {
        // Normalize value
        double normVal = (value - static_cast<double>(minValue)) /
                         (static_cast<double>(maxValue) - static_cast<double>(minValue));
        normVal = std::clamp(normVal, 0.0, 1.0);

        // Get actual color values from attributes
        color low = lowColor;
        color mid = midColor;
        color high = highColor;

        // Three-color gradient: low -> mid -> high
        double r, g, b;
        if (normVal < 0.5) {
            // Interpolate between low and mid
            double t = normVal * 2.0;
            r = low.red() * (1.0 - t) + mid.red() * t;
            g = low.green() * (1.0 - t) + mid.green() * t;
            b = low.blue() * (1.0 - t) + mid.blue() * t;
        } else {
            // Interpolate between mid and high
            double t = (normVal - 0.5) * 2.0;
            r = mid.red() * (1.0 - t) + high.red() * t;
            g = mid.green() * (1.0 - t) + high.green() * t;
            b = mid.blue() * (1.0 - t) + high.blue() * t;
        }

        return color{r, g, b, 1.0};
    }

    void ensureSurface(int width, int height) {
        if (SignalInspectorSurface && surfaceWidth == width && surfaceHeight == height) {
            return;
        }

        if (SignalInspectorSurface) {
            jgraphics_surface_destroy(SignalInspectorSurface);
        }

        SignalInspectorSurface = jgraphics_image_surface_create(
            c74::max::JGRAPHICS_FORMAT_ARGB32,
            width, height
        );
        surfaceWidth = width;
        surfaceHeight = height;
        needsFullRedraw = true;
    }

    void renderSignalInspector(int width, int height) {
        int current = currentFrame.load();

        if (!SignalInspectorSurface || current == 0 || displayBins == 0) {
            return;
        }

        c74::max::t_jgraphics* g = jgraphics_create(SignalInspectorSurface);
        if (!g) return;

        if (needsFullRedraw) {
            // Clear entire surface
            jgraphics_set_source_jrgba(g, color{0.0, 0.0, 0.0, 1.0});
            jgraphics_rectangle(g, 0, 0, width, height);
            jgraphics_fill(g);
            lastRenderFrame = 0;
        }

        // Calculate pixel dimensions
        int maxFrames = static_cast<int>(frames);
        double frameWidth = static_cast<double>(width) / maxFrames;

        // IMPORTANT: Use height divided by displayBins so half spectrum fills full height
        double binHeight = static_cast<double>(height) / displayBins;

        // Determine which frames to render (only new ones)
        int startFrame = needsFullRedraw ? 0 : lastRenderFrame;
        int endFrame = current;

        // Render frames
        for (int f = startFrame; f < endFrame && f < static_cast<int>(SignalInspectorData.size()); ++f) {
            const auto& frame = SignalInspectorData[f];
            if (frame.empty()) continue;

            double x = f * frameWidth;

            int binsInFrame = std::min(displayBins, static_cast<int>(frame.size()));
            for (int b = 0; b < binsInFrame; ++b) {
                // Y-axis: flip so low frequencies are at bottom
                double y = (displayBins - 1 - b) * binHeight;

                color pixelColor = interpolateColor(frame[b]);
                jgraphics_set_source_jrgba(g, pixelColor);
                jgraphics_rectangle(g, x, y, frameWidth + 0.5, binHeight + 0.5);
                jgraphics_fill(g);
            }
        }

        lastRenderFrame = endFrame;
        jgraphics_destroy(g);
        needsFullRedraw = false;
    }

    // Paint message
    message<> paint { this, "paint", MIN_FUNCTION {
        ui::target t { args };
        auto g = (c74::max::t_jgraphics*)t;

        const int w = static_cast<int>(t.width());
        const int h = static_cast<int>(t.height());

        // Ensure surface exists
        ensureSurface(w, h);

        // Render SignalInspector to surface
        renderSignalInspector(w, h);

        // Draw surface to screen
        if (SignalInspectorSurface) {
            c74::max::t_rect srcRect = {0, 0, static_cast<double>(w), static_cast<double>(h)};
            c74::max::t_rect destRect = {0, 0, static_cast<double>(w), static_cast<double>(h)};
            jgraphics_image_surface_draw(g, SignalInspectorSurface, srcRect, destRect);
        }

        // Draw border
        jgraphics_set_source_jrgba(g, color{0.5, 0.5, 0.5, 1.0});
        jgraphics_set_line_width(g, 1.0);
        jgraphics_rectangle(g, 0.5, 0.5, w - 1.0, h - 1.0);
        jgraphics_stroke(g);

        return {};
    }};
};

MIN_EXTERNAL(SignalInspector);