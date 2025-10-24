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
        , autoMin(0.0)
        , autoMax(100.0)
        , autoPercentile5(0.0)
        , autoPercentile95(100.0)
        , isDraggingSlider(false)
        , isDraggingTop(false)
        , isDraggingBottom(false)
        , lastBinRangeMin(0.0)
        , lastBinRangeMax(0.5)
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

    attribute<bool> autorange {
        this, "autorange", true,
        description{"Automatically adjust color range based on data min/max"}
    };

    attribute<bool> usepercentile {
        this, "usepercentile", true,
        description{"Use 5th/95th percentiles instead of absolute min/max (more robust to outliers)"}
    };

    attribute<number> gamma {
        this, "gamma", 0.5,
        range{0.1, 2.0},
        description{"Gamma correction for contrast (< 1 brightens mids, > 1 darkens mids)"}
    };

    attribute<bool> usedb {
        this, "usedb", false,
        description{"Convert to decibel scale (20*log10) before color mapping"}
    };

    attribute<number> dbfloor {
        this, "dbfloor", -60.0,
        range{-120.0, 0.0},
        description{"Floor value for dB conversion (clips values below this)"}
    };

    attribute<number> binrangemin {
        this, "binrangemin", 0.0,
        range{0.0, 1.0},
        description{"Minimum bin ratio to display (0.0 = DC, 1.0 = Nyquist)"}
    };

    attribute<number> binrangemax {
        this, "binrangemax", 0.5,
        range{0.0, 1.0},
        description{"Maximum bin ratio to display (0.0 = DC, 1.0 = Nyquist)"}
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
        autoMin = 0.0;
        autoMax = 100.0;
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
            updateDisplayBins();
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

    // Auto-range values
    double autoMin;
    double autoMax;
    double autoPercentile5;
    double autoPercentile95;

    // Slider interaction
    bool isDraggingSlider;
    bool isDraggingTop;
    bool isDraggingBottom;
    double lastBinRangeMin;
    double lastBinRangeMax;
    static constexpr int sliderWidth = 20;
    static constexpr int sliderMargin = 5;

    // Timing for decimation
    std::chrono::steady_clock::time_point lastUpdateTime;
    std::chrono::milliseconds updateInterval{50}; // 20 Hz default

    // Timer for automatic updates
    timer<> updateTimer { this, MIN_FUNCTION {
        // Check if bin range attributes changed (from inspector or messages)
        double currentMin = static_cast<double>(binrangemin);
        double currentMax = static_cast<double>(binrangemax);

        if (currentMin != lastBinRangeMin || currentMax != lastBinRangeMax) {
            updateDisplayBins();
            lastBinRangeMin = currentMin;
            lastBinRangeMax = currentMax;
        }

        commitFrame();
        currentFrame.store(writeFrame.load());

        // Update auto-range if enabled
        if (static_cast<bool>(autorange)) {
            updateAutoRange();
        }

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
        autoMin = 0.0;
        autoMax = 100.0;

        // Start the timer
        double rate = static_cast<double>(updateratehz);
        updateTimer.delay(1000.0 / rate);
        return {};
    }};

    void updateDisplayBins() {
        if (numBins == 0) return;

        // Convert ratios to bin indices
        double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

        // Ensure min <= max
        if (minRatio > maxRatio) {
            std::swap(minRatio, maxRatio);
        }

        int minBin = static_cast<int>(minRatio * (numBins - 1));
        int maxBin = static_cast<int>(maxRatio * (numBins - 1));

        // Ensure at least 1 bin
        if (minBin == maxBin) {
            if (maxBin < numBins - 1) {
                maxBin++;
            } else {
                minBin--;
            }
        }

        displayBins = maxBin - minBin + 1;
        needsFullRedraw = true;
    }

    void updateAutoRange() {
        int current = writeFrame.load();
        if (current == 0) return;

        // Calculate the actual number of frames currently visible on screen
        int maxFrames = static_cast<int>(frames);
        int visibleFrames = std::min(maxFrames, current);

        // Start from the most recent visible frames
        int startFrame = std::max(0, current - visibleFrames);

        // Collect all values for percentile calculation
        std::vector<double> allValues;
        allValues.reserve(visibleFrames * displayBins);

        double minVal = std::numeric_limits<double>::max();
        double maxVal = std::numeric_limits<double>::lowest();

        // Only analyze the frames that are currently visible
        for (int f = startFrame; f < current && f < static_cast<int>(SignalInspectorData.size()); ++f) {
            const auto& frame = SignalInspectorData[f];
            if (frame.empty()) continue;

            for (const auto& val : frame) {
                if (std::isfinite(val)) {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                    allValues.push_back(val);
                }
            }
        }

        // Only update if we found valid values
        if (minVal != std::numeric_limits<double>::max() &&
            maxVal != std::numeric_limits<double>::lowest() &&
            !allValues.empty()) {

            // Calculate absolute min/max with margin
            double range = maxVal - minVal;
            if (range < 1e-6) {
                range = 1.0;
            }
            autoMin = minVal - range * 0.05;
            autoMax = maxVal + range * 0.05;

            // Calculate percentiles
            std::sort(allValues.begin(), allValues.end());
            size_t p5_idx = static_cast<size_t>(allValues.size() * 0.05);
            size_t p95_idx = static_cast<size_t>(allValues.size() * 0.95);

            autoPercentile5 = allValues[p5_idx];
            autoPercentile95 = allValues[p95_idx];

            // Add small margin to percentiles too
            double percentileRange = autoPercentile95 - autoPercentile5;
            if (percentileRange < 1e-6) {
                percentileRange = 1.0;
            }
            autoPercentile5 -= percentileRange * 0.02;
            autoPercentile95 += percentileRange * 0.02;
        }
    }

    void commitFrame() {
        if (accumulatedFrame.empty() || numBins == 0) return;

        // Convert ratios to bin indices
        double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

        if (minRatio > maxRatio) {
            std::swap(minRatio, maxRatio);
        }

        int minBin = static_cast<int>(minRatio * (numBins - 1));
        int maxBin = static_cast<int>(maxRatio * (numBins - 1));

        if (minBin == maxBin) {
            if (maxBin < numBins - 1) {
                maxBin++;
            } else if (minBin > 0) {
                minBin--;
            }
        }

        std::vector<double> frameToStore;
        int binsToStore = maxBin - minBin + 1;
        frameToStore.reserve(binsToStore);

        for (int i = minBin; i <= maxBin && i < static_cast<int>(accumulatedFrame.size()); ++i) {
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
        // Step 1: Convert to dB if enabled
        if (static_cast<bool>(usedb)) {
            // Handle zero/negative values
            if (value <= 0.0) {
                value = static_cast<double>(dbfloor);
            } else {
                value = 20.0 * std::log10(value);
                // Clip to floor
                value = std::max(value, static_cast<double>(dbfloor));
            }
        }

        // Step 2: Determine min/max range for normalization
        double minVal, maxVal;

        if (static_cast<bool>(autorange)) {
            // Use percentiles if enabled, otherwise absolute min/max
            if (static_cast<bool>(usepercentile)) {
                minVal = autoPercentile5;
                maxVal = autoPercentile95;
            } else {
                minVal = autoMin;
                maxVal = autoMax;
            }
        } else {
            // Manual range
            minVal = static_cast<double>(minValue);
            maxVal = static_cast<double>(maxValue);
        }

        // Step 3: Normalize to 0-1
        double normVal = (value - minVal) / (maxVal - minVal);
        normVal = std::clamp(normVal, 0.0, 1.0);

        // Step 4: Apply gamma correction
        double gammaVal = static_cast<double>(gamma);
        normVal = std::pow(normVal, gammaVal);

        // Step 5: Map to color gradient
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

        // Calculate pixel dimensions - account for slider on left
        int spectrogramLeft = sliderWidth + sliderMargin * 2;
        int spectrogramWidth = width - spectrogramLeft;

        int maxFrames = static_cast<int>(frames);
        double frameWidth = static_cast<double>(spectrogramWidth) / maxFrames;

        // CRITICAL: Always use full height divided by displayBins
        // This ensures the selected range fills the entire vertical space
        double binHeight = static_cast<double>(height) / displayBins;

        // Determine which frames to render (only new ones)
        int startFrame = needsFullRedraw ? 0 : lastRenderFrame;
        int endFrame = current;

        // Render frames
        for (int f = startFrame; f < endFrame && f < static_cast<int>(SignalInspectorData.size()); ++f) {
            const auto& frame = SignalInspectorData[f];
            if (frame.empty()) continue;

            double x = spectrogramLeft + f * frameWidth;

            // The frame data contains only the selected bins (displayBins count)
            // We render each bin scaled to fill the full height
            int binsInFrame = std::min(displayBins, static_cast<int>(frame.size()));
            for (int b = 0; b < binsInFrame; ++b) {
                // Y-axis: flip so low frequencies are at bottom
                // Map bin index b to full height range
                double y = height - (b + 1) * binHeight;

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

    void drawSlider(c74::max::t_jgraphics* g, int width, int height) {
        if (numBins == 0) return;

        // Get ratio values (0.0 to 1.0)
        double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

        // Convert to screen coordinates (invert because low freq at bottom)
        double sliderBottom = (1.0 - minRatio) * height;  // Low freq at bottom
        double sliderTop = (1.0 - maxRatio) * height;     // High freq at top
        double sliderHeight = sliderBottom - sliderTop;

        // Draw slider background (full spectrum range)
        jgraphics_set_source_jrgba(g, color{0.2, 0.2, 0.2, 1.0});
        jgraphics_rectangle(g, sliderMargin, 0, sliderWidth, height);
        jgraphics_fill(g);

        // Draw selected range
        jgraphics_set_source_jrgba(g, color{0.4, 0.6, 0.8, 1.0});
        jgraphics_rectangle(g, sliderMargin, sliderTop, sliderWidth, sliderHeight);
        jgraphics_fill(g);

        // Draw handles
        double handleHeight = 8.0;

        // Top handle (high frequency)
        jgraphics_set_source_jrgba(g, color{0.8, 0.8, 0.8, 1.0});
        jgraphics_rectangle(g, sliderMargin, sliderTop - handleHeight/2, sliderWidth, handleHeight);
        jgraphics_fill(g);

        // Bottom handle (low frequency)
        jgraphics_rectangle(g, sliderMargin, sliderBottom - handleHeight/2, sliderWidth, handleHeight);
        jgraphics_fill(g);

        // Draw border around slider
        jgraphics_set_source_jrgba(g, color{0.5, 0.5, 0.5, 1.0});
        jgraphics_set_line_width(g, 1.0);
        jgraphics_rectangle(g, sliderMargin + 0.5, 0.5, sliderWidth, height - 1.0);
        jgraphics_stroke(g);
    }

    void drawMinMaxValues(c74::max::t_jgraphics* g, int width, int height) {
        // Get the actual min/max values being used for coloring
        double minVal, maxVal;

        if (static_cast<bool>(autorange)) {
            if (static_cast<bool>(usepercentile)) {
                minVal = autoPercentile5;
                maxVal = autoPercentile95;
            } else {
                minVal = autoMin;
                maxVal = autoMax;
            }
        } else {
            minVal = static_cast<double>(minValue);
            maxVal = static_cast<double>(maxValue);
        }

        // Format the values with dB indicator if enabled
        char minText[64];
        char maxText[64];
        const char* unit = static_cast<bool>(usedb) ? " dB" : "";
        snprintf(minText, sizeof(minText), "Min: %.2f%s", minVal, unit);
        snprintf(maxText, sizeof(maxText), "Max: %.2f%s", maxVal, unit);

        // Set text properties
        jgraphics_set_source_jrgba(g, color{1.0, 1.0, 1.0, 0.9});  // White with slight transparency
        jgraphics_select_font_face(g, "Arial", c74::max::JGRAPHICS_FONT_SLANT_NORMAL, c74::max::JGRAPHICS_FONT_WEIGHT_BOLD);
        jgraphics_set_font_size(g, 11.0);

        // Calculate text position (top right corner with padding)
        double padding = 8.0;
        double lineHeight = 14.0;

        double maxTextWidth, maxTextHeight;
        jgraphics_text_measure(g, maxText, &maxTextWidth, &maxTextHeight);
        double maxTextX = width - maxTextWidth - padding;
        double maxTextY = padding + lineHeight;

        double minTextWidth, minTextHeight;
        jgraphics_text_measure(g, minText, &minTextWidth, &minTextHeight);
        double minTextX = width - minTextWidth - padding;
        double minTextY = maxTextY + lineHeight;

        // Draw text with slight shadow for readability
        jgraphics_set_source_jrgba(g, color{0.0, 0.0, 0.0, 0.5});
        jgraphics_move_to(g, maxTextX + 1, maxTextY + 1);
        jgraphics_show_text(g, maxText);
        jgraphics_move_to(g, minTextX + 1, minTextY + 1);
        jgraphics_show_text(g, minText);

        // Draw actual text
        jgraphics_set_source_jrgba(g, color{1.0, 1.0, 1.0, 0.9});
        jgraphics_move_to(g, maxTextX, maxTextY);
        jgraphics_show_text(g, maxText);
        jgraphics_move_to(g, minTextX, minTextY);
        jgraphics_show_text(g, minText);
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

        // Draw the frequency range slider on top
        drawSlider(g, w, h);

        // Draw min/max values in top right corner
        drawMinMaxValues(g, w, h);

        // Draw border
        jgraphics_set_source_jrgba(g, color{0.5, 0.5, 0.5, 1.0});
        jgraphics_set_line_width(g, 1.0);
        jgraphics_rectangle(g, 0.5, 0.5, w - 1.0, h - 1.0);
        jgraphics_stroke(g);

        return {};
    }};

    // Mouse interaction for slider
    message<> mousedown { this, "mousedown", MIN_FUNCTION {
        ui::target t { args };
        const int w = static_cast<int>(t.width());
        const int h = static_cast<int>(t.height());

        double x = args[0];
        double y = args[1];

        // Check if click is in slider area
        if (x >= sliderMargin && x <= (sliderMargin + sliderWidth) && numBins > 0) {
            double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
            double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

            // Convert ratio to y positions (inverted - low freq at bottom)
            double sliderBottom = (1.0 - minRatio) * h;  // Low freq at bottom
            double sliderTop = (1.0 - maxRatio) * h;     // High freq at top

            double handleHeight = 12.0;  // Increased for easier clicking

            // Check which handle or area was clicked (with priority to handles)
            if (y >= (sliderTop - handleHeight) && y <= (sliderTop + handleHeight)) {
                isDraggingTop = true;
                isDraggingSlider = true;
            } else if (y >= (sliderBottom - handleHeight) && y <= (sliderBottom + handleHeight)) {
                isDraggingBottom = true;
                isDraggingSlider = true;
            } else if (y > sliderTop && y < sliderBottom) {
                // Clicked in middle - drag both
                isDraggingTop = true;
                isDraggingBottom = true;
                isDraggingSlider = true;
            }
        }

        return {};
    }};

    message<> mousedrag { this, "mousedrag", MIN_FUNCTION {
        if (!isDraggingSlider || numBins == 0) return {};

        ui::target t { args };
        const int h = static_cast<int>(t.height());
        double y = args[1];

        // Clamp y to bounds
        y = std::clamp(y, 0.0, static_cast<double>(h));

        // Convert y position to ratio (inverted because low freq at bottom)
        double ratio = 1.0 - (y / h);  // 0 at bottom (DC), 1 at top (Nyquist)
        ratio = std::clamp(ratio, 0.0, 1.0);

        double currentMin = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double currentMax = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

        if (isDraggingTop && isDraggingBottom) {
            // Dragging middle - move both maintaining range
            double range = currentMax - currentMin;
            double centerRatio = ratio;
            double newMin = centerRatio - range / 2.0;
            double newMax = newMin + range;

            // Keep in bounds
            if (newMin < 0.0) {
                newMin = 0.0;
                newMax = range;
            }
            if (newMax > 1.0) {
                newMax = 1.0;
                newMin = newMax - range;
            }

            // Use atoms vector for set() method
            binrangemin.set({newMin});
            binrangemax.set({newMax});
        } else if (isDraggingTop) {
            // Dragging top handle (high frequency)
            double newMax = std::max(ratio, currentMin + 0.01);  // Ensure minimum range
            newMax = std::clamp(newMax, 0.0, 1.0);
            binrangemax.set({newMax});
        } else if (isDraggingBottom) {
            // Dragging bottom handle (low frequency)
            double newMin = std::min(ratio, currentMax - 0.01);  // Ensure minimum range
            newMin = std::clamp(newMin, 0.0, 1.0);
            binrangemin.set({newMin});
        }

        // Force immediate update
        updateDisplayBins();
        redraw();

        return {};
    }};

    message<> mouseup { this, "mouseup", MIN_FUNCTION {
        isDraggingSlider = false;
        isDraggingTop = false;
        isDraggingBottom = false;
        return {};
    }};
};

MIN_EXTERNAL(SignalInspector);