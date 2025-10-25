// SignalInspector.cpp - Efficient STFT visualizer with threaded queue-based frame processing

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
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>

using namespace c74::min;
using namespace c74::min::ui;

// -----------------------------------------------------------------------------
// Thread-safe queue with proper shutdown support
// -----------------------------------------------------------------------------
template <typename T>
class tsqueue {
public:
    void enqueue(T v) {
        std::lock_guard<std::mutex> lock(m_);
        if (shutdown_) return; // Don't accept new items during shutdown
        q_.emplace_back(std::move(v));
        cv_.notify_one();
    }

    bool try_dequeue(T& out) {
        std::lock_guard<std::mutex> lock(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool wait_dequeue(T& out, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(m_);
        if (cv_.wait_for(lock, timeout, [this] { return !q_.empty() || shutdown_; })) {
            if (shutdown_ && q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
            return true;
        }
        return false;
    }

    void wake_all() {
        std::lock_guard<std::mutex> lock(m_);
        shutdown_ = true;
        cv_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_);
        q_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_);
        return q_.size();
    }

private:
    mutable std::mutex m_;
    std::deque<T> q_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
};

class SignalInspector : public object<SignalInspector>, public ui_operator<400, 300>, public vector_operator<> {
public:
    MIN_DESCRIPTION { "Efficient STFT SignalInspector visualizer with threaded queue-based processing" };
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
        , writePosition(0)
        , numBins(0)
        , displayBins(0)
        , needsFullRedraw(true)
        , lastRenderFrame(0)
        , lastRenderPosition(-1)
        , autoMin(0.0)
        , autoMax(100.0)
        , autoPercentile5(0.0)
        , autoPercentile95(100.0)
        , isDraggingSlider(false)
        , isDraggingTop(false)
        , isDraggingBottom(false)
        , lastBinRangeMin(0.0)
        , lastBinRangeMax(0.5)
        , plotThreadRunning(false)
    {
        // Initialize with default frame count
        SignalInspectorData.resize(200);

        // Start the plotting thread
        startPlotThread();
    }

    ~SignalInspector() {
        // Stop the plotting thread first
        stopPlotThread();

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

    attribute<bool> scrolling {
        this, "scrolling", false,
        description{"Scrolling mode: true = circular buffer (scroll), false = left-to-right with shift"}
    };

    attribute<number> minValue { this, "minval", 0.0, description{"Minimum value (linear)"} };
    attribute<number> maxValue { this, "maxval", 100.0, description{"Maximum value (linear)"} };

    // Color scheme attributes
    attribute<color> lowColor { this, "lowcolor", color{0.0, 0.0, 0.2, 1.0} };
    attribute<color> midColor { this, "midcolor", color{1.0, 0.5, 0.0, 1.0} };
    attribute<color> highColor { this, "highcolor", color{1.0, 1.0, 0.0, 1.0} };

    // Slider color attributes
    attribute<color> sliderBgColor { this, "sliderbgcolor", color{0.2, 0.2, 0.2, 1.0},
        description{"Slider background color"} };
    attribute<color> sliderSelColor { this, "sliderselcolor", color{0.4, 0.6, 0.8, 1.0},
        description{"Slider selection color"} };
    attribute<color> sliderHandleColor { this, "sliderhandlecolor", color{0.8, 0.8, 0.8, 1.0},
        description{"Slider handle color"} };
    attribute<color> sliderBorderColor { this, "sliderbordercolor", color{0.5, 0.5, 0.5, 1.0},
        description{"Slider border color"} };

    // Clear the SignalInspector
    message<> clear { this, "clear", MIN_FUNCTION {
        // Clear the queue
        frameQueue.clear();

        std::lock_guard<std::mutex> lock(dataMutex);
        writeFrame.store(0);
        currentFrame.store(0);
        writePosition.store(0);
        lastRenderFrame = 0;
        lastRenderPosition = -1;
        numBins = 0;
        displayBins = 0;
        SignalInspectorData.clear();
        int maxFrames = static_cast<int>(frames);
        SignalInspectorData.resize(maxFrames);
        needsFullRedraw = true;
        autoMin = 0.0;
        autoMax = 100.0;
        return {};
    }};

    // Vector operator perform method - pushes frames to queue
    void operator()(audio_bundle input, audio_bundle output) {
        // Get the input vector size
        auto in_vec = input.samples(0);
        int vecsize = static_cast<int>(input.frame_count());

        // Create a frame vector and push to queue
        std::vector<double> frame;
        frame.reserve(vecsize);
        for (int i = 0; i < vecsize; ++i) {
            frame.push_back(static_cast<double>(in_vec[i]));
        }

        // Push to queue for processing by plot thread
        frameQueue.enqueue(std::move(frame));

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
    std::mutex dataMutex;  // Protects SignalInspectorData
    c74::max::t_jsurface* SignalInspectorSurface;
    int surfaceWidth, surfaceHeight;
    std::atomic<int> currentFrame;  // For rendering
    std::atomic<int> writeFrame;    // For writing new data
    std::atomic<int> writePosition; // Current write position for non-scrolling mode (0 to maxFrames-1)
    int numBins;        // Total bins from input
    int displayBins;    // Bins to actually display
    bool needsFullRedraw;
    int lastRenderFrame;
    int lastRenderPosition; // Track last rendered position in non-scrolling mode

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

    // Queue for incoming frames
    tsqueue<std::vector<double>> frameQueue;

    // Plotting thread
    std::thread plotThread;
    std::atomic<bool> plotThreadRunning;

    // Start the plotting thread
    void startPlotThread() {
        plotThreadRunning = true;
        plotThread = std::thread([this]() {
            this->plotThreadLoop();
        });
    }

    // Stop the plotting thread
    void stopPlotThread() {
        plotThreadRunning = false;
        frameQueue.wake_all();
        if (plotThread.joinable()) {
            plotThread.join();
        }
    }

    // Main plotting thread loop
    void plotThreadLoop() {
        while (plotThreadRunning) {
            // Calculate wait time based on frame rate
            double rate = static_cast<double>(updateratehz);
            std::chrono::milliseconds waitTime(static_cast<int>(1000.0 / rate));

            // Wait for the frame interval
            std::this_thread::sleep_for(waitTime);

            // Process all queued frames
            bool processedAny = false;
            std::vector<double> frame;

            while (frameQueue.try_dequeue(frame)) {
                commitFrame(frame);
                processedAny = true;
            }

            // If we processed any frames, update the display
            if (processedAny) {
                currentFrame.store(writeFrame.load());

                // Check if bin range attributes changed
                double currentMin = static_cast<double>(binrangemin);
                double currentMax = static_cast<double>(binrangemax);

                if (currentMin != lastBinRangeMin || currentMax != lastBinRangeMax) {
                    updateDisplayBins();
                    lastBinRangeMin = currentMin;
                    lastBinRangeMax = currentMax;
                }

                // Update auto-range if enabled
                if (static_cast<bool>(autorange)) {
                    updateAutoRange();
                }

                // Trigger redraw
                redraw();
            }
        }
    }

    // Commit a frame to the SignalInspector data
    void commitFrame(const std::vector<double>& frame) {
        if (frame.empty()) return;

        int maxFrames = static_cast<int>(frames);
        bool isScrolling = static_cast<bool>(scrolling);

        {
            std::lock_guard<std::mutex> lock(dataMutex);

            // Ensure we have space in SignalInspectorData
            if (SignalInspectorData.empty() || SignalInspectorData.size() != static_cast<size_t>(maxFrames)) {
                SignalInspectorData.resize(maxFrames);
            }

            if (isScrolling) {
                // Scrolling mode: circular buffer
                int wf = writeFrame.load();
                SignalInspectorData[wf] = frame;
                writeFrame.store((wf + 1) % maxFrames);
            } else {
                // Non-scrolling mode: linear left-to-right
                int wp = writePosition.load();

                // Check if we need to shift
                if (wp >= maxFrames) {
                    // Shift everything left by half
                    int shiftAmount = maxFrames / 2;
                    for (int i = 0; i < maxFrames - shiftAmount; ++i) {
                        SignalInspectorData[i] = SignalInspectorData[i + shiftAmount];
                    }
                    // Clear the right half
                    for (int i = maxFrames - shiftAmount; i < maxFrames; ++i) {
                        SignalInspectorData[i].clear();
                    }
                    wp = maxFrames - shiftAmount;
                    needsFullRedraw = true; // Need full redraw after shift
                }

                // Store frame at current position
                SignalInspectorData[wp] = frame;
                writePosition.store(wp + 1);
            }
        }
    }

    void updateDisplayBins() {
        if (numBins == 0) return;

        double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

        int startBin = static_cast<int>(minRatio * numBins);
        int endBin = static_cast<int>(maxRatio * numBins);

        startBin = std::clamp(startBin, 0, numBins - 1);
        endBin = std::clamp(endBin, startBin + 1, numBins);

        displayBins = endBin - startBin;
        needsFullRedraw = true;
    }

    void updateAutoRange() {
        int maxFrames = static_cast<int>(frames);
        if (displayBins == 0) return;

        std::vector<double> allValues;
        allValues.reserve(maxFrames * displayBins);

        double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);
        int startBin = static_cast<int>(minRatio * numBins);
        int endBin = static_cast<int>(maxRatio * numBins);

        bool convertDb = static_cast<bool>(usedb);
        double dbFloor = static_cast<double>(dbfloor);

        {
            std::lock_guard<std::mutex> lock(dataMutex);
            if (SignalInspectorData.empty()) return;

            for (const auto& frame : SignalInspectorData) {
                if (frame.empty()) continue;
                for (int b = startBin; b < endBin && b < static_cast<int>(frame.size()); ++b) {
                    double val = frame[b];
                    if (convertDb) {
                        val = (val > 0.0) ? 20.0 * std::log10(val) : dbFloor;
                        val = std::max(val, dbFloor);
                    }
                    allValues.push_back(val);
                }
            }
        }

        if (allValues.empty()) return;

        auto minmax = std::minmax_element(allValues.begin(), allValues.end());
        autoMin = *minmax.first;
        autoMax = *minmax.second;

        if (static_cast<bool>(usepercentile) && allValues.size() > 10) {
            std::sort(allValues.begin(), allValues.end());
            size_t idx5 = static_cast<size_t>(0.05 * allValues.size());
            size_t idx95 = static_cast<size_t>(0.95 * allValues.size());
            idx5 = std::min(idx5, allValues.size() - 1);
            idx95 = std::min(idx95, allValues.size() - 1);
            autoPercentile5 = allValues[idx5];
            autoPercentile95 = allValues[idx95];
        } else {
            autoPercentile5 = autoMin;
            autoPercentile95 = autoMax;
        }
    }

    color interpolateColor(double t, const color& c1, const color& c2) const {
        t = std::clamp(t, 0.0, 1.0);
        return color{
            c1.red() + t * (c2.red() - c1.red()),
            c1.green() + t * (c2.green() - c1.green()),
            c1.blue() + t * (c2.blue() - c1.blue()),
            c1.alpha() + t * (c2.alpha() - c1.alpha())
        };
    }

    color valueToColor(double value) const {
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

        if (maxVal <= minVal) maxVal = minVal + 1.0;

        double normalized = (value - minVal) / (maxVal - minVal);
        normalized = std::clamp(normalized, 0.0, 1.0);

        double g = static_cast<double>(gamma);
        normalized = std::pow(normalized, g);

        color low = static_cast<color>(lowColor);
        color mid = static_cast<color>(midColor);
        color high = static_cast<color>(highColor);

        if (normalized < 0.5) {
            return interpolateColor(normalized * 2.0, low, mid);
        } else {
            return interpolateColor((normalized - 0.5) * 2.0, mid, high);
        }
    }

    void ensureSurface(int width, int height) {
        if (!SignalInspectorSurface || surfaceWidth != width || surfaceHeight != height) {
            if (SignalInspectorSurface) {
                jgraphics_surface_destroy(SignalInspectorSurface);
            }
            SignalInspectorSurface = jgraphics_image_surface_create(c74::max::JGRAPHICS_FORMAT_ARGB32, width, height);
            surfaceWidth = width;
            surfaceHeight = height;
            needsFullRedraw = true;
        }
    }

    void renderSignalInspector(int width, int height) {
        if (!SignalInspectorSurface || displayBins == 0) return;

        auto g = jgraphics_create(SignalInspectorSurface);
        if (!g) return;

        int maxFrames = static_cast<int>(frames);
        // Slider is on the right, so spectrogram width is reduced from the right
        int SignalInspectorWidth = width - sliderWidth - 2 * sliderMargin;
        if (SignalInspectorWidth <= 0) {
            jgraphics_destroy(g);
            return;
        }

        double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);
        int startBin = static_cast<int>(minRatio * numBins);
        int endBin = static_cast<int>(maxRatio * numBins);

        bool convertDb = static_cast<bool>(usedb);
        double dbFloor = static_cast<double>(dbfloor);
        bool isScrolling = static_cast<bool>(scrolling);

        double pixelWidth = static_cast<double>(SignalInspectorWidth) / maxFrames;
        double pixelHeight = static_cast<double>(height) / displayBins;

        // Lock only during data access
        std::lock_guard<std::mutex> lock(dataMutex);

        // Check again after acquiring lock
        if (SignalInspectorData.empty()) {
            jgraphics_destroy(g);
            return;
        }

        if (isScrolling) {
            // SCROLLING MODE: Render entire circular buffer
            int cf = currentFrame.load();

            if (needsFullRedraw || cf != lastRenderFrame) {
                // Clear background
                jgraphics_set_source_jrgba(g, color{0.0, 0.0, 0.0, 1.0});
                jgraphics_rectangle(g, 0, 0, width, height);
                jgraphics_fill(g);

                // Render all frames
                for (int f = 0; f < maxFrames; ++f) {
                    int dataIndex = (cf + f) % maxFrames;

                    if (dataIndex >= static_cast<int>(SignalInspectorData.size())) continue;

                    const auto& frame = SignalInspectorData[dataIndex];
                    if (frame.empty()) continue;

                    for (int b = 0; b < displayBins; ++b) {
                        int binIndex = startBin + b;
                        if (binIndex >= static_cast<int>(frame.size())) continue;

                        double value = frame[binIndex];
                        if (convertDb) {
                            value = (value > 0.0) ? 20.0 * std::log10(value) : dbFloor;
                            value = std::max(value, dbFloor);
                        }

                        color c = valueToColor(value);
                        jgraphics_set_source_jrgba(g, c);

                        double x = f * pixelWidth;
                        double y = (displayBins - 1 - b) * pixelHeight;

                        jgraphics_rectangle(g, x, y, std::ceil(pixelWidth) + 0.5, std::ceil(pixelHeight) + 0.5);
                        jgraphics_fill(g);
                    }
                }

                needsFullRedraw = false;
                lastRenderFrame = cf;
            }
        } else {
            // NON-SCROLLING MODE: Incremental left-to-right rendering
            int wp = writePosition.load();

            // Full redraw if needed (after shift or first time)
            if (needsFullRedraw) {
                // Clear background
                jgraphics_set_source_jrgba(g, color{0.0, 0.0, 0.0, 1.0});
                jgraphics_rectangle(g, 0, 0, width, height);
                jgraphics_fill(g);

                // Render all frames up to current write position
                for (int f = 0; f < wp && f < maxFrames; ++f) {
                    if (f >= static_cast<int>(SignalInspectorData.size())) break;

                    const auto& frame = SignalInspectorData[f];
                    if (frame.empty()) continue;

                    for (int b = 0; b < displayBins; ++b) {
                        int binIndex = startBin + b;
                        if (binIndex >= static_cast<int>(frame.size())) continue;

                        double value = frame[binIndex];
                        if (convertDb) {
                            value = (value > 0.0) ? 20.0 * std::log10(value) : dbFloor;
                            value = std::max(value, dbFloor);
                        }

                        color c = valueToColor(value);
                        jgraphics_set_source_jrgba(g, c);

                        double x = f * pixelWidth;
                        double y = (displayBins - 1 - b) * pixelHeight;

                        jgraphics_rectangle(g, x, y, std::ceil(pixelWidth) + 0.5, std::ceil(pixelHeight) + 0.5);
                        jgraphics_fill(g);
                    }
                }

                needsFullRedraw = false;
                lastRenderPosition = wp - 1;
            } else {
                // INCREMENTAL: Only render new frames since last render
                int startPos = lastRenderPosition + 1;
                int endPos = wp;

                for (int f = startPos; f < endPos && f < maxFrames; ++f) {
                    if (f >= static_cast<int>(SignalInspectorData.size())) break;

                    const auto& frame = SignalInspectorData[f];
                    if (frame.empty()) continue;

                    for (int b = 0; b < displayBins; ++b) {
                        int binIndex = startBin + b;
                        if (binIndex >= static_cast<int>(frame.size())) continue;

                        double value = frame[binIndex];
                        if (convertDb) {
                            value = (value > 0.0) ? 20.0 * std::log10(value) : dbFloor;
                            value = std::max(value, dbFloor);
                        }

                        color c = valueToColor(value);
                        jgraphics_set_source_jrgba(g, c);

                        double x = f * pixelWidth;
                        double y = (displayBins - 1 - b) * pixelHeight;

                        jgraphics_rectangle(g, x, y, std::ceil(pixelWidth) + 0.5, std::ceil(pixelHeight) + 0.5);
                        jgraphics_fill(g);
                    }
                }

                lastRenderPosition = endPos - 1;
            }
        }

        jgraphics_destroy(g);
    }

    void drawSlider(c74::max::t_jgraphics* g, int width, int height) {
        if (numBins == 0) return;

        double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

        double sliderBottom = (1.0 - minRatio) * height;
        double sliderTop = (1.0 - maxRatio) * height;
        double sliderHeight = sliderBottom - sliderTop;

        // Position slider on the right side
        double sliderX = width - sliderWidth - sliderMargin;

        // Background
        jgraphics_set_source_jrgba(g, static_cast<color>(sliderBgColor));
        jgraphics_rectangle(g, sliderX, 0, sliderWidth, height);
        jgraphics_fill(g);

        // Selection
        jgraphics_set_source_jrgba(g, static_cast<color>(sliderSelColor));
        jgraphics_rectangle(g, sliderX, sliderTop, sliderWidth, sliderHeight);
        jgraphics_fill(g);

        double handleHeight = 8.0;

        // Handles
        jgraphics_set_source_jrgba(g, static_cast<color>(sliderHandleColor));
        jgraphics_rectangle(g, sliderX, sliderTop - handleHeight/2, sliderWidth, handleHeight);
        jgraphics_fill(g);

        jgraphics_rectangle(g, sliderX, sliderBottom - handleHeight/2, sliderWidth, handleHeight);
        jgraphics_fill(g);

        // Border
        jgraphics_set_source_jrgba(g, static_cast<color>(sliderBorderColor));
        jgraphics_set_line_width(g, 1.0);
        jgraphics_rectangle(g, sliderX + 0.5, 0.5, sliderWidth, height - 1.0);
        jgraphics_stroke(g);
    }

    void drawMinMaxValues(c74::max::t_jgraphics* g, int width, int height) {
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

        char minText[64];
        char maxText[64];
        const char* unit = static_cast<bool>(usedb) ? " dB" : "";
        snprintf(minText, sizeof(minText), "Min: %.2f%s", minVal, unit);
        snprintf(maxText, sizeof(maxText), "Max: %.2f%s", maxVal, unit);

        jgraphics_set_source_jrgba(g, color{1.0, 1.0, 1.0, 0.9});
        jgraphics_select_font_face(g, "Arial", c74::max::JGRAPHICS_FONT_SLANT_NORMAL, c74::max::JGRAPHICS_FONT_WEIGHT_BOLD);
        jgraphics_set_font_size(g, 11.0);

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

        jgraphics_set_source_jrgba(g, color{0.0, 0.0, 0.0, 0.5});
        jgraphics_move_to(g, maxTextX + 1, maxTextY + 1);
        jgraphics_show_text(g, maxText);
        jgraphics_move_to(g, minTextX + 1, minTextY + 1);
        jgraphics_show_text(g, minText);

        jgraphics_set_source_jrgba(g, color{1.0, 1.0, 1.0, 0.9});
        jgraphics_move_to(g, maxTextX, maxTextY);
        jgraphics_show_text(g, maxText);
        jgraphics_move_to(g, minTextX, minTextY);
        jgraphics_show_text(g, minText);
    }

    message<> paint { this, "paint", MIN_FUNCTION {
        ui::target t { args };
        auto g = (c74::max::t_jgraphics*)t;

        const int w = static_cast<int>(t.width());
        const int h = static_cast<int>(t.height());

        ensureSurface(w, h);
        renderSignalInspector(w, h);

        if (SignalInspectorSurface) {
            c74::max::t_rect srcRect = {0, 0, static_cast<double>(w), static_cast<double>(h)};
            c74::max::t_rect destRect = {0, 0, static_cast<double>(w), static_cast<double>(h)};
            jgraphics_image_surface_draw(g, SignalInspectorSurface, srcRect, destRect);
        }

        drawSlider(g, w, h);
        drawMinMaxValues(g, w, h);

        jgraphics_set_source_jrgba(g, color{0.5, 0.5, 0.5, 1.0});
        jgraphics_set_line_width(g, 1.0);
        jgraphics_rectangle(g, 0.5, 0.5, w - 1.0, h - 1.0);
        jgraphics_stroke(g);

        return {};
    }};

    message<> mousedown { this, "mousedown", MIN_FUNCTION {
        // Use event class to extract coordinates properly
        event e { args };
        auto t = e.target();

        const double w = t.width();
        const double h = t.height();
        const double x = e.x();
        const double y = e.y();

        // Slider is on the right side
        double sliderX = w - sliderWidth - sliderMargin;

        if (x >= sliderX && x <= (sliderX + sliderWidth) && numBins > 0) {
            double minRatio = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
            double maxRatio = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

            double sliderBottom = (1.0 - minRatio) * h;
            double sliderTop = (1.0 - maxRatio) * h;

            double handleHeight = 12.0;

            if (y >= (sliderTop - handleHeight) && y <= (sliderTop + handleHeight)) {
                isDraggingTop = true;
                isDraggingSlider = true;
            } else if (y >= (sliderBottom - handleHeight) && y <= (sliderBottom + handleHeight)) {
                isDraggingBottom = true;
                isDraggingSlider = true;
            } else if (y > sliderTop && y < sliderBottom) {
                isDraggingTop = true;
                isDraggingBottom = true;
                isDraggingSlider = true;
            }
        }

        return {};
    }};

    message<> mousedoubleclick { this, "mousedoubleclick", MIN_FUNCTION {
        // Use event class to extract coordinates properly
        event e { args };
        auto t = e.target();

        const double w = t.width();
        const double x = e.x();

        // Slider is on the right side
        double sliderX = w - sliderWidth - sliderMargin;

        // Check if double-click is on the slider
        if (x >= sliderX && x <= (sliderX + sliderWidth) && numBins > 0) {
            // Reset to default values
            binrangemin.set({0.0});
            binrangemax.set({0.5});

            updateDisplayBins();
            needsFullRedraw = true;
            redraw();
        }

        return {};
    }};

    message<> mousedrag { this, "mousedrag", MIN_FUNCTION {
        if (!isDraggingSlider || numBins == 0) return {};

        // Use event class to extract coordinates properly
        event e { args };
        auto t = e.target();

        const double h = t.height();
        const double y = e.y();

        double clampedY = std::clamp(y, 0.0, h);

        double ratio = 1.0 - (clampedY / h);
        ratio = std::clamp(ratio, 0.0, 1.0);

        double currentMin = std::clamp(static_cast<double>(binrangemin), 0.0, 1.0);
        double currentMax = std::clamp(static_cast<double>(binrangemax), 0.0, 1.0);

        if (isDraggingTop && isDraggingBottom) {
            double range = currentMax - currentMin;
            double centerRatio = ratio;
            double newMin = centerRatio - range / 2.0;
            double newMax = newMin + range;

            if (newMin < 0.0) {
                newMin = 0.0;
                newMax = range;
            }
            if (newMax > 1.0) {
                newMax = 1.0;
                newMin = newMax - range;
            }

            binrangemin.set({newMin});
            binrangemax.set({newMax});
        } else if (isDraggingTop) {
            double newMax = std::max(ratio, currentMin + 0.01);
            newMax = std::clamp(newMax, 0.0, 1.0);
            binrangemax.set({newMax});
        } else if (isDraggingBottom) {
            double newMin = std::min(ratio, currentMax - 0.01);
            newMin = std::clamp(newMin, 0.0, 1.0);
            binrangemin.set({newMin});
        }

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