// based on KNLMeansCL by Khanattila

#include <algorithm>
#include <array>
#include <atomic>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include "common.h"

static bool success(cudaError_t result) {
    return result == cudaSuccess;
}

static const char * get_error(cudaError_t error) {
    return cudaGetErrorString(error);
}

#define showError(expr) show_error_impl(expr, # expr, __LINE__)
template <typename T>
static void show_error_impl(T result, const char * source, int line_no) {
    if (!success(result)) [[unlikely]] {
        std::fprintf(stderr, "[%d] %s failed: %s\n", line_no, source, get_error(result));
    }
}

#define checkError(expr) do {                                                       \
    if (auto result = expr; !success(result)) [[unlikely]] {                        \
        std::ostringstream error;                                                   \
        error << '[' << __LINE__ << "] '" # expr "' failed: " << get_error(result); \
        return set_error(error.str().c_str());                                      \
    }                                                                               \
} while (0)

static void cudaStreamDestroyCustom(cudaStream_t stream) {
    showError(cudaStreamDestroy(stream));
}

static void cudaEventDestroyCustom(cudaEvent_t event) {
    showError(cudaEventDestroy(event));
}

static void cudaFreeCustom(void * p) {
    showError(cudaFree(p));
}

static void cudaFreeHostCustom(void * p) {
    showError(cudaFreeHost(p));
}


struct node_freer {
    const VSAPI * & vsapi;
    VSNodeRef * node {};
    void release() {
        node = nullptr;
    }
    ~node_freer() {
        if (node) {
            vsapi->freeNode(node);
        }
    }
};


template <typename T, auto deleter, bool unsafe=false>
    requires
        std::default_initializable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T> &&
        (std::is_pointer_v<T> || unsafe) // e.g. CUdeviceptr is not a pointer
struct Resource {
    T data;

    [[nodiscard]] constexpr Resource() noexcept = default;

    [[nodiscard]] constexpr Resource(T x) noexcept : data(x) {}

    [[nodiscard]] constexpr Resource(Resource&& other) noexcept
            : data(std::exchange(other.data, T{}))
    { }

    Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(data);
        data = std::exchange(other.data, T{});
        return *this;
    }

    Resource operator=(Resource other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T x) noexcept {
        if (x) {
            deleter(x);
            x = T{};
        }
    }

    Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};


template <typename T>
static T square(const T & x) {
    return x * x;
}


struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};


// [num_planes, width, height]
static std::array<int, 3> get_filter_shape(ChannelMode channels, const VSVideoInfo * vi) {
    switch (channels) {
        case ChannelMode::Y:
            return { 1, vi->width, vi->height };
        case ChannelMode::UV:
            return { 2, vi->width >> vi->format->subSamplingW, vi->height >> vi->format->subSamplingH };
        case ChannelMode::YUV:
        case ChannelMode::RGB:
            return { 3, vi->width, vi->height };
    }

    return {};
}


struct NLMeansStreamData {
    Resource<cudaStream_t, cudaStreamDestroyCustom> stream;

    Resource<cudaEvent_t, cudaEventDestroyCustom> event;

    int buffer_stride;
    int image_stride;
    Resource<uint8_t *, cudaFreeHostCustom> h_buffer;
    Resource<void *, cudaFreeCustom, true> d_src_ref;
    Resource<float *, cudaFreeCustom, true> d_buffer;
    Resource<float *, cudaFreeCustom, true> d_buffer_bwd;
    Resource<float *, cudaFreeCustom, true> d_buffer_fwd; // may be empty
    Resource<float *, cudaFreeCustom, true> d_wdst;
    Resource<float *, cudaFreeCustom, true> d_weight;
    Resource<float *, cudaFreeCustom, true> d_max_weight;
    Resource<void *, cudaFreeCustom, true> d_dst;
};


struct NLMeansData {
    VSNodeRef * node;
    int radius; // d
    int spatial_radius; // a
    int block_radius; // s
    float h2_inv_norm; // h
    ChannelMode channels;
    int wmode;
    float wref;
    VSNodeRef * ref_node; // rclip
    int device_id;

    ticket_semaphore semaphore;
    std::vector<NLMeansStreamData> stream_data;
    std::vector<int> ticket;
    std::mutex ticket_lock;
};

static void VS_CC NLMeansInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<const NLMeansData *>(*instanceData);

    auto vi = vsapi->getVideoInfo(d->node);
    vsapi->setVideoInfo(vi, 1, node);
}

static const VSFrameRef *VS_CC NLMeansGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<NLMeansData *>(*instanceData);

    if (activationReason == arInitial) {
        int start = std::max(n - d->radius, 0);
        auto vi = vsapi->getVideoInfo(d->node);
        int end = std::min(n + d->radius, vi->numFrames - 1);
        for (int i = start; i <= end; i++) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
            if (d->ref_node) {
                vsapi->requestFrameFilter(i, d->ref_node, frameCtx);
            }
        }
        return nullptr;
    } else if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    auto set_error = [vsapi, frameCtx](const char * error_message) -> std::nullptr_t {
        vsapi->setFilterError(error_message, frameCtx);
        return nullptr;
    };

    checkError(cudaSetDevice(d->device_id));

    auto vi = vsapi->getVideoInfo(d->node);

    std::vector<std::unique_ptr<const VSFrameRef, decltype(vsapi->freeFrame)>> src_frames;
    src_frames.reserve((d->ref_node ? 2 : 1) * (2 * d->radius + 1));
    for (int i = n - d->radius; i <= n + d->radius; i++) {
        src_frames.emplace_back(
            vsapi->getFrameFilter(std::clamp(i, 0, vi->numFrames - 1), d->node, frameCtx),
            vsapi->freeFrame
        );
    }
    if (d->ref_node) {
        for (int i = n - d->radius; i <= n + d->radius; i++) {
            src_frames.emplace_back(
                vsapi->getFrameFilter(std::clamp(i, 0, vi->numFrames - 1), d->ref_node, frameCtx),
                vsapi->freeFrame
            );
        }
    }

    auto & src_center_frame = src_frames[d->radius];
    auto format = vsapi->getFrameFormat(src_center_frame.get());

    const VSFrameRef * fr[] {
        (d->channels == ChannelMode::UV) ? src_center_frame.get() : nullptr,
        (d->channels == ChannelMode::Y) ? src_center_frame.get() : nullptr,
        (d->channels == ChannelMode::Y) ? src_center_frame.get() : nullptr
    };
    const int pl[] { 0, 1, 2 };
    std::unique_ptr<VSFrameRef, decltype(vsapi->freeFrame)> dst_frame {
        vsapi->newVideoFrame2(format, vi->width, vi->height, fr, pl, src_center_frame.get(), core),
        vsapi->freeFrame
    };

    d->semaphore.acquire();

    int ticket;
    {
        std::lock_guard lock { d->ticket_lock };
        ticket = d->ticket.back();
        d->ticket.pop_back();
    }

    auto & stream_data = d->stream_data[ticket];

    auto [filter_num_planes, filter_width, filter_height] = get_filter_shape(d->channels, vi);

    {
        auto h_buffer = stream_data.h_buffer.data;

        for (const auto & src_frame : src_frames) {
            for (int plane = 0; plane < 3; plane++) {
                if (fr[plane] != nullptr) {
                    continue;
                }

                auto srcp = vsapi->getReadPtr(src_frame.get(), plane);
                int pitch = vsapi->getStride(src_frame.get(), plane);

                vs_bitblt(
                    h_buffer, stream_data.image_stride * vi->format->bytesPerSample,
                    srcp, pitch,
                    filter_width * vi->format->bytesPerSample, filter_height
                );

                h_buffer += filter_height * stream_data.image_stride * vi->format->bytesPerSample;
            }
        }
    }

    {
        checkError(cudaMemcpyAsync(
            stream_data.d_src_ref,
            stream_data.h_buffer,
            (d->ref_node ? 2 : 1) * (2 * d->radius + 1) * filter_num_planes * filter_height * stream_data.image_stride * vi->format->bytesPerSample,
            cudaMemcpyHostToDevice,
            stream_data.stream
        ));

        checkError(nlmeans(
            stream_data.d_dst,
            stream_data.d_src_ref,
            stream_data.d_buffer,
            stream_data.d_buffer_bwd,
            stream_data.d_buffer_fwd,
            stream_data.d_wdst,
            stream_data.d_weight,
            stream_data.d_max_weight,
            vi->format->sampleType == stFloat,
            vi->format->bitsPerSample,
            filter_width,
            filter_height,
            stream_data.image_stride,
            stream_data.buffer_stride,
            d->radius,
            d->spatial_radius,
            d->block_radius,
            d->h2_inv_norm,
            d->channels,
            d->wmode,
            d->wref,
            d->ref_node != nullptr,
            stream_data.stream
        ));

        checkError(cudaMemcpyAsync(
            stream_data.h_buffer,
            stream_data.d_dst,
            filter_num_planes * filter_height * stream_data.image_stride * vi->format->bytesPerSample,
            cudaMemcpyDeviceToHost,
            stream_data.stream
        ));

        checkError(cudaEventRecord(stream_data.event, stream_data.stream));
        checkError(cudaEventSynchronize(stream_data.event));
    }

    {
        auto h_buffer = stream_data.h_buffer.data;

        for (int plane = 0; plane < 3; plane++) {
            if (fr[plane] != nullptr) {
                continue;
            }

            auto dstp = vsapi->getWritePtr(dst_frame.get(), plane);
            int pitch = vsapi->getStride(dst_frame.get(), plane);

            vs_bitblt(
                dstp, pitch,
                h_buffer, stream_data.image_stride * vi->format->bytesPerSample,
                filter_width * vi->format->bytesPerSample, filter_height
            );

            h_buffer += filter_height * stream_data.image_stride * vi->format->bytesPerSample;
        }
    }

    {
        std::lock_guard lock { d->ticket_lock };
        d->ticket.emplace_back(ticket);
    }
    d->semaphore.release();

    return dst_frame.release();
}

static void VS_CC NLMeansFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<const NLMeansData *>(instanceData);

    vsapi->freeNode(d->node);
    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }

    showError(cudaSetDevice(d->device_id));

    delete d;
}

static void VS_CC NLMeansCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = std::make_unique<NLMeansData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    node_freer node_freer_node { vsapi, d->node };

    auto set_error = [vsapi, out](const char * error_message) -> void {
        vsapi->setError(out, error_message);
        return ;
    };

    auto vi = vsapi->getVideoInfo(d->node);

    if (!isConstantFormat(vi) ||
        (vi->format->sampleType == stInteger && vi->format->bitsPerSample > 16) ||
        (vi->format->sampleType == stFloat && vi->format->bitsPerSample != 32)
     ) {
        return set_error("only constant format 1-16 bit integer or 32-bit float input is supported");
    }

    int error;

    d->radius = int64ToIntS(vsapi->propGetInt(in, "d", 0, &error));
    if (error) {
        d->radius = 1;
    }
    if (d->radius < 0) {
        return set_error("\"d\" must be non-negative");
    }

    d->spatial_radius = int64ToIntS(vsapi->propGetInt(in, "a", 0, &error));
    if (error) {
        d->spatial_radius = 2;
    }
    if (d->spatial_radius <= 0) {
        return set_error("\"a\" must be positive");
    }

    d->block_radius = int64ToIntS(vsapi->propGetInt(in, "s", 0, &error));
    if (error) {
        d->block_radius = 4;
    }
    if (d->block_radius < 0) {
        return set_error("\"s\" must be non-negative");
    }

    auto h = vsapi->propGetFloat(in, "h", 0, &error);
    if (error) {
        h = 1.2;
    }
    if (h <= 0.0) {
        return set_error("\"h\" must be positive");
    }
    d->h2_inv_norm = static_cast<float>(square(255) / (3 * square(2 * d->block_radius + 1) * square(h)));

    auto channels = vsapi->propGetData(in, "channels", 0, &error);
    if (error) {
        channels = "AUTO";
    }
    auto channels_len = strlen(channels);
    if (channels_len == 1 && *channels == 'Y') {
        d->channels = ChannelMode::Y;
    } else if (channels_len == 2 && strncmp(channels, "UV", 2) == 0) {
        d->channels = ChannelMode::UV;
    } else if (channels_len == 3 && strncmp(channels, "YUV", 3) == 0) {
        d->channels = ChannelMode::YUV;
    } else if (channels_len == 3 && strncmp(channels, "RGB", 3) == 0) {
        d->channels = ChannelMode::RGB;
    } else if (channels_len == 4 && strncmp(channels, "AUTO", 4) == 0) {
        if (vi->format->colorFamily == cmRGB) {
            d->channels = ChannelMode::RGB;
        } else {
            d->channels = ChannelMode::Y;
        }
    } else {
        return set_error("\"channels\" must be \"Y\", \"UV\', \"YUV\", \"RGB\" or \"AUTO\"");
    }

    if (d->channels == ChannelMode::Y) {
        if (vi->format->colorFamily != cmGray && vi->format->colorFamily != cmYUV) {
            return set_error("color family must be Gray or YUV for \"channels\" == \"Y\"");
        }
    } else if (d->channels == ChannelMode::UV) {
        if (vi->format->colorFamily != cmYUV) {
            return set_error("color family must be YUV for \"channels\" == \"UV\"");
        }
    } else if (d->channels == ChannelMode::YUV) {
        if (vi->format->colorFamily != cmYUV || vi->format->subSamplingW || vi->format->subSamplingH) {
            return set_error("color family must be YUV444 for \"channels\" == \"YUV\"");
        }
    } else if (d->channels == ChannelMode::RGB) {
        if (vi->format->colorFamily != cmRGB) {
            return set_error("color family must be RGB for \"channels\" == \"RGB\"");
        }
    }

    d->wmode = static_cast<int>(vsapi->propGetInt(in, "wmode", 0, &error));
    if (error) {
        d->wmode = 0;
    }
    if (d->wmode < 0 || d->wmode > 3) {
        return set_error("\"wmode\" must be 0, 1, 2 or 3");
    }

    d->wref = static_cast<float>(vsapi->propGetFloat(in, "wref", 0, &error));
    if (error) {
        d->wref = 1.0f;
    }

    d->ref_node = vsapi->propGetNode(in, "rclip", 0, &error);
    if (error) {
        d->ref_node = nullptr;
    }
    if (d->ref_node) {
        const auto ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (!isSameFormat(vi, ref_vi) || vi->numFrames != ref_vi->numFrames) {
            vsapi->freeNode(d->ref_node);
            return set_error("\"rclip\" must be of the same format as \"clip\"");
        }
    }

    node_freer node_freer_ref { vsapi, d->ref_node };

    d->device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        d->device_id = 0;
    }

    int num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }
    if (num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }
    d->semaphore.current.store(num_streams - 1, std::memory_order::relaxed);
    d->ticket.reserve(num_streams);
    for (int i = 0; i < num_streams; i++) {
        d->ticket.emplace_back(i);
    }

    checkError(cudaSetDevice(d->device_id));

    auto [filter_num_planes, filter_width, filter_height] = get_filter_shape(d->channels, vi);

    d->stream_data.resize(num_streams);
    for (int i = 0; i < num_streams; i++) {
        auto & stream_data = d->stream_data[i];

        checkError(cudaStreamCreateWithFlags(&stream_data.stream.data, cudaStreamNonBlocking));

        checkError(cudaEventCreateWithFlags(
            &stream_data.event.data,
            cudaEventBlockingSync | cudaEventDisableTiming
        ));

        size_t image_pitch;
        checkError(cudaMallocPitch(
            &stream_data.d_src_ref.data,
            &image_pitch,
            filter_width * vi->format->bytesPerSample,
            (d->ref_node ? 2 : 1) * (2 * d->radius + 1) * filter_num_planes * filter_height
        ));
        stream_data.image_stride = static_cast<int>(image_pitch / vi->format->bytesPerSample);

        checkError(cudaMalloc(&stream_data.d_dst.data, filter_num_planes * filter_height * image_pitch));

        size_t buffer_pitch;
        checkError(cudaMallocPitch(&stream_data.d_max_weight.data, &buffer_pitch, filter_width * sizeof(float), filter_height));
        stream_data.buffer_stride = static_cast<int>(buffer_pitch / sizeof(float));

        checkError(cudaMalloc(&stream_data.d_buffer.data, filter_height * buffer_pitch));
        checkError(cudaMalloc(&stream_data.d_buffer_bwd.data, filter_height * buffer_pitch));
        if (d->radius > 0) {
            checkError(cudaMalloc(&stream_data.d_buffer_fwd.data, filter_height * buffer_pitch));
        }
        checkError(cudaMalloc(&stream_data.d_wdst.data, filter_num_planes * filter_height * buffer_pitch));
        checkError(cudaMalloc(&stream_data.d_weight.data, filter_height * buffer_pitch));

        auto h_buffer_size = std::max(
            (d->ref_node ? 2 : 1) * (2 * d->radius + 1) * filter_num_planes * filter_height * image_pitch,
            filter_num_planes * filter_height * buffer_pitch
        );
        checkError(cudaMallocHost(&stream_data.h_buffer.data, h_buffer_size));
    }

    vsapi->createFilter(
        in, out, "NLMeans",
        NLMeansInit, NLMeansGetFrame, NLMeansFree,
        fmParallel, 0, d.release(), core
    );

    node_freer_node.release();
    node_freer_ref.release();
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc(
        "io.github.amusementclub.nlm_cuda",
        "nlm_cuda",
        "Non-local means denoise filter implemented in CUDA",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc(
        "NLMeans",
        "clip:clip;"
        "d:int:opt;"
        "a:int:opt;"
        "s:int:opt;"
        "h:float:opt;"
        "channels:data:opt;"
        "wmode:int:opt;"
        "wref:float:opt;"
        "rclip:clip:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        NLMeansCreate, nullptr, plugin
    );
}
