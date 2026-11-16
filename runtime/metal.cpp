
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace carbon {

using Shape = std::array<size_t, 3>;
using Clock = std::chrono::steady_clock;

auto pool() { return NS::TransferPtr(NS::AutoreleasePool::alloc()->init()); }

auto text(const std::string& value) {
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("Metal strings cannot contain NUL");
    auto result = NS::TransferPtr(NS::String::alloc()->init(value.c_str(), NS::UTF8StringEncoding));
    if (!result)
        throw std::invalid_argument("invalid UTF-8");
    return result;
}

std::string error(NS::Error* value) {
    return value ? value->localizedDescription()->utf8String() : "Metal operation failed";
}

MTL::Size size3(Shape shape) {
    for (auto value : shape)
        if (!value)
            throw std::invalid_argument("launch dimensions must be positive");
    return MTL::Size(shape[0], shape[1], shape[2]);
}

struct Context {
    NS::SharedPtr<MTL::Device> device;
    NS::SharedPtr<MTL::CommandQueue> queue;
    std::mutex mutex;
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> cache;

    Context() {
        auto scope = pool();
        device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        if (!device)
            throw std::runtime_error("no Metal device");
        queue = NS::TransferPtr(device->newCommandQueue());
        if (!queue)
            throw std::runtime_error("cannot create Metal queue");
    }
};

struct Allocation {
    std::shared_ptr<Context> context;
    NS::SharedPtr<MTL::Buffer> handle;
    std::atomic<size_t> flights{0};
};

struct Buffer {
    std::shared_ptr<Allocation> allocation;
    size_t offset;
    size_t size;

    bool overlaps(const Buffer& other) const {
        return allocation == other.allocation && offset < other.offset + other.size &&
               other.offset < offset + size;
    }

    void bounds(size_t start, size_t count) const {
        if (start > size || count > size - start)
            throw std::out_of_range("buffer access exceeds view");
    }

    void host_access() const {
        if (allocation->flights.load())
            throw std::runtime_error("buffer is in flight; wait before host access");
    }

    Buffer view(size_t start, size_t count) const {
        bounds(start, count);
        if (!count)
            throw std::invalid_argument("view size must be positive");
        return {allocation, offset + start, count};
    }

    void write(nb::bytes data, size_t start) {
        bounds(start, data.size());
        host_access();
        std::memcpy(static_cast<char*>(allocation->handle->contents()) + offset + start,
                    data.c_str(), data.size());
    }

    nb::bytes read() const {
        host_access();
        return nb::bytes(static_cast<char*>(allocation->handle->contents()) + offset, size);
    }
};

struct Pipeline {
    std::shared_ptr<Context> context;
    NS::SharedPtr<MTL::ComputePipelineState> handle;
    std::string entry;
    bool cache_hit;
    double compile_ms;
};

struct Dispatch {
    Pipeline pipeline;
    std::vector<Buffer> buffers;
    std::vector<bool> writes;
    std::string constants;
    MTL::Size grid;
    MTL::Size threads;
    size_t scratch_bytes;
    std::optional<Buffer> indirect;

    std::vector<std::pair<Buffer, bool>> bindings() const {
        std::vector<std::pair<Buffer, bool>> result;
        for (size_t i = 0; i < buffers.size(); ++i)
            result.emplace_back(buffers[i], writes[i]);
        if (indirect)
            result.emplace_back(*indirect, false);
        return result;
    }

    bool conflicts(const Dispatch& other) const {
        for (const auto& [mine, writing] : bindings())
            for (const auto& [theirs, written] : other.bindings())
                if ((writing || written) && mine.overlaps(theirs))
                    return true;
        return false;
    }

    void encode(MTL::ComputeCommandEncoder* encoder) const {
        encoder->setComputePipelineState(pipeline.handle.get());
        for (size_t i = 0; i < buffers.size(); ++i)
            encoder->setBuffer(buffers[i].allocation->handle.get(), buffers[i].offset, i);
        if (!constants.empty())
            encoder->setBytes(constants.data(), constants.size(), buffers.size());
        encoder->setThreadgroupMemoryLength(scratch_bytes, 0);
        if (indirect)
            encoder->dispatchThreadgroups(indirect->allocation->handle.get(), indirect->offset, threads);
        else
            encoder->dispatchThreadgroups(grid, threads);
    }
};

struct Flight {
    std::vector<std::shared_ptr<Allocation>> allocations;
    std::atomic<bool> finished{false};

    explicit Flight(const std::vector<std::vector<Dispatch>>& steps) {
        for (const auto& step : steps)
            for (const auto& dispatch : step)
                for (const auto& binding : dispatch.bindings())
                    allocations.push_back(binding.first.allocation);
        for (const auto& allocation : allocations)
            allocation->flights.fetch_add(1);
    }

    void finish() {
        if (!finished.exchange(true))
            for (const auto& allocation : allocations)
                allocation->flights.fetch_sub(1);
    }

    ~Flight() { finish(); }
};

class Completion {
  public:
    Completion(NS::SharedPtr<MTL::CommandBuffer> command, std::shared_ptr<Flight> flight)
        : command_(std::move(command)), flight_(std::move(flight)) {}

    void wait() const {
        auto scope = pool();
        command_->waitUntilCompleted();
        flight_->finish();
        if (command_->status() != MTL::CommandBufferStatusCompleted)
            throw std::runtime_error(error(command_->error()));
    }

    bool done() const {
        auto status = command_->status();
        return status == MTL::CommandBufferStatusCompleted || status == MTL::CommandBufferStatusError;
    }

    std::optional<double> gpu_ms() const {
        if (command_->status() != MTL::CommandBufferStatusCompleted)
            return std::nullopt;
        double start = command_->GPUStartTime();
        double end = command_->GPUEndTime();
        if (start <= 0 || end < start)
            return std::nullopt;
        return (end - start) * 1000.0;
    }

  private:
    NS::SharedPtr<MTL::CommandBuffer> command_;
    std::shared_ptr<Flight> flight_;
};

class Executable {
  public:
    Executable(std::shared_ptr<Context> context, std::vector<std::vector<Dispatch>> steps)
        : context_(std::move(context)), steps_(std::move(steps)) {
        if (steps_.empty())
            throw std::invalid_argument("an executable needs a dispatch");
        for (const auto& step : steps_) {
            if (step.empty())
                throw std::invalid_argument("a group needs a dispatch");
            for (size_t i = 0; i < step.size(); ++i) {
                if (step[i].pipeline.context != context_)
                    throw std::invalid_argument("dispatch belongs to another Device");
                for (size_t j = 0; j < i; ++j)
                    if (step[i].conflicts(step[j]))
                        throw std::invalid_argument("dispatches of one group must not write bytes another uses");
            }
        }
    }

    Completion submit(size_t repetitions) const {
        if (!repetitions)
            throw std::invalid_argument("repetitions must be positive");
        auto scope = pool();
        auto command = NS::RetainPtr(context_->queue->commandBuffer());
        if (!command)
            throw std::runtime_error("cannot create command buffer");
        auto flight = std::make_shared<Flight>(steps_);
        MTL::ComputeCommandEncoder* encoder = nullptr;
        bool concurrent = false;
        for (size_t repeat = 0; repeat < repetitions; ++repeat) {
            for (const auto& step : steps_) {
                bool group = step.size() > 1;
                if (!encoder || group || concurrent) {
                    if (encoder)
                        encoder->endEncoding();
                    encoder = command->computeCommandEncoder(group ? MTL::DispatchTypeConcurrent
                                                                   : MTL::DispatchTypeSerial);
                    if (!encoder)
                        throw std::runtime_error("cannot create compute encoder");
                    concurrent = group;
                }
                for (const auto& dispatch : step)
                    dispatch.encode(encoder);
            }
        }
        encoder->endEncoding();
        command->addCompletedHandler(^(MTL::CommandBuffer*) { flight->finish(); });
        command->commit();
        return Completion(std::move(command), std::move(flight));
    }

  private:
    std::shared_ptr<Context> context_;
    std::vector<std::vector<Dispatch>> steps_;
};

class Device {
  public:
    Device() : context_(std::make_shared<Context>()) {}

    std::string name() const { return context_->device->name()->utf8String(); }
    size_t max_threadgroup_memory() const { return context_->device->maxThreadgroupMemoryLength(); }
    // 5002 is MTL::GPUFamilyMetal4, absent from older metal-cpp.
    bool tensor_ops() const { return context_->device->supportsFamily(MTL::GPUFamily(5002)); }
    // MTL::GPUFamilyAppleN is 1000 + N; the first family supported from the top is the newest.
    int apple_family() const {
        for (int family = 1020; family > 1000; --family)
            if (context_->device->supportsFamily(MTL::GPUFamily(family)))
                return family - 1000;
        return 0;
    }

    Buffer buffer(size_t bytes) const {
        if (!bytes || bytes > context_->device->maxBufferLength())
            throw std::invalid_argument("buffer size exceeds device limits");
        auto scope = pool();
        auto allocation = std::make_shared<Allocation>();
        allocation->context = context_;
        allocation->handle = NS::TransferPtr(context_->device->newBuffer(
            bytes, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
        if (!allocation->handle)
            throw std::runtime_error("cannot allocate Metal buffer");
        return {std::move(allocation), 0, bytes};
    }

    Pipeline compile(const std::string& source, const std::string& entry, bool fast_math) const {
        auto start = Clock::now();
        auto scope = pool();
        auto shader = text(source);
        auto function_name = text(entry);
        std::string key = source + '\0' + entry + (fast_math ? "\1" : "\2");
        std::lock_guard lock(context_->mutex);
        auto found = context_->cache.find(key);
        bool hit = found != context_->cache.end();
        NS::SharedPtr<MTL::ComputePipelineState> pipeline;
        if (hit) {
            pipeline = found->second;
        } else {
            auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
            options->setFastMathEnabled(fast_math);
            // Metal 4 tensors and tensor ops; 262144 is MTL::LanguageVersion4_0, absent from older metal-cpp.
            if (source.find("<metal_tensor>") != std::string::npos)
                options->setLanguageVersion(MTL::LanguageVersion(262144));
            NS::Error* failure = nullptr;
            auto library = NS::TransferPtr(context_->device->newLibrary(shader.get(), options.get(), &failure));
            if (!library)
                throw std::runtime_error(error(failure));
            auto function = NS::TransferPtr(library->newFunction(function_name.get()));
            if (!function)
                throw std::invalid_argument("Metal entry not found: " + entry);
            pipeline = NS::TransferPtr(context_->device->newComputePipelineState(function.get(), &failure));
            if (!pipeline)
                throw std::runtime_error(error(failure));
            context_->cache.emplace(key, pipeline);
        }
        double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return {context_, std::move(pipeline), entry, hit, elapsed};
    }

    Dispatch dispatch(const Pipeline& pipeline, std::vector<Buffer> buffers, nb::bytes constants, Shape grid,
                      Shape threads, size_t scratch_bytes, std::optional<std::vector<bool>> writes,
                      std::optional<Buffer> indirect) const {
        if (pipeline.context != context_)
            throw std::invalid_argument("pipeline belongs to another Device");
        if (writes && writes->size() != buffers.size())
            throw std::invalid_argument("writes needs one flag per buffer");
        // Metal's limits for one dispatch: 31 buffer arguments, and 4 KiB of constant bytes set inline.
        if (buffers.size() + (constants.size() != 0) > 31 || constants.size() > 4096)
            throw std::invalid_argument("argument bindings exceed Metal limits");
        for (const auto& buffer : buffers) {
            if (buffer.allocation->context != context_)
                throw std::invalid_argument("buffer belongs to another Device");
            if (buffer.offset % 4)
                throw std::invalid_argument("binding offset must be four-byte aligned");
        }
        if (indirect && (indirect->allocation->context != context_ || indirect->offset % 4 || indirect->size < 12))
            throw std::invalid_argument("an indirect grid needs 12 four-byte aligned bytes of this Device");
        auto group = size3(threads);
        auto maximum = context_->device->maxThreadsPerThreadgroup();
        size_t total_limit = pipeline.handle->maxTotalThreadsPerThreadgroup();
        if (group.width > maximum.width || group.height > maximum.height || group.depth > maximum.depth ||
            group.width > total_limit || group.height > total_limit / group.width ||
            group.depth > total_limit / (group.width * group.height))
            throw std::invalid_argument("threadgroup exceeds device or pipeline limits");
        size_t static_bytes = pipeline.handle->staticThreadgroupMemoryLength();
        size_t limit = context_->device->maxThreadgroupMemoryLength();
        if (scratch_bytes % 16 || static_bytes > limit || scratch_bytes > limit - static_bytes)
            throw std::invalid_argument("invalid threadgroup scratch size");
        auto written = writes ? std::move(*writes) : std::vector<bool>(buffers.size(), true);
        return {pipeline, std::move(buffers), std::move(written), std::string(constants.c_str(), constants.size()),
                size3(grid), group, scratch_bytes, std::move(indirect)};
    }

    Executable prepare(std::vector<std::variant<Dispatch, std::vector<Dispatch>>> items) const {
        std::vector<std::vector<Dispatch>> steps;
        for (auto& item : items)
            if (auto* dispatch = std::get_if<Dispatch>(&item))
                steps.push_back({std::move(*dispatch)});
            else
                steps.push_back(std::move(std::get<std::vector<Dispatch>>(item)));
        return Executable(context_, std::move(steps));
    }

  private:
    std::shared_ptr<Context> context_;
};

}

NB_MODULE(_metal, module) {
    using namespace carbon;
    nb::class_<Buffer>(module, "Buffer",
                       "A byte range of a shared Metal buffer; the host may touch it only while no submitted command "
                       "buffer uses it.")
        .def_ro("size", &Buffer::size, "Bytes in the range.")
        .def_ro("offset", &Buffer::offset, "Byte offset of the range in its allocation.")
        .def("overlaps", &Buffer::overlaps, nb::arg("other"), "Whether the two ranges share bytes of one allocation.")
        .def("view", &Buffer::view, nb::arg("offset"), nb::arg("size"),
             "The `size` bytes at `offset` within this range, sharing its allocation.")
        .def("write", &Buffer::write, nb::arg("data"), nb::arg("offset") = 0,
             "Copy `data` into the range at byte `offset`; raises while the buffer is in flight.")
        .def("read", &Buffer::read, "The range's bytes; raises while the buffer is in flight.");
    nb::class_<Pipeline>(module, "Pipeline", "A compiled compute pipeline.")
        .def_ro("entry", &Pipeline::entry, "The kernel function's name.")
        .def_ro("cache_hit", &Pipeline::cache_hit, "Whether the device's pipeline cache already held it.")
        .def_ro("compile_ms", &Pipeline::compile_ms, "Milliseconds Device.compile took, cache lookup included.")
        .def_prop_ro("simd_width", [](const Pipeline& p) { return p.handle->threadExecutionWidth(); },
                     "Threads per SIMD group.")
        .def_prop_ro("max_threads", [](const Pipeline& p) { return p.handle->maxTotalThreadsPerThreadgroup(); },
                     "The most threads one threadgroup of this pipeline may have.");
    nb::class_<Dispatch>(module, "Dispatch",
                         "One dispatch, checked against the device: pipeline, bindings, constants, grid and "
                         "threadgroup size.");
    nb::class_<Executable>(module, "Executable",
                           "Dispatches encoded in order into one command buffer per submission; the dispatches of a "
                           "group may run at once.")
        .def("submit", &Executable::submit, nb::arg("repetitions") = 1,
             "Commit one command buffer that runs the dispatches `repetitions` times over, and return its "
             "Completion. Command buffers of one Device run in submission order, each seeing every write of the ones "
             "before it.");
    nb::class_<Completion>(module, "Completion", "A submitted command buffer.")
        .def("wait", &Completion::wait, nb::call_guard<nb::gil_scoped_release>(),
             "Block, without the GIL, until the command buffer completes; raises if it failed.")
        .def_prop_ro("done", &Completion::done, "Whether the command buffer has completed or failed.")
        .def_prop_ro("gpu_ms", &Completion::gpu_ms,
                     "GPU milliseconds from the command buffer's start to its end; None until it completes.");
    nb::class_<Device>(module, "Device", "The system's default Metal device, its command queue and pipeline cache.")
        .def(nb::init<>())
        .def_prop_ro("target", [](const Device&) { return "metal"; }, "The backend name, always \"metal\".")
        .def_prop_ro("name", &Device::name, "The GPU's name, such as \"Apple M4\".")
        .def_prop_ro("max_threadgroup_memory", &Device::max_threadgroup_memory,
                     "Bytes of threadgroup memory one threadgroup may use.")
        .def_prop_ro("apple_family", &Device::apple_family,
                     "The newest Apple GPU family the device supports: 9 for M3 and M4, 10 for M5; 0 if none.")
        .def_prop_ro("tensor_ops", &Device::tensor_ops, "Whether the device runs Metal 4 tensor operations.")
        .def("buffer", &Device::buffer, nb::arg("bytes"), "A new shared, hazard-tracked buffer of `bytes` bytes.")
        .def("compile", &Device::compile, nb::arg("source"), nb::arg("entry"), nb::arg("fast_math") = false,
             nb::call_guard<nb::gil_scoped_release>(),
             "The pipeline for function `entry` of Metal `source`, compiled without the GIL and cached per source, "
             "entry and fast-math setting.")
        .def("dispatch", &Device::dispatch, nb::arg("pipeline"), nb::arg("buffers"),
             nb::arg("constants") = nb::bytes("", 0), nb::arg("grid") = Shape{1, 1, 1},
             nb::arg("threads") = Shape{1, 1, 1}, nb::arg("scratch_bytes") = 0, nb::arg("writes") = nb::none(),
             nb::arg("indirect") = nb::none(),
             "A dispatch of `grid` threadgroups of `threads` threads: `buffers` bind in order, `constants` follow "
             "as one more binding, and each threadgroup gets `scratch_bytes`, a multiple of 16, of threadgroup "
             "memory. `writes` flags the buffers the dispatch may write; None flags them all. With `indirect`, the "
             "grid is instead the three u32 counts in the first 12 bytes of that range when the dispatch runs, so "
             "an earlier dispatch can write it. Raises past the device's or pipeline's limits.")
        .def("prepare", &Device::prepare, nb::arg("dispatches"),
             "An Executable of this device's `dispatches`, run in order. A sequence of dispatches among them is a "
             "group that may run at once; raises if one of a group writes bytes another binds.");
}
