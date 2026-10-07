#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <atomic>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int K = 10, MAXT = strata::kernels::cpu::MAXT, CAP = MAXT * K;
struct ReduceMeta { int32_t row[CAP]; float weight[CAP]; };

bool check(cudaError_t status, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string("remote expert decode optimization: ") + cudaGetErrorString(status);
    return false;
}

__global__ void reduce_experts(const float* parts, const ReduceMeta* meta, float* sum) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float v = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int i = t * K + j, r = meta->row[i];
        if (r >= 0) v += parts[(int64_t) r * H + c] * meta->weight[i];
    }
    sum[(int64_t) t * H + c] = v;
}

// Like the upstream mapped-row copy, also skip rows returned in the remote sum.
__global__ void copy_cpu_rows(float4* dst, const volatile float4* src, const int32_t* mask,
                              const int32_t* hit_rows, const int32_t* count) {
    const int row = blockIdx.x;
    __shared__ int skip;
    if (threadIdx.x == 0) {
        int s = mask[row];
        for (int i = 0; i < *count; ++i) s |= hit_rows[i] == row;
        skip = s;
    }
    __syncthreads();
    const int64_t start = (int64_t) row * (H / 4);
    for (int i = threadIdx.x; i < H / 4; i += blockDim.x)
        dst[start + i] = skip ? make_float4(0, 0, 0, 0) : const_cast<const float4*>(src)[start + i];
}

// KURAI async helper merge: the device-side half of the doorbell.  It writes a VRAM word only - never host
// memory: the copy engine carries the value to the mapped counter the window's graph spins on.
__global__ void set_word(uint32_t* p, uint32_t v) { *p = v; }

// STRATA_REMOTE_FLUSH (KURAI 07/10): WDDM/HIP defer submissions - nothing runs until something queries or syncs
// the stream (measured with flushbench: a window tail sat at 37/48 until one query).  The sync decode path spins
// on cudaStreamQuery after every layer, which is why it never saw the tail stall; the async path must ask for a
// flush itself.  none|layer|end|sync|both (default both): layer = a query after each mark, end = one query at
// the window boundary (flush_helpers), sync = a stream sync there.
enum class FlushMode { none, layer, end, sync, both };
FlushMode flush_mode() {
    static const FlushMode mode = [] {
        const char* v = std::getenv("STRATA_REMOTE_FLUSH");
        if (v == nullptr) return FlushMode::both;
        if (std::strcmp(v, "none") == 0) return FlushMode::none;
        if (std::strcmp(v, "layer") == 0) return FlushMode::layer;
        if (std::strcmp(v, "end") == 0) return FlushMode::end;
        if (std::strcmp(v, "sync") == 0) return FlushMode::sync;
        return FlushMode::both;
    }();
    return mode;
}
bool flush_per_layer() {
    const FlushMode m = flush_mode();
    return m == FlushMode::layer || m == FlushMode::both;
}
bool flush_at_end() {
    const FlushMode m = flush_mode();
    return m == FlushMode::end || m == FlushMode::sync || m == FlushMode::both;
}
} // namespace

// The ring (layer*G+grp+1) of the dispatch in flight on this thread; 0 = async off for this dispatch (the
// prompt path and every non-verify caller keep the blocking finish: they read the mapped rows directly).
thread_local uint32_t t_ring = 0;

void RemoteExpertOpt::attach(RemoteExperts& remote) {
    remote.remote_opt_ = this;
    peers_.push_back({&remote});
}

bool RemoteExpertOpt::init(std::string& err) {
    void* mapped = nullptr;
    if (!check(cudaHostAlloc((void**) &h_sum_, MAXT * (H + K) * sizeof(float),
                             cudaHostAllocPortable | cudaHostAllocMapped), err) ||
        !check(cudaHostGetDevicePointer(&mapped, h_sum_, 0), err)) return false;
    m_sum_ = (float*) mapped;
    h_mask_ = reinterpret_cast<int32_t*>(h_sum_ + MAXT * H);
    m_mask_ = reinterpret_cast<int32_t*>(m_sum_ + MAXT * H);
    std::memset(h_sum_, 0, MAXT * (H + K) * sizeof(float));   // the serve path never accumulates: a zero sum is what its combine adds
    // KURAI async helper merge: one mapped done counter per helper, plus each side's view of it.
    static const bool async = [] {
        const char* v = std::getenv("STRATA_REMOTE_ASYNC");
        return v != nullptr && *v != '0';
    }();
    async_ = async && !peers_.empty();
    if (async_) {
        if (!check(cudaHostAlloc((void**) &h_done_, peers_.size() * sizeof(uint32_t),
                                 cudaHostAllocPortable | cudaHostAllocMapped), err)) return false;
        std::memset(h_done_, 0, peers_.size() * sizeof(uint32_t));
        async_peers_.resize(peers_.size());
        {
            const OnDevice on(primary_device());
            void* base = nullptr;
            if (!check(cudaHostGetDevicePointer(&base, h_done_, 0), err)) return false;
            for (size_t i = 0; i < peers_.size(); ++i) {
                async_peers_[i].done_primary = reinterpret_cast<uint32_t*>(base) + i;
                void* out = nullptr;
                if (!check(cudaHostGetDevicePointer(&out, peers_[i].remote->h_out_, 0), err)) return false;
                async_peers_[i].out_primary = (float*) out;
            }
        }
    }
    for (size_t i = 0; i < peers_.size(); ++i) {
        auto& p = peers_[i];
        const OnDevice on(p.remote->device_);
        if (!check(cudaMalloc((void**) &p.sum, MAXT * H * sizeof(float)), err)) return false;
        if (async_ && !check(cudaMalloc((void**) &async_peers_[i].d_done, sizeof(uint32_t)), err)) return false;
    }
    return true;
}

RemoteExpertOpt::~RemoteExpertOpt() {
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (p.remote->stream_) cudaStreamSynchronize(p.remote->stream_);
        if (p.sum) cudaFree(p.sum);
        p.remote->remote_opt_ = nullptr;
    }
    if (h_sum_) cudaFreeHost(h_sum_);
}

size_t RemoteExpertOpt::metadata_bytes() { return sizeof(ReduceMeta); }

void RemoteExpertOpt::begin(const float* weights, int token_begin, int tokens) {
    weights_ = weights;
    token_begin_ = token_begin;
    tokens_ = tokens;
    std::memset(h_sum_ + token_begin * H, 0, tokens * H * sizeof(float));
    std::memset(h_mask_ + token_begin * K, 0, tokens * K * sizeof(int32_t));
}

void RemoteExpertOpt::prepare(const RemoteExperts& remote, void* metadata) const {
    auto* m = (ReduceMeta*) metadata;
    std::fill(m->row, m->row + tokens_ * K, -1);
    for (size_t i = 0; i < remote.original_row_.size(); ++i)
        m->row[remote.original_row_[i]] = (int32_t) i;
    std::memcpy(m->weight, weights_, tokens_ * K * sizeof(float));
}

bool RemoteExpertOpt::reduce(RemoteExperts& remote, const void* metadata, std::string& err) {
    const auto& p = *std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    reduce_experts<<<dim3((H + 255) / 256, tokens_), 256, 0, remote.stream_>>>(
        remote.d_out_, (const ReduceMeta*) metadata, p.sum);
    // The summed rows go to the mapped h_out_ through the copy engine (the graph's combine reads them there
    // behind this dispatch's done counter).
    return check(cudaMemcpyAsync(remote.h_out_, p.sum, tokens_ * H * sizeof(float),
                                  cudaMemcpyDeviceToHost, remote.stream_), err);
}

void RemoteExpertOpt::accumulate(const RemoteExperts& remote) {
    for (int64_t i = 0; i < tokens_ * H; ++i) h_sum_[token_begin_ * H + i] += remote.h_out_[i];
    for (int i = 0; i < tokens_ * K; ++i)
        h_mask_[token_begin_ * K + i] |= remote.owned_[(size_t) i] != 0;
}

bool RemoteExpertOpt::finish_async_opt(RemoteExperts& remote, std::string& err) {
    const auto it = std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    if (it == peers_.end()) { err = "remote expert decode optimization: finish_async_opt on an unknown helper"; return false; }
    const size_t idx = (size_t) (it - peers_.begin());
    // The mask is the host's half of the accumulation and does not depend on the helper's results: the graph's
    // copy_rows skips these rows as soon as the CPU flag is up.
    for (int i = 0; i < tokens_ * K; ++i) h_mask_[token_begin_ * K + i] |= remote.owned_[(size_t) i] != 0;
    // The mark travels through the COPY ENGINE: set_word writes the helper's VRAM word and the 4-byte D2H
    // copies it into the mapped counter the graph spins on (the same transfer the result staging always used).
    set_word<<<1, 1, 0, remote.stream_>>>(async_peers_[idx].d_done, t_ring);
    static const bool trace = [] { const char* v = std::getenv("STRATA_ASYNC_TRACE"); return v != nullptr && *v != '0'; }();
    if (trace) std::fprintf(stderr, "[kurai-async] mark peer=%zu ring=%u\n", idx, t_ring);
    if (!check(cudaMemcpyAsync(h_done_ + idx, async_peers_[idx].d_done, sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, remote.stream_), err)) return false;
    // KURAI: the tail of this window's commands is not submitted without a flush; the per-layer mode asks for
    // one after every mark, so the doorbell flows while the host and the graph run ahead.
    if (flush_per_layer()) (void) cudaStreamQuery(remote.stream_);
    return true;
}

void RemoteExpertOpt::mark_done_now(const RemoteExperts& remote) {
    if (t_ring == 0) return;   // not armed: no graph is waiting, and a 0 would clobber the counter
    const auto it = std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    if (it == peers_.end()) return;
    const size_t idx = (size_t) (it - peers_.begin());
    set_word<<<1, 1, 0, remote.stream_>>>(async_peers_[idx].d_done, t_ring);
    (void) cudaMemcpyAsync(h_done_ + idx, async_peers_[idx].d_done, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, remote.stream_);
    if (flush_per_layer()) (void) cudaStreamQuery(remote.stream_);
}

void RemoteExpertOpt::wait_done(void* stream, uint32_t ring) const {
    if (!async_) return;
    for (const PeerAsync& ap : async_peers_)
        if (ap.done_primary != nullptr)
            strata::kernels::wait_flag_ge(ap.done_primary, ring, stream);
}

std::string RemoteExpertOpt::helpers_diag() const {
    if (!async_ || h_done_ == nullptr) return std::string();
    std::string out;
    for (size_t i = 0; i < peers_.size(); ++i) {
        const OnDevice on(peers_[i].remote->device_);
        const cudaError_t q = cudaStreamQuery(peers_[i].remote->stream_);
        char buf[192];
        std::snprintf(buf, sizeof buf, "  helper%zu doorbell=%u stream=%s\n", i,
                      *(const volatile uint32_t*) (h_done_ + i),
                      q == cudaSuccess ? "idle" : (q == cudaErrorNotReady ? "busy" : cudaGetErrorString(q)));
        out += buf;
    }
    return out;
}

void RemoteExpertOpt::reset_done() {
    if (h_done_ == nullptr) return;
    for (size_t i = 0; i < peers_.size(); ++i) *(volatile uint32_t*) (h_done_ + i) = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
}

void RemoteExpertOpt::release_done() {
    if (h_done_ == nullptr) return;
    for (size_t i = 0; i < peers_.size(); ++i) *(volatile uint32_t*) (h_done_ + i) = 0xFFFFFFFFu;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
}

bool RemoteExpertOpt::poll_helpers(std::string& err) {
    if (!async_) return true;
    for (const Peer& p : peers_) {
        const OnDevice on(p.remote->device_);
        const cudaError_t e = cudaStreamQuery(p.remote->stream_);
        if (e == cudaSuccess || e == cudaErrorNotReady) continue;
        err = std::string("remote expert helper (CUDA") + std::to_string(p.remote->device_) + "): " +
              cudaGetErrorString(e);
        return false;
    }
    return true;
}

// KURAI: the window boundary's flush (STRATA_REMOTE_FLUSH=end|sync|both).  The verifier calls this right before
// it blocks on its own graph, when every dispatch of the window has been enqueued: whatever the driver still
// held back is submitted here ("end": one query per helper; "sync": a full wait - only the last layers' tail).
void RemoteExpertOpt::flush_helpers() {
    if (!async_ || !flush_at_end()) return;
    const bool sync = flush_mode() == FlushMode::sync;
    for (const Peer& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (sync) (void) cudaStreamSynchronize(p.remote->stream_);
        else (void) cudaStreamQuery(p.remote->stream_);
    }
}

void RemoteExpertOpt::set_dispatch_ring(uint32_t ring) { t_ring = ring; }

bool RemoteExpertOpt::async_ready() const { return async_ && t_ring != 0; }

void RemoteExpertOpt::copy_rows(float* dst, const float* src, int token_begin, int tokens,
                         const int32_t* hit_rows, const int32_t* count, void* stream) const {
    copy_cpu_rows<<<tokens * K, 128, 0, (cudaStream_t) stream>>>(
        (float4*) dst, (const volatile float4*) src, m_mask_ + token_begin * K, hit_rows, count);
}

void RemoteExpertOpt::combine(float* dst, float* scratch, int token_begin, int tokens,
                       const uint32_t* skip, uint32_t ring, void* stream) const {
    if (async_) {
        // KURAI async: the sums are the helpers' own D2H'd rows (h_out_), not the host-accumulated m_sum_.
        // The graph's done-wait (wait_done) gates this read, so nothing waited for them on the host.
        for (const PeerAsync& ap : async_peers_) {
            if (ap.out_primary == nullptr) continue;
            const float* sum = ap.out_primary;
            if (skip) {
                // An all-primary device-planned group bypasses the host: ignore its stale sum.
                strata::kernels::copy_or_zero_from_mapped(scratch, sum, tokens * H, skip, ring, stream);
                sum = scratch;
            }
            strata::kernels::add_inplace(dst, sum, tokens * H, stream);
        }
        return;
    }
    const float* sum = m_sum_ + token_begin * H;
    if (skip) {
        // An all-primary device-planned group bypasses the host: ignore its stale sum.
        strata::kernels::copy_or_zero_from_mapped(scratch, sum, tokens * H, skip, ring, stream);
        sum = scratch;
    }
    strata::kernels::add_inplace(dst, sum, tokens * H, stream);
}

bool RemoteExpertOpt::owns(int64_t layer, int32_t expert) const {
    for (const auto& p : peers_)
        if (p.remote->cache_.slot_of(layer, expert) >= 0) return true;
    return false;
}

bool RemoteExpertOpt::adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                     const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
                     ExpertSource& source) {
    if (max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> resident(primary.size(), 0);
    for (size_t i = 0; i < primary.size(); ++i) resident[i] = primary[i] >= 0;
    for (const auto& [i, slot] : pending) resident[(size_t) i] = 1;
    for (const auto& p : peers_) {
        const auto& r = *p.remote;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l)
            for (int32_t e = 0; e < r.n_expert_; ++e)
                if (r.cache_.slot_of(l, e) >= 0) resident[(size_t) l * r.n_expert_ + e] = 1;
    }
    struct Swap { float gain; int32_t layer, in, out; };
    std::string err;
    for (auto& p : peers_) {
        auto& r = *p.remote;
        std::vector<Swap> swaps;
        std::vector<std::pair<float, int32_t>> candidates, victims;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l) {
            candidates.clear(); victims.clear();
            const size_t base = (size_t) l * r.n_expert_;
            float primary_cold = 1.0e30f;
            for (int32_t e = 0; e < r.n_expert_; ++e) {
                const float u = usage[base + e];
                if (primary[base + e] >= 0) primary_cold = std::min(primary_cold, u);
                if (r.cache_.slot_of(l, e) >= 0) victims.emplace_back(u, e);
                else if (!resident[base + e] && u >= 2.0f) candidates.emplace_back(u, e);
            }
            const size_t n = std::min(candidates.size(), victims.size());
            if (!n) continue;
            std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::partial_sort(victims.begin(), victims.begin() + n, victims.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < n; ++i) {
                if (victims[i].first > primary_cold) break;
                if (candidates[i].first < victims[i].first + 1.5f) break;
                swaps.push_back({candidates[i].first - victims[i].first, l, candidates[i].second, victims[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
        if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
        if (swaps.empty()) continue;
        const OnDevice on(r.device_);
        for (const auto& s : swaps)
            if (!r.cache_.fill_slot(r.cache_.slot_of(s.layer, s.out), source.blob(s.layer, s.in),
                                    r.stream_, err, (int64_t) lay.blob_bytes(s.layer))) return false;
        if (!check(cudaStreamSynchronize(r.stream_), err)) return false;
        for (const auto& s : swaps) {
            r.cache_.replace(s.layer, s.out, s.in);
            const size_t base = (size_t) s.layer * r.n_expert_;
            resident[base + s.out] = 0;
            resident[base + s.in] = 1;
        }
    }
    return true;
}
} // namespace strata::core
