#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class ExpertSource;

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    void attach(RemoteExperts& remote); // before RemoteExperts::open
    bool init(std::string& err);        // after all helper caches are open
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

    // KURAI (STRATA_REMOTE_ASYNC=1): the helper no longer blocks the host in finish().  Its rows are scattered
    // into the mapped rows by its own device (or, with --remote-expert-opt, its reduced sum is accumulated on
    // its own device), it posts a done counter in mapped memory, and the window's graph waits on that counter
    // (wait_done) before it reads the rows/sum.  The helper's latency then overlaps the CPU pool instead of
    // sitting on the serving thread's critical path.  Counters are reset per window (reset_done) and saturated
    // on the timeout paths (release_done), like the verifier's own flags.
    bool async_enabled() const { return async_; }
    /// finish() without the wait on the --remote-expert-opt path: the mask is set here, the mark is posted
    /// through the copy engine (a 4-byte D2H from the helper's VRAM word), and the sums are the helper's own
    /// D2H'd h_out_ rows (finish() no longer waits for them).
    bool finish_async_opt(RemoteExperts& remote, std::string& err);
    /// The graph's side: wait until this helper has reached `ring` (a no-op when async is off).
    void wait_done(void* stream, uint32_t ring) const;
    /// KURAI (STRATA_REMOTE_FLUSH=end|sync|both): submit the helper stream's tail once per window, before the
    /// verifier blocks on its own graph.  WDDM/HIP defers submissions; without this the last commands of a
    /// window - the last doorbell mark included - never run (the sync path covered this with its per-layer
    /// cudaStreamQuery spin; the async path must ask explicitly).
    void flush_helpers();
    /// Stall diagnostics (STRATA_REMOTE_ASYNC debug): each helper's doorbell value and stream state.
    std::string helpers_diag() const;
    void reset_done();
    void release_done();
    /// Non-blocking: did a helper stream fail since the last poll?
    bool poll_helpers(std::string& err);
    /// The ring (layer*G+grp+1) of the dispatch the serving loop is about to make.
    static void set_dispatch_ring(uint32_t ring);

private:
    friend class RemoteExperts;
    static size_t metadata_bytes();
    bool active() const { return weights_ != nullptr; }
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);
    /// The dispatch in flight has an armed ring: this thread is inside a verify window's pool call.
    bool async_ready() const;
    /// Post the done counter after the host's own accumulate (the --remote-expert-opt path keeps its sync
    /// accumulate but must still satisfy the graph's wait).
    void mark_done_now(const RemoteExperts& remote);

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    std::vector<Peer> peers_;
    float* h_sum_ = nullptr;
    float* m_sum_ = nullptr;
    int32_t* h_mask_ = nullptr;
    int32_t* m_mask_ = nullptr;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
    // async helper merge (KURAI): one mapped done counter per helper (h_done_ is the host array, the peers
    // hold each device's view of it).  Every value the helper posts travels through the COPY ENGINE (a 4-byte
    // D2H from d_done): its kernels never write host-mapped memory - on this HIP/Windows build those writes
    // do not reach the primary or the wait never sees them (the first window stalls, issue #29).
    bool async_ = false;
    uint32_t* h_done_ = nullptr;
    struct PeerAsync {
        uint32_t* d_done = nullptr;        ///< this helper's VRAM word the mark kernel writes
        uint32_t* done_primary = nullptr;  ///< the primary's view of the mapped counter (the graph's wait)
        float* out_primary = nullptr;      ///< the primary's view of the helper's summed rows (h_out_)
    };
    std::vector<PeerAsync> async_peers_;
};
} // namespace strata::core
