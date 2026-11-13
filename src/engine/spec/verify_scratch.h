#pragma once
#include <cassert>

#include <sycl/sycl.hpp>

#include "kernels.h"

namespace si {

// The verify cycle's argmax staging.  This is the *only* device scratch the
// two drafters share: each allocates it lazily on first use (so a run that
// never verifies pays nothing) and both write the verify rows' argmax into
// it before copying back to h_argmax.
//
// Both buffers are created together, deliberately.  MTP's candidate readout
// (PF_MTP_CAND) additionally needs the value side, and it decides that from
// its own mtp_cand_cap_ - so if the value side could exist without the id
// side, a drafter that ran first and allocated only the ids would leave MTP
// writing through a null pointer.  That is not hypothetical: the two drafter
// sites each had their own `if (!d_argmax)` block, and only MTP's allocated
// both.  test_spec never caught it because it runs one drafter per process.
struct verify_scratch {
    int32_t * d_argmax = nullptr; // [kMaxB] the verify rows' argmax ids
    float * d_argval = nullptr;   // [kMaxB] ... and their values
    void ensure(sycl::queue & q) {
        if (!d_argmax) {
            d_argmax = sycl::malloc_device<int32_t>(kMaxB, q);
            d_argval = sycl::malloc_device<float>(kMaxB, q);
        }
        // the invariant this type exists to keep, asserted rather than assumed
        assert(d_argmax != nullptr && d_argval != nullptr);
    }
};

} // namespace si
