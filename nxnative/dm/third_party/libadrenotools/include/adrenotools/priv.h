// SPDX-License-Identifier: BSD-2-Clause
// Copyright © 2021 Billy Laws

#pragma once

#include <stdint.h>

/**
 * @brief Bitfield enum of additional driver features that can be used with adrenotools_open_libvulkan
 */
enum {
    ADRENOTOOLS_DRIVER_CUSTOM = 1 << 0,
    ADRENOTOOLS_DRIVER_FILE_REDIRECT = 1 << 1,
    ADRENOTOOLS_DRIVER_GPU_MAPPING_IMPORT = 1 << 2,
    ADRENOTOOLS_DRIVER_KGSL_PERF = 1 << 3, //!< Opt-in owned-context request; also works with the system driver.
};

#define ADRENOTOOLS_GPU_MAPPING_SUCCEEDED_MAGIC 0xDEADBEEF

/**
 * @brief A replacement GPU memory mapping for use with ADRENOTOOLS_DRIVER_GPU_MAPPING_IMPORT
 */
struct adrenotools_gpu_mapping {
    void *host_ptr;
    uint64_t gpu_addr; //!< The GPU address of the mapping to import, if mapping import/init succeeds this will be set to ADRENOTOOLS_GPU_MAPPING_SUCCEEDED_MAGIC
    uint64_t size;
    uint64_t flags;
};

/** Process-lifetime diagnostics for the opt-in context hook; accepted is not hardware activation. */
struct adrenotools_kgsl_perf_stats {
    uint64_t intercepted; //!< Real KGSL context creates seen by the hook.
    uint64_t flagged; //!< Successful creates carrying KGSL_CONTEXT_PWR_CONSTRAINT.
    uint64_t fallback; //!< EINVAL retries with the driver's original flags (no new constraint).
    uint64_t accepted; //!< Successful per-context SETPROPERTY requests.
    uint64_t refused; //!< Failed optional SETPROPERTY requests; contexts remain usable.
};
