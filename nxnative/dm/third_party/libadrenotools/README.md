### Adreno Tools 
A library for applying rootless Adreno GPU driver modifications/replacements. Currently supports loading custom GPU drivers such as [turnip](https://docs.mesa3d.org/android.html#building-using-the-android-ndk),  enabling BCn textures and redirecting file operations to allow accessing shader dumps and modification of the [driver config file](https://gist.github.com/bylaws/04130932e2634d1c6a2a9729e3940d60) without root.

#### Documentation
API is documented in the `include/adrenotools` headers.

#### Support
Android 9+
Arm64

Please create an issue if support for anything else is desired.

### FAQ

#### Is there an example project?

There is a simple bare-bones project [AdrenoToolsTest](https://github.com/darksylinc/AdrenoToolsTest) demonstrating how to get libadrenotools working.

#### How do I use this to update the drivers on my phone? Where's the apk?

You don't. This library is **not** for installing into Android and is **not** for end users.
This library is aimed at other developers.

Each individual app must explicitly make use of libadrenotools in order to load custom drivers into an app / game.

#### How do I use this library to make \<favourite game\> use newer drivers?

See previous question. It's up to the game developer to add support & use this library.

You could contact them to so they add support for it; but that's out of our power.

### Local nx2android changes

`ADRENOTOOLS_DRIVER_KGSL_PERF` (`1 << 3`) optionally loads
`dm/android/driver/kgsl_power_hook.c` before the Vulkan driver in its namespace.
The default flags and existing custom-driver path are unchanged. This feature can
load the original system Vulkan driver without a replacement driver binary. Its
escaped regular namespace avoids inheriting the vendor namespace's existing
global libraries; explicit platform-library links keep private driver dependencies
local. Namespace or hook failure retains the ordinary driver without a performance
request. Driver libraries that bypass `ioctl` interposition may remain unobserved.

The hook adds `KGSL_CONTEXT_PWR_CONSTRAINT` only to confirmed KGSL context creates,
then requests `KGSL_CONSTRAINT_PWR_MAX` using the returned context ID on the same
driver-owned fd. Unsupported added flags retry the original create once on
`EINVAL`; an optional property failure preserves the successful context and
original result/errno. No global `KGSL_PROP_PWRCTRL`, CPU, governor, slumber or
thermal setting is changed. Normal GPU work activates/renews a supported request;
context destruction and kernel constraint expiry govern cleanup. Pause is not a
promise of an immediate frequency drop.

The feature flag is the opt-in; absent `DM_GPU_PERF`, it requests the constraint.
`DM_GPU_PERF=2` selects an observation control using the identical loader/hook path
while leaving context flags and properties unchanged. `DM_GPU_PERF=1` requests MAX.
`adrenotools_get_kgsl_perf_stats` exposes process-lifetime observed-create, flagged,
original-flags-fallback, accepted and refused counters. Neither a loaded library
nor an accepted property proves hardware activation: verify coverage, activation,
frequency, expiry and lifecycle on the target device.
