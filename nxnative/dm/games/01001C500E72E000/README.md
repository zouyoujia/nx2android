# DRAGON QUEST BUILDERS 2

Hong Kong update 1.7.3 (`v262144`), KTGL / SDK 5.4.0. The game constants and
shader tables reuse the local nxnative profile. Android adds only `android.tsv`.

## Android validation

On 2026-11-04, PJD110 (Adreno 750) reached the Chinese title screen and
`PRESS A BUTTON` with a release build and the stock Qualcomm Vulkan driver.
Gameplay, save/reload and sustained progression remain unverified.

The profile loads `main`, `subsdk0` (runtime GLSLC), `subsdk1` (movie library)
and `sdk`, with a 4 GB layout and `app_pool=0xCD500000`. The complete shader set
has 2,107 shaders, including six outputs captured at startup on Apple Silicon.
The merged XCI supplies both application and patch content to extraction.

The shared core reads indexed KTGL LINKDATA members with variable chunk sizes,
allocates offline SSL contexts, and recognizes SDK 5 audio DeviceSink commands.
The Android shader pack inlines subgroup operations in helper functions to
avoid a Qualcomm compiler crash; only ten shaders require normalization.

No bundled Vulkan driver or diagnostic environment flags are needed for the
validated title screen. See [the port architecture](../../../../docs/ARCHITECTURE.md).
