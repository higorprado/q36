/* No built-in q36 streaming hotlist is enabled by default.
 * The ds4-imatrix-derived profile reduced misses but slowed the measured
 * 4k cold prefill/decode runs, so keep hotlist seeding opt-in through
 * Q36_VK_STREAMING_EXPERT_HOTLIST until a q36 profile wins benchmarks. */
static const uint16_t q36_default_streaming_hotlist[][2] = {
    {0, 0},
};
static const uint32_t q36_default_streaming_hotlist_count = 0;
