// Shared SIFT query scanner.  Integer L2 is exact: 128 * 255^2 < 2^24.
// Compile: c++ -O3 -std=c++17 -march=native -shared -fPIC uint8_scan.cpp -o uint8_scan.so
// No OpenMP: this kernel scans one query on one calling thread.
#include <cstddef>
#include <cstdint>

extern "C" int nora_scan_u8_128(const uint8_t* base, uint64_t base_n,
                               const uint32_t* ids, uint64_t n,
                               const uint8_t* query, float* distances) {
    for (uint64_t i = 0; i < n; ++i) {
        if (i + 16 < n && ids[i + 16] < base_n) {
            const uint8_t* next = base + uint64_t(ids[i + 16]) * 128;
            __builtin_prefetch(next, 0, 1);
            __builtin_prefetch(next + 64, 0, 1);
        }
        if (ids[i] >= base_n) {
            distances[i] = -1.f;  // Python fills small float32 insertion overlay.
            continue;
        }
        const uint8_t* row = base + uint64_t(ids[i]) * 128;
        uint32_t sum = 0;
        for (unsigned j = 0; j < 128; ++j) {
            int delta = int(row[j]) - int(query[j]);
            sum += uint32_t(delta * delta);
        }
        distances[i] = float(sum);
    }
    return 0;
}
