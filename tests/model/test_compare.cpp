// compare CPU reference stages against llama.cpp dumped tensors
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"

#include "cpu_ref.h"

using namespace si;

struct dump {
    std::vector<int> shape;
    std::vector<float> data;
};

static dump load_dump(const std::string & path) {
    dump d;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return d;
    int32_t hdr[5];
    if (fread(hdr, 4, 5, f) != 5) { fclose(f); return d; }
    size_t n = (size_t) hdr[1] * hdr[2] * hdr[3] * hdr[4];
    d.shape = {hdr[1], hdr[2], hdr[3], hdr[4]};
    d.data.resize(n);
    size_t got = fread(d.data.data(), 4, n, f);
    fclose(f);
    if (got != n) d.data.clear();
    return d;
}

static bool verbose = false;
int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    verbose = getenv("VERBOSE") != nullptr;
    std::string dumpdir = argc > 2 ? argv[2] : "/tmp/kilo/refdump";
    std::vector<int> toks;
    for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    if (toks.empty())
        toks = {248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198};

    model m;
    m.load(model_path);
    cpu_ref ref(m, 128);
    ref.record = true;
    ref.forward(toks);

    int n_stages = 0;
    for (auto & s : ref.snaps) {
        std::string path = dumpdir + "/" + s.name + ".bin";
        dump d = load_dump(path);
        if (d.data.empty()) { printf("%-28s (no dump)\n", s.name.c_str()); continue; }
        double maxd = 0, maxv = 0, sumd = 0;
        size_t n = std::min(d.data.size(), s.data.size());
        for (size_t i = 0; i < n; i++) {
            const double diff = std::fabs((double) d.data[i] - s.data[i]);
            maxd = std::max(maxd, diff);
            maxv = std::max(maxv, (double) std::fabs(d.data[i]));
            sumd += diff;
        }
        const bool ok = maxd < 1e-2 * std::max(1.0, maxv);
        printf("%-28s n=%-7zu max|ref|=%.4f max|diff|=%.6f mean|diff|=%.6f %s\n",
               s.name.c_str(), n, maxv, maxd, sumd / n, ok ? "OK" : "MISMATCH");
        static int qcount = 0;
        if (s.name == "Qcur_normed-3" && qcount < 2) {
            printf("    [detail %d] mine: ", qcount);
            for (int i : {0, 1, 2, 8, 16, 31, 32, 33, 40, 63, 64, 65, 100}) printf("%.4f ", s.data[i]);
            printf("\n    [detail %d] ref : ", qcount);
            for (int i : {0, 1, 2, 8, 16, 31, 32, 33, 40, 63, 64, 65, 100}) printf("%.4f ", d.data[i]);
            printf("\n");
            qcount++;
        }
        if (!ok && verbose) {
            int shown = 0;
            for (size_t i = 0; i < n && shown < 8; i++) {
                double diff = std::fabs((double) d.data[i] - s.data[i]);
                if (diff > 0.5 * maxd) {
                    printf("    idx %zu: ref=%.6f mine=%.6f diff=%.6f\n", i, d.data[i], s.data[i], diff);
                    shown++;
                }
            }
        }
        n_stages++;
    }
    printf("%d stages compared\n", n_stages);
    return 0;
}
