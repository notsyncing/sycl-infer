#include <cstdio>
#include <string>
#include <vector>

#include "gguf.h"
#include "tokenizer.h"

int main(int argc, char ** argv) {
    const char * model = argc > 1 ? argv[1] : "/home/sfc/临时/Qwen3.5-0.8B-Q4_K_M.gguf";
    gguf_file f;
    f.load(model);
    tokenizer tk;
    tk.load(f);
    printf("vocab=%d eos=%d eot=%d im_start=%d im_end=%d think=%d endthink=%d\n",
        tk.n_vocab, tk.eos_id, tk.eot_id, tk.im_start_id, tk.im_end_id, tk.think_id, tk.endthink_id);

    std::vector<std::string> tests = {
        "Hello world!",
        "你好，世界！",
        "The quick brown fox jumps over the lazy dog 1234567890.",
        "def foo(x):\n    return x * 2  # comment\n",
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n",
        "  multiple   spaces\tand\ttabs\r\nnewline",
        "emoji 🚀 test 🎉",
        "Ġweird Ġtokens",
    };
    for (const auto & t : tests) {
        auto ids = tk.encode(t);
        printf("\n[%s]\n  n=%zu ids=", t.c_str(), ids.size());
        for (int id : ids) printf("%d ", id);
        printf("\n  roundtrip: %s\n", tk.decode(ids) == t ? "OK" : "MISMATCH");
        if (tk.decode(ids) != t) {
            printf("  got: [%s]\n", tk.decode(ids).c_str());
        }
    }
    return 0;
}
