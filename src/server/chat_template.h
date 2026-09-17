#pragma once
#include <string>
#include <vector>

#include "chat.h"

namespace si {

// Render a Jinja chat template (the `tokenizer.chat_template` string from the
// GGUF) for a text-only conversation, using the vendored minja engine
// (third_party/minja).
//
// Returns false when the template cannot be parsed/rendered (or raises through
// raise_exception), leaving `out` unspecified so the caller can fall back to
// the built-in renderer.
bool render_chat_template(const std::string & tmpl, const std::vector<chat_msg> & msgs, bool add_generation_prompt,
                          bool enable_thinking, std::string & out);

} // namespace si
