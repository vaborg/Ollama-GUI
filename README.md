## Features

* **Multi-Provider Support**: Seamlessly switch between Ollama, OpenAI, Anthropic, and Google Gemini. Conversation history is stored in a canonical format and translated on-the-fly for each vendor's API requirements (e.g., strict user/assistant alternation for Anthropic).
* **Thread-Safe Streaming**: Network requests run on a dedicated worker thread via `libcurl` using Server-Sent Events (SSE) or NDJSON, updating the GTK4 UI strictly on the main loop.
* **Reasoning & Thinking Support**: Displays model "thoughts" (from DeepSeek/OpenAI `reasoning_content`, Anthropic `thinking_delta`, Gemini `thought` parts, and Ollama `thinking`) inside a collapsible **`+ Thinking`** section above the main reply.
* **Auto-Adapt (Quirk Learning)**: If a model rejects a specific parameter with an HTTP 400/422 error (e.g., `o1`/`o3` rejecting `temperature`, or Claude rejecting simultaneous `temperature` and `top_p`), the client automatically learns the restriction, saves it per model, and immediately retries the request without the offending parameter.
* **Multimodal & Text Attachments**: 
  * **Images** (up to 100 MiB): PNG, JPEG, GIF, and WebP are sent as-is in base64; other image formats (BMP, TIFF, etc.) are automatically re-encoded to PNG via `GdkTexture`.
  * **Text Files** (up to 64 MiB): Inlined directly into the prompt wrapped in `<file name="...">...</file>` tags.
* **Code Block Extraction**: Parses CommonMark fenced code blocks from the latest assistant reply. Saves a single snippet via a file dialog (inferring extensions and filenames from the fence info string) or dumps multiple snippets into a target directory. Scripts starting with `#!` are automatically marked executable (`chmod 0755`).
* **Export & Inspection**: 
  * **Save Chat**: Export the conversation as a formatted Markdown (`.md`) transcript or as a complete canonical JSON history (`.json`, including base64 images).
  * **Request Preview**: Inspect the exact HTTP URL, headers (with masked API keys), and JSON payload that will be sent over the wire.

---

## Build & Installation

### Dependencies
Requires **GTK 4.10+** (for `GtkAlertDialog`, `GtkFileDialog`) and **libcurl 7.32+**.

On Debian/Ubuntu:
```bash
sudo apt install build-essential pkg-config libgtk-4-dev \
                 libjson-glib-dev libcurl4-openssl-dev
```

### Compilation
Compile the single source file using `gcc` and `pkg-config`:
```bash
gcc -O2 -Wall -o ollama-gui snippet.c \
    $(pkg-config --cflags --libs gtk4 json-glib-1.0 libcurl)
```

Run the binary:
```bash
./ollama-gui
```

---

## Configurable Parameters & Settings

Settings can be adjusted in the left-hand pane of the UI and are persisted across sessions.

### 1. Provider & Authentication Pad
| Parameter | Description | Default Hosts & Models |
| :--- | :--- | :--- |
| **Provider** | Selects the target API format. | • **Ollama**: `http://localhost:11434` <br>• **OpenAI**: `https://api.openai.com/v1` (`gpt-4o-mini`)<br>• **Anthropic**: `https://api.anthropic.com/v1` (`claude-sonnet-4-5`)<br>• **Google**: `https://generativelanguage.googleapis.com/v1beta` (`gemini-2.5-flash`) |
| **API Key** | Bearer token or API key for cloud providers. Disabled for local Ollama; optional for local OpenAI-compatible servers. | *Empty* |
| **Host** | Base URL for the selected provider. | *Provider-dependent (see above)* |
| **Remember keys on disk** | When checked, saves API keys in plain text to the configuration file with `0600` (user read/write only) permissions. Unchecking purges saved keys from disk. | `Unchecked` (False) |

### 2. Model & System Prompt
* **Model**: Type a model ID manually (press `Enter` to add it to the dropdown) or click **Load models** to fetch available chat models from the provider's endpoint (`/api/tags` or `/models`).
* **System Prompt**: Sets global instructions for the conversation. Automatically mapped to `role: "system"` (Ollama/OpenAI), top-level `system` (Anthropic), or `systemInstruction` (Gemini).

### 3. Generation & Sampling Parameters
Each numeric sampling parameter has a checkbox. **If unchecked, the parameter is omitted from the JSON payload**, leaving it at the provider's default.

| Parameter | Default | Toggleable | Provider Behavior & Notes |
| :--- | :--- | :--- | :--- |
| **Temperature** | `0.80` | Yes (On) | Controls randomness. Clamped to `[0.0, 1.0]` for Anthropic. Omitted for Anthropic when Thinking is `On`. |
| **Top-p** | `0.90` | Yes (Off) | Nucleus sampling `(0.0, 1.0]`. For Anthropic, only `temperature` *or* `top_p` is sent, never both. |
| **Max tokens** | `4096` | Yes (Off) | Maps to `max_tokens` (or `max_completion_tokens` on OpenAI reasoners), `maxOutputTokens` (Gemini), or `num_predict` (Ollama). Required by Anthropic (defaults to `8192`, or `16000` with Thinking `On` if unchecked). |
| **Seed** | `42` | Yes (Off) | Deterministic sampling seed. Ignored by Anthropic. |
| **Thinking** | `Model default` | Dropdown | Options: `Model default (not sent)`, `On`, `Off`.<br>• **Anthropic**: Sends `adaptive` thinking (or legacy `budget_tokens` if adapted) when `On`, and `disabled` when `Off`.<br>• **Gemini**: Configures `thinkingConfig` (`thinkingBudget` & `includeThoughts`) on 2.5/3.x models.<br>• **Ollama**: Sets `"think": true/false`. |
| **Effort** | `Model default` | Dropdown | Options: `Model default (not sent)`, `low`, `medium`, `high`, `xhigh`, `max`.<br>• **Anthropic**: Sets `output_config.effort`.<br>• **OpenAI**: Sets `reasoning_effort` (`xhigh` and `max` are clamped to `high`).<br>• Ignored by Gemini and Ollama. |
| **Auto-adapt** | `Checked` | Checkbox | Automatically strips rejected parameters on HTTP 400/422 errors, records the model quirk, and retries (up to 4 times). Learned restrictions can be cleared per model via the **Reset** button. |
| **Stream** | `Checked` | Checkbox | Streams tokens in real time vs. waiting for the complete response body. |

---

## Configuration File (`config.ini`)

Configuration is automatically saved to:
```text
~/.config/ollama-gui/config.ini
```
*(File permissions are enforced to `0600`).*

### INI Structure
* `[ui]`: Stores `provider` index, active `model` string, and `stream` boolean.
* `[gen]`: Stores values and toggle states (`temperature`, `use_temperature`, `top_p`, `use_top_p`, `max_tokens`, `use_max_tokens`, `seed`, `use_seed`, `thinking`, `effort`, `auto_adapt`).
* `[auth]`: Stores `remember` boolean.
* `[hosts]`: Custom endpoint URLs keyed by `ollama`, `openai`, `anthropic`, and `google`.
* `[keys]`: API keys per provider (only present if `remember=true`).
* `[quirks]`: Bitmask flags of learned parameter restrictions keyed by `provider/model`.

---

## Compile-Time Constants

If you wish to tweak internal limits, the following macros are defined in the source file:

* `ERR_BODY_CAP` (`64 * 1024`): Maximum bytes of a streamed response body retained in memory for HTTP error reporting.
* `ATT_IMG_MAX` (`100 MiB`): Maximum allowed file size per image attachment.
* `ATT_TXT_MAX` (`64 MiB`): Maximum allowed file size per text attachment.
* `AUTO_SCROLL_EDGE` (`48.0` px) & `AUTO_SCROLL_SPEED` (`28.0` px/tick): Controls the edge zone and speed when drag-selecting text across the chat viewport.

## Keyboard Shortcuts
* **`Ctrl + Enter`**: Send message.
* **`Enter`** *(in Model entry box)*: Add custom model name to the dropdown list and select it.
