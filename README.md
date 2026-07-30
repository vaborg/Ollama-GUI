# Ollama GUI

A GTK4 desktop client for the [Ollama REST API](https://docs.ollama.com/api), written in C
against libcurl and json-glib. It is a request builder as much as a chat window: every
field on the left maps to a documented field of the JSON body, and **Preview request**
prints the exact bytes before anything is sent.

Four pages on the left — Request, Models, Server, Usage — share one chat transcript on
the right.

---

## Build

```bash
sudo apt install build-essential pkg-config libgtk-4-dev \
                 libjson-glib-dev libcurl4-openssl-dev
make
./ollama-gui
```

Or without the Makefile:

```bash
gcc -O2 -Wall -o ollama-gui ollama_gui.c \
    $(pkg-config --cflags --libs gtk4 json-glib-1.0 libcurl)
```

Requires GTK 4.10 or newer (`GtkFileDialog`, `GtkAlertDialog`). Verify the three
dependencies resolve before compiling:

```bash
pkg-config --modversion gtk4 json-glib-1.0 libcurl
```

Ollama itself is not a build dependency. The client speaks HTTP to whatever host you
point it at, though the Server page can start a local daemon for you.

---

## Request page

### Connection

| Field | Meaning | Sent as |
| --- | --- | --- |
| Host | Base URL of the server. Trailing slashes are stripped. | request URL prefix |
| ⟳ | Reload the model list, the installed inventory and the server version | `GET /api/tags`, `/api/ps`, `/api/version` |
| Model | Chosen from the models the server reports | `model` |
| Endpoint | `/api/chat` builds a `messages` array; `/api/generate` builds a single `prompt` | endpoint path |

The host is remembered between runs. Default is `http://localhost:11434`.

### System prompt

Free text. On `/api/chat` it becomes the first message with `"role": "system"`; on
`/api/generate` it becomes the top-level `system` field. Leave empty to omit it.

### Output

| Field | Effect | Sent as |
| --- | --- | --- |
| Format = free text | no constraint | *(field omitted)* |
| Format = JSON mode | model is asked for valid JSON | `"format": "json"` |
| Format = JSON schema | decoding is constrained to your schema | `"format": { …schema… }` |
| JSON schema box | the schema itself; must parse as JSON or the request is refused before sending | value of `format` |

The schema box is only read when Format is *JSON schema*. An empty box falls back to
plain JSON mode.

### Generation options

Empty fields are omitted entirely, so the server's own defaults apply. A field that does
not parse as a number is ignored rather than sent as garbage.

| Field | Type | Sent as | Notes |
| --- | --- | --- | --- |
| temperature | float | `options.temperature` | 0 for deterministic-ish output |
| top_p | float | `options.top_p` | nucleus sampling cutoff |
| num_ctx | int | `options.num_ctx` | context window in tokens; larger costs memory |
| num_predict | int | `options.num_predict` | max tokens to generate, `-1` = unlimited |
| seed | int | `options.seed` | fix it together with `temperature 0` for repeatable runs |
| keep_alive | string | `keep_alive` | how long the model stays resident, e.g. `5m`, `0`, `-1` |
| char limit | int | *(client-side only)* | truncation limit per text attachment, default 20000 characters |

### Toggles

| Toggle | Effect |
| --- | --- |
| Stream responses | `"stream": true`; tokens appear as they arrive and Stop becomes meaningful |
| Enable thinking | `"think": true`; only thinking-capable models accept this, others answer HTTP 400 |
| Send chat history | prior turns are replayed in `messages`; untick for one-shot requests |
| Also ground schema in the prompt | appends the schema text to your message, so the model sees the field names and not only the decoder constraint |
| Split follows mouse pointer | move the pointer left or right to steer the divider between panel and chat |
| Colour | markdown-aware rendering of replies: headings, fenced code with syntax highlighting, inline code, bold, links, quotes |

### Attachments

Pick a load mode, press **Attach…**, choose one or more files. The list below shows what
will be included; attachments are consumed by the next Send.

| Load mode | What happens | Lands in |
| --- | --- | --- |
| auto | by extension: `.png .jpg .jpeg .webp .gif .bmp` → image, `.json` → JSON, everything else → text | either |
| text | fenced block labelled with the file extension, truncated at *char limit* | `content` |
| image (base64) | file is base64-encoded, one entry per image | `images[]` |
| JSON | parsed first, so a malformed file fails in the client, then fenced as `json` | `content` |
| raw -> base64 | base64 of the raw bytes inline | `content` |

Text files are truncated on a UTF-8 character boundary and the truncation is marked in
the message, so the model knows it saw a fragment.

---

## Models page

| Control | Meaning |
| --- | --- |
| model name entry | Name as published in the library, e.g. `llama3.2:3b`, `qwen3:8b`, `gpt-oss:120b-cloud` |
| Pull | `POST /api/pull` with streaming progress |
| Cancel | aborts the transfer at the next chunk |
| progress bar / status | `completed / total` bytes and the server's stage text (`pulling manifest`, `verifying sha256`, …) |
| Refresh list | reloads everything |
| installed table | name, on-disk size, parameter count, quantisation, and whether the model is resident right now |

There is no library-browse endpoint, so names are typed, not picked from a catalogue. The
*loaded* column comes from `/api/ps`, which lists models currently held in memory. After a
successful pull the Request page's Model dropdown is rebuilt, so the new model is
immediately selectable.

---

## Server page

| Control | Meaning |
| --- | --- |
| server: … | result of `GET /api/version` — running with a version, or not reachable |
| Start cmd | command line to run, default `ollama serve`; parsed with shell quoting rules but not run through a shell |
| Start server | spawns it as a child process; stdout and stderr are merged into the log |
| Stop | sends SIGTERM to **that child only** — a systemd-managed daemon is unaffected |
| Check | probe the version again |
| Command | sign-in command, default `ollama signin` |
| Run sign-in | registers this machine's key with your ollama.com account, which cloud models require; a browser opens, and the URL is also printed to the log |
| API key | sent as `Authorization: Bearer …` on every request, pull and probe |
| Store the key… | writes the key into the config file, which is created mode 0600; off by default |
| Log | live output of both commands |

The API key is only needed when the Host points straight at `https://ollama.com`. A local
server that you have signed in signs its own cloud requests. If `OLLAMA_API_KEY` is set in
the environment, the field is pre-filled from it.

---

## Usage page

**Ollama exposes no account quota endpoint.** Token counts arrive with every response
(`prompt_eval_count`, `eval_count`), but plan limits and remaining balance live only in the
web dashboard. This page is therefore a local tally priced with rates you supply — an
independent estimate, not a reading of your account.

| Field | Meaning |
| --- | --- |
| in / 1M tok | price per million prompt tokens, in your currency |
| out / 1M tok | price per million generated tokens |
| budget | budget for the current period; leave at 0 to count tokens without pricing |
| currency | label only, no conversion is done |
| Apply | re-price the ledger with the current rates and save |
| Reset period | clear the ledger and stamp today as the new period start |
| Save table… | write the table to a text file |

The summary line reports spend, **paid usage** (spend up to the budget), **extra usage**
(spend above it) and what is left. A model counts as billed when its name ends in `-cloud`
or contains `:cloud`, or when the host is ollama.com. Everything else is tallied as local
and unbilled on a separate line.

Cost is computed as

```
cost = (billed_in_tokens × price_in + billed_out_tokens × price_out) / 1e6
```

---

## Composer and buttons

| Control | Meaning |
| --- | --- |
| Ctrl+Enter | send, without leaving the text box |
| Send | build the payload, echo it into the transcript, stream the reply |
| Stop | abort at the next chunk; partial text is kept in the history |
| Preview request | open a window with the exact JSON body, without sending |
| Extract code… | every fenced block of the last reply is written as a separate file |
| Extract points… | headings and bullet/numbered items of the last reply → `points.md` |
| Copy reply | last reply to the clipboard |
| Save chat… | the transcript as it appears, as text |
| New chat | clear the transcript and the message history (the ledger is untouched) |

Both extraction buttons ask for a folder and then create
`<folder>/ollama-extract/<YYYYMMDD-HHMMSS>/`, holding `code_01.py`, `code_02.c`, … or
`points.md`. Extensions come from the fence language, falling back to `.txt` for anything
unrecognised.

---

## Examples

### 1. First run, entirely local

1. **Server** page → **Start server** if `server:` reports not reachable. Watch the log for
   `Listening on 127.0.0.1:11434`.
2. **Models** page → type `llama3.2:3b` → **Pull**. The bar tracks the download.
3. **Request** page → pick the model in the dropdown → type in the composer → **Send**.

### 2. Summarise a document

Attachments: load mode **text**, **Attach…**, pick `report.md`. Set *char limit* to `40000`
if the file is long. Composer:

> Summarise this in five sentences, then list every numeric claim it makes.

The file arrives as a fenced block with its name, so the model can refer to it. Raise
*num_ctx* to something that fits the document — a 40000-character attachment needs roughly
`num_ctx` 16384 or more.

### 3. Ask about an image

Pull a vision model (`llava:7b`, `qwen2.5vl:7b`), set load mode **image (base64)**, attach
the picture, and ask your question. The image goes into `images[]`; the composer text stays
in `content`. Untick *Send chat history* afterwards if you do not want the base64 blob
replayed on every subsequent turn.

### 4. Structured output you can parse

Format → **JSON schema**, and in the schema box:

```json
{
  "type": "object",
  "properties": {
    "title":    {"type": "string"},
    "authors":  {"type": "array", "items": {"type": "string"}},
    "year":     {"type": "integer"},
    "keywords": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["title", "authors", "year"]
}
```

Leave *Also ground schema in the prompt* ticked. Attach the paper as **text**, ask
"extract the metadata", and the reply is valid JSON against that schema. **Copy reply**, or
**Save chat…** if you want the whole session.

### 5. Reproducible generation

*temperature* `0`, *seed* `42`, *Stream responses* off, *Send chat history* off. The same
prompt now yields the same output on the same model and server build. Non-streaming means
the reply appears in one piece, which is what you want when the output is a file rather
than something to read as it lands.

### 6. Get the code out of a long answer

Ask for an implementation, wait for the reply, press **Extract code…**, choose a folder.
Each fenced block becomes its own file with a sensible extension — useful when the model
returns a header, a source file and a build command in one answer.

### 7. Cloud model with budget tracking

1. **Server** page → **Run sign-in**, confirm in the browser.
2. **Models** page → pull the cloud variant, e.g. `gpt-oss:120b-cloud`.
3. **Usage** page → enter your plan's rates in *in / 1M tok* and *out / 1M tok*, set
   *budget*, press **Apply**.
4. Use the model normally. After each reply the ledger row for that model grows and the
   summary shows paid usage, extra usage and what is left.

### 8. Talk to ollama.com directly

Host → `https://ollama.com`. **Server** page → paste your API key. Every request now
carries `Authorization: Bearer …`. All models are treated as billed while that host is
selected. Tick *Store the key…* only if a 0600 file in your config directory is an
acceptable place for it.

### 9. Debug a request before spending tokens

Fill in the fields, press **Preview request**, and read the JSON. This is the fastest way to
see why a schema is not being applied, whether history is inflating the body, or which
`options` actually made it in.

---

## Files and environment

| Path / variable | Purpose |
| --- | --- |
| `~/.config/ollama-gui/state.json` | host, prices, budget, currency, commands, period start, ledger, optionally the API key; mode 0600 |
| `OLLAMA_API_KEY` | pre-fills the API key field at startup |
| `<chosen folder>/ollama-extract/<timestamp>/` | output of the two extraction buttons |

State is written after every response and again on exit, so the ledger survives restarts.
Deleting the file resets everything to defaults.

---

## Notes and limits

Requests run on worker threads and hand text to the GTK main loop through `g_idle_add`, so
the window stays responsive; no GTK call is made off the main thread. Cancellation is
cooperative — Stop takes effect when the next chunk arrives, not instantly.

Attached images stay in the conversation history and are resent on every turn while *Send
chat history* is ticked. This is correct for follow-up questions about the picture and
expensive for anything else.

*Enable thinking* is rejected by models without a thinking mode; the server answers HTTP
400 and the status line reports it.

The token ledger counts what Ollama reports. If the server omits `prompt_eval_count` or
`eval_count` — some cached-prompt paths do — that request contributes nothing to the tally.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `server: not reachable` | daemon not running — start it on the Server page, or check the Host field |
| Model dropdown empty | same, or the host points somewhere without models; press ⟳ |
| HTTP 401 on a cloud model | not signed in, or no API key when addressing ollama.com directly |
| HTTP 400 with a thinking model error | untick *Enable thinking* |
| Pull stalls at `pulling manifest` | no network route to the registry, or a name that does not exist |
| Reply is truncated JSON | raise *num_predict*, or lower the amount of attached text |
| Nothing in the ledger after a reply | the response carried no token counts, or the model was renamed between runs |
