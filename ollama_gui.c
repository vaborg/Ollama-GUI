/*
 * ollama_gui.c — single-file GTK4 chat client for local Ollama and the
 * cloud APIs of OpenAI, Anthropic and Google Gemini, with a provider
 * pad for API keys.
 *
 * Build dependencies (Debian/Ubuntu):
 *
 *   sudo apt install build-essential pkg-config libgtk-4-dev \
 *                    libjson-glib-dev libcurl4-openssl-dev
 *
 * Compile:
 *
 *   gcc -O2 -Wall -o cloud-chat ollama_gui.c \
 *       $(pkg-config --cflags --libs gtk4 json-glib-1.0 libcurl)
 *
 * Requires GTK 4.10 or newer (GtkAlertDialog) and libcurl 7.32 or
 * newer (CURLOPT_XFERINFOFUNCTION).
 *
 * The canonical conversation format is Ollama's (role/content/images);
 * it is translated on the wire for each vendor. Streaming runs on a
 * worker thread; no GTK call is ever made off the main thread.
 * Attachments: images go out as base64 image parts (PNG/JPEG/GIF/WebP
 * as-is, other formats re-encoded to PNG); UTF-8 text files are inlined
 * into the user turn as <file name="...">...</file>. "Save chat" writes
 * Markdown, or the canonical JSON history when the name ends in .json.
 * "Save code" writes the fenced code blocks of the latest reply that
 * has any: one block via a save dialog, several into a chosen folder.
 * Transcript: both your turns and the replies are full-width bubbles
 * (yours tinted blue, with a "You" caption); a reply's thinking sits
 * behind a collapsed "+ Thinking" toggle.
 * Parameters: each sampling parameter has an on/off switch; thinking and
 * effort can be left at the model default (omitted from the request).
 * Output length: "Max tokens" is off by default, so OpenAI, Gemini and
 * Ollama get their model maximum. Anthropic, where the field is
 * mandatory, gets 32000 (64000 with thinking on), clamped to 21333 for
 * non-streaming requests as the API demands. When an API answers that
 * the value is above the model's limit, that limit is learned (shown
 * under Parameters, saved in the config, resettable) and the turn is
 * resent with it. Auto-continue: a reply that still ends because of the
 * token limit is extended automatically — the partial reply is kept, a
 * "continue" nudge is sent, and the continuation is appended to the
 * same bubble and merged into the same history turn (up to 8 times).
 * Auto-adapt: when an API answers HTTP 400/422 naming a parameter, that
 * parameter is remembered as unsupported for that provider/model (saved
 * in the config, shown under Parameters, resettable) and the turn is
 * resent without it.
 * Config: ~/.config/ollama-gui/config.ini (chmod 600). API keys are
 * stored there in plain text ONLY if "Remember keys" is checked. The
 * last model name is remembered per provider.
 */

#include <gtk/gtk.h>
#include <json-glib/json-glib.h>
#include <curl/curl.h>
#include <glib/gstdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#if GLIB_CHECK_VERSION(2, 74, 0)
#define APP_FLAGS G_APPLICATION_DEFAULT_FLAGS
#else
#define APP_FLAGS G_APPLICATION_FLAGS_NONE
#endif

/* bytes of a streamed body kept for HTTP error reports */
#define ERR_BODY_CAP (64 * 1024)

/* edge zone (px) and max speed (px/tick) for drag-select auto-scroll */
#define AUTO_SCROLL_EDGE  48.0
#define AUTO_SCROLL_SPEED 28.0

/* auto-continue after the output-token limit, and auto-adapt resends */
#define MAX_CONTINUES      8u
#define MAX_ADAPT_RETRIES  6u

/* Anthropic: max_tokens is mandatory; these are used when "Max tokens"
 * is switched off. Non-streaming requests above 21333 are refused. */
#define ANTHROPIC_MAX_DEFAULT        32000
#define ANTHROPIC_MAX_DEFAULT_THINK  64000
#define ANTHROPIC_MAX_NONSTREAM      21333

static const char CONTINUE_PROMPT[] =
    "Continue exactly where you left off, without repeating anything you "
    "already wrote and without any preamble.";

/* ================================================================== */
/*  provider table                                                    */
/* ================================================================== */

typedef enum {
    PROV_OLLAMA = 0,     /* local, canonical format */
    PROV_OPENAI,          /* api.openai.com + any compatible server */
    PROV_ANTHROPIC,
    PROV_GOOGLE,
    PROV_COUNT
} ProvId;

typedef struct {
    const char *id_name;        /* config key            */
    const char *label;          /* drop-down label       */
    const char *default_host;
    const char *default_model;
} ProvInfo;

static const ProvInfo PROV[PROV_COUNT] = {
    [PROV_OLLAMA]    = { "ollama",    "Ollama (local)",
                         "http://localhost:11434", "" },
    [PROV_OPENAI]    = { "openai",    "OpenAI / compatible",
                         "https://api.openai.com/v1", "gpt-4o-mini" },
    [PROV_ANTHROPIC] = { "anthropic", "Anthropic (Claude)",
                         "https://api.anthropic.com/v1", "claude-sonnet-4-5" },
    [PROV_GOOGLE]    = { "google",    "Google (Gemini)",
                         "https://generativelanguage.googleapis.com/v1beta",
                         "gemini-2.5-flash" },
};

static const ProvInfo *prov_get(ProvId id)
{
    return &PROV[(id < PROV_COUNT) ? id : PROV_OLLAMA];
}

/* ================================================================== */
/*  canonical history entry                                           */
/* ================================================================== */

typedef struct {
    char     *role;      /* "system" | "user" | "assistant" */
    char     *text;
    GPtrArray *images;   /* char* base64, may be NULL */
} Msg;

static Msg *msg_new(const char *role, const char *text)
{
    Msg *m = g_new0(Msg, 1);

    m->role = g_strdup(role ? role : "user");
    m->text = g_strdup(text ? text : "");
    return m;
}

static void msg_free(gpointer p)
{
    Msg *m = p;

    g_free(m->role);
    g_free(m->text);
    if (m->images)
        g_ptr_array_unref(m->images);
    g_free(m);
}

/* ================================================================== */
/*  small json helpers                                                */
/* ================================================================== */

static const char *j_str(JsonObject *o, const char *k)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, k))
        return NULL;
    n = json_object_get_member(o, k);
    return JSON_NODE_HOLDS_VALUE(n) ? json_node_get_string(n) : NULL;
}

static gint64 j_int(JsonObject *o, const char *k, gint64 d)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, k))
        return d;
    n = json_object_get_member(o, k);
    return JSON_NODE_HOLDS_VALUE(n) ? json_node_get_int(n) : d;
}

static gboolean j_bool(JsonObject *o, const char *k)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, k))
        return FALSE;
    n = json_object_get_member(o, k);
    return JSON_NODE_HOLDS_VALUE(n) ? json_node_get_boolean(n) : FALSE;
}

static JsonObject *j_obj(JsonObject *o, const char *k)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, k))
        return NULL;
    n = json_object_get_member(o, k);
    return JSON_NODE_HOLDS_OBJECT(n) ? json_node_get_object(n) : NULL;
}

static JsonArray *j_arr(JsonObject *o, const char *k)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, k))
        return NULL;
    n = json_object_get_member(o, k);
    return JSON_NODE_HOLDS_ARRAY(n) ? json_node_get_array(n) : NULL;
}

static char *node_to_string(JsonNode *n, gboolean pretty)
{
    JsonGenerator *g = json_generator_new();
    char *s;

    json_generator_set_root(g, n);
    json_generator_set_pretty(g, pretty);
    if (pretty)
        json_generator_set_indent(g, 2);
    s = json_generator_to_data(g, NULL);
    g_object_unref(g);
    return s;
}

static char *j_ser(JsonNode *n)
{
    return node_to_string(n, FALSE);
}

/* {"error":"..."} or {"error":{"message":"..."}} -> message */
static char *err_text(JsonObject *o)
{
    JsonNode *n;
    const char *s = NULL;

    if (!o || !json_object_has_member(o, "error"))
        return NULL;
    n = json_object_get_member(o, "error");
    if (JSON_NODE_HOLDS_VALUE(n))
        s = json_node_get_string(n);
    else if (JSON_NODE_HOLDS_OBJECT(n))
        s = j_str(json_node_get_object(n), "message");
    return g_strdup(s ? s : "unknown API error");
}

/* base64 magic bytes -> MIME (attachments carry no extension here) */
static const char *image_mime(const char *b64)
{
    if (!b64)                            return "image/png";
    if (g_str_has_prefix(b64, "/9j/"))  return "image/jpeg";
    if (g_str_has_prefix(b64, "R0lGOD")) return "image/gif";
    if (g_str_has_prefix(b64, "UklGR"))  return "image/webp";
    return "image/png";
}

static char *strip_host(const char *host, const char *dflt)
{
    char *s;
    gsize n;

    if (!host || !*host)
        host = dflt;
    s = g_strstrip(g_strdup(host));
    n = strlen(s);
    while (n && s[n - 1] == '/')
        s[--n] = '\0';
    return s;
}

/* ================================================================== */
/*  per-provider message builders                                     */
/* ================================================================== */

/* OpenAI understands "system" natively — pass history through. */
static JsonArray *openai_messages(GPtrArray *hist)
{
    JsonArray *out = json_array_new();
    guint i, k;

    if (!hist)
        return out;
    for (i = 0; i < hist->len; i++) {
        Msg *m = g_ptr_array_index(hist, i);
        JsonObject *msg = json_object_new();

        json_object_set_string_member(msg, "role", m->role ? m->role : "user");
        if (m->images && m->images->len > 0) {
            JsonArray *parts = json_array_new();

            if (m->text && *m->text) {
                JsonObject *t = json_object_new();
                json_object_set_string_member(t, "type", "text");
                json_object_set_string_member(t, "text", m->text);
                json_array_add_object_element(parts, t);
            }
            for (k = 0; k < m->images->len; k++) {
                const char *b64 = g_ptr_array_index(m->images, k);
                JsonObject *img = json_object_new(), *iu = json_object_new();
                char *url = g_strdup_printf("data:%s;base64,%s",
                                            image_mime(b64), b64);

                json_object_set_string_member(img, "type", "image_url");
                json_object_set_string_member(iu, "url", url);
                json_object_set_object_member(img, "image_url", iu);
                json_array_add_object_element(parts, img);
                g_free(url);
            }
            json_object_set_array_member(msg, "content", parts);
        } else {
            json_object_set_string_member(msg, "content", m->text ? m->text : "");
        }
        json_array_add_object_element(out, msg);
    }
    return out;
}

/* Anthropic: system goes top-level, turns must strictly alternate
 * user/assistant and start with "user" — so consecutive same-role
 * turns are merged and a leading assistant turn gets a synthetic
 * user turn in front of it. */
typedef struct {
    const char *role;      /* static string */
    GString    *text;
    GPtrArray  *imgs;      /* borrowed const char* */
} ABucket;

static void abucket_free(gpointer p)
{
    ABucket *b = p;

    g_string_free(b->text, TRUE);
    g_ptr_array_unref(b->imgs);
    g_free(b);
}

static JsonArray *anthropic_messages(GPtrArray *hist, GString *system)
{
    GPtrArray *bk;
    JsonArray *out = json_array_new();
    ABucket *cur = NULL;
    guint i, k;

    if (!hist)
        return out;
    bk = g_ptr_array_new_with_free_func(abucket_free);
    for (i = 0; i < hist->len; i++) {
        Msg *m = g_ptr_array_index(hist, i);
        const char *role;

        if (!m->role)
            continue;
        if (!strcmp(m->role, "system")) {
            if (m->text && *m->text) {
                if (system->len)
                    g_string_append_c(system, '\n');
                g_string_append(system, m->text);
            }
            continue;
        }
        role = strcmp(m->role, "assistant") ? "user" : "assistant";
        if ((!m->text || !*m->text) && (!m->images || m->images->len == 0))
            continue;                              /* empty turns rejected */
        if (!cur || strcmp(cur->role, role)) {
            cur = g_new0(ABucket, 1);
            cur->role = role;
            cur->text = g_string_new(NULL);
            cur->imgs = g_ptr_array_new();
            g_ptr_array_add(bk, cur);
        }
        if (m->text && *m->text) {
            if (cur->text->len)
                g_string_append_c(cur->text, '\n');
            g_string_append(cur->text, m->text);
        }
        if (m->images)
            for (k = 0; k < m->images->len; k++)
                g_ptr_array_add(cur->imgs,
                                (gpointer)g_ptr_array_index(m->images, k));
    }

    if (bk->len &&
        strcmp(((ABucket *)g_ptr_array_index(bk, 0))->role, "user")) {
        ABucket *u = g_new0(ABucket, 1);

        u->role = "user";
        u->text = g_string_new("Continue this conversation:");
        u->imgs = g_ptr_array_new();
        g_ptr_array_insert(bk, 0, u);
    }

    for (i = 0; i < bk->len; i++) {
        ABucket *b = g_ptr_array_index(bk, i);
        JsonObject *msg = json_object_new();
        JsonArray *content = json_array_new();

        json_object_set_string_member(msg, "role", b->role);
        if (b->text->len) {
            JsonObject *t = json_object_new();
            json_object_set_string_member(t, "type", "text");
            json_object_set_string_member(t, "text", b->text->str);
            json_array_add_object_element(content, t);
        }
        for (k = 0; k < b->imgs->len; k++) {
            const char *b64 = g_ptr_array_index(b->imgs, k);
            JsonObject *ib = json_object_new(), *src = json_object_new();

            json_object_set_string_member(ib, "type", "image");
            json_object_set_string_member(src, "type", "base64");
            json_object_set_string_member(src, "media_type", image_mime(b64));
            json_object_set_string_member(src, "data", b64);
            json_object_set_object_member(ib, "source", src);
            json_array_add_object_element(content, ib);
        }
        json_object_set_array_member(msg, "content", content);
        json_array_add_object_element(out, msg);
    }
    g_ptr_array_free(bk, TRUE);
    return out;
}

/* Gemini: "assistant"->"model", system -> systemInstruction */
static JsonArray *gemini_contents(GPtrArray *hist, GString *system)
{
    JsonArray *out = json_array_new();
    guint i, k;

    if (!hist)
        return out;
    for (i = 0; i < hist->len; i++) {
        Msg *m = g_ptr_array_index(hist, i);
        const char *grole;
        JsonObject *msg;
        JsonArray *parts;

        if (!m->role)
            continue;
        if (!strcmp(m->role, "system")) {
            if (m->text && *m->text) {
                if (system->len)
                    g_string_append_c(system, '\n');
                g_string_append(system, m->text);
            }
            continue;
        }
        if ((!m->text || !*m->text) && (!m->images || m->images->len == 0))
            continue;
        grole = strcmp(m->role, "assistant") ? "user" : "model";

        msg = json_object_new();
        json_object_set_string_member(msg, "role", grole);
        parts = json_array_new();
        if (m->text && *m->text) {
            JsonObject *t = json_object_new();
            json_object_set_string_member(t, "text", m->text);
            json_array_add_object_element(parts, t);
        }
        if (m->images)
            for (k = 0; k < m->images->len; k++) {
                const char *b64 = g_ptr_array_index(m->images, k);
                JsonObject *ip = json_object_new(), *inl = json_object_new();

                json_object_set_string_member(inl, "mime_type", image_mime(b64));
                json_object_set_string_member(inl, "data", b64);
                json_object_set_object_member(ip, "inline_data", inl);
                json_array_add_object_element(parts, ip);
            }
        json_object_set_array_member(msg, "parts", parts);
        json_array_add_object_element(out, msg);
    }
    return out;
}

/* Ollama: verbatim copy of the canonical history */
static JsonArray *ollama_messages(GPtrArray *hist)
{
    JsonArray *out = json_array_new();
    guint i, k;

    if (!hist)
        return out;
    for (i = 0; i < hist->len; i++) {
        Msg *m = g_ptr_array_index(hist, i);
        JsonObject *nm = json_object_new();

        json_object_set_string_member(nm, "role", m->role ? m->role : "user");
        json_object_set_string_member(nm, "content", m->text ? m->text : "");
        if (m->images && m->images->len) {
            JsonArray *imgs = json_array_new();

            for (k = 0; k < m->images->len; k++)
                json_array_add_string_element(imgs,
                    g_ptr_array_index(m->images, k));
            json_object_set_array_member(nm, "images", imgs);
        }
        json_array_add_object_element(out, nm);
    }
    return out;
}

/* ================================================================== */
/*  request assembly                                                  */
/* ================================================================== */

typedef struct {
    char  *url;
    char  *body;      /* JSON to POST                        */
    char **headers;   /* NULL-terminated "Name: value" list */
    guint  sent;      /* P_* / S_* bits actually put in the body */
    int    max_sent;  /* output-token limit in the body, 0 = none */
} ProvRequest;

static void header_add(char ***hdrs, const char *fmt, ...)
{
    va_list ap;
    char *line;
    guint n = 0;

    va_start(ap, fmt);
    line = g_strdup_vprintf(fmt, ap);
    va_end(ap);
    if (*hdrs)
        while ((*hdrs)[n])
            n++;
    *hdrs = g_renew(char *, *hdrs, n + 2);
    (*hdrs)[n] = line;
    (*hdrs)[n + 1] = NULL;
}

static void prov_request_free(ProvRequest *r)
{
    char **h;

    if (!r)
        return;
    if (r->headers)
        for (h = r->headers; *h; h++)
            g_free(*h);
    g_free(r->headers);
    g_free(r->url);
    g_free(r->body);
    g_free(r);
}

/* parameter switches, and per-model quirks learned from HTTP 400s */
enum {
    P_TEMP      = 1u << 0,
    P_TOPP      = 1u << 1,
    P_MAXTOK    = 1u << 2,
    P_SEED      = 1u << 3,
    P_THINK     = 1u << 4,   /* send no thinking config at all          */
    P_EFFORT    = 1u << 5,
    P_NO_OFF    = 1u << 6,   /* thinking can't be disabled: Off = omit  */
    P_LEGACY    = 1u << 7,   /* Anthropic: budget_tokens, not adaptive  */
    P_MCT       = 1u << 8,   /* OpenAI: max_completion_tokens           */
    P_NODISPLAY = 1u << 9,   /* Anthropic: no thinking.display          */
    S_ADAPTIVE  = 1u << 16,  /* sent-only markers for the classifier    */
    S_DISABLED  = 1u << 17
};
#define Q_MASK 0xffffu

enum { THINK_DEFAULT = 0, THINK_ON, THINK_OFF };

static const char *const EFFORTS[] = {
    NULL, "low", "medium", "high", "xhigh", "max"
};

typedef struct {
    gboolean    use_temp, use_topp, use_maxtok, use_seed;
    double      temp, top_p;
    int         max_tokens;
    int         max_cap;    /* model's output limit, 0 = unknown */
    gint64      seed;
    int         think;      /* THINK_*                         */
    const char *effort;     /* NULL = model default (omitted)  */
    guint       quirks;     /* P_* bits never to send/how to send */
} GenParams;

/* o1/o3/o4… and gpt-5 refuse "max_tokens" and custom sampling */
static gboolean openai_reasoner(const char *m)
{
    return m && ((m[0] == 'o' && g_ascii_isdigit(m[1])) ||
                 g_str_has_prefix(m, "gpt-5"));
}

/* known restrictions, so the first request already goes through;
 * anything else is learned from the API's own 400 messages */
static guint builtin_quirks(ProvId id, const char *model)
{
    if (!model)
        return 0;
    if (id == PROV_ANTHROPIC &&
        (g_str_has_prefix(model, "claude-fable") ||
         g_str_has_prefix(model, "claude-mythos")))
        return P_TEMP | P_TOPP | P_NO_OFF;    /* sampling removed, thinking always on */
    if (id == PROV_OPENAI && openai_reasoner(model))
        return P_TEMP | P_TOPP | P_SEED | P_MCT;
    return 0;
}

/* known output-token ceilings (0 = unknown, let the API tell us) */
static guint builtin_cap(ProvId id, const char *model)
{
    if (!model)
        return 0;
    if (id == PROV_ANTHROPIC) {
        if (g_str_has_prefix(model, "claude-3-5-"))
            return 8192;
        if (g_str_has_prefix(model, "claude-3-7"))
            return 64000;
        if (g_str_has_prefix(model, "claude-3-"))
            return 4096;
        if (g_str_has_prefix(model, "claude-opus-4-") &&
            model[14] && strchr("012", model[14]))      /* 4, 4.1 */
            return 32000;
    } else if (id == PROV_OPENAI) {
        if (g_str_has_prefix(model, "gpt-4o"))
            return 16384;
        if (g_str_has_prefix(model, "gpt-4-") || !strcmp(model, "gpt-4") ||
            g_str_has_prefix(model, "gpt-3.5"))
            return 4096;
    }
    return 0;
}

/* Which of the parameters we sent does this error message complain
 * about? Returns only bits not already in `quirks`. */
static guint classify_rejection(ProvId id, const char *msg, guint sent,
                                guint quirks)
{
    char *m = g_ascii_strdown(msg ? msg : "", -1);
    guint add = 0;

    if ((sent & P_TOPP) &&
        (strstr(m, "top_p") || strstr(m, "topp") || strstr(m, "top-p")))
        add |= P_TOPP;
    /* "temperature and top_p cannot both be specified": top_p is enough */
    if ((sent & P_TEMP) && strstr(m, "temperature") &&
        !((add & P_TOPP) && strstr(m, "both")))
        add |= P_TEMP;
    if ((sent & P_SEED) && strstr(m, "seed"))
        add |= P_SEED;
    if ((sent & P_EFFORT) && strstr(m, "effort"))
        add |= P_EFFORT;
    if ((sent & P_MAXTOK) && id != PROV_ANTHROPIC) {    /* required there */
        if (id == PROV_OPENAI && !(quirks & P_MCT) &&
            strstr(m, "max_completion_tokens"))
            add |= P_MCT;
        else if (strstr(m, "max_tokens") || strstr(m, "max_completion_tokens") ||
                 strstr(m, "maxoutputtokens") || strstr(m, "num_predict"))
            add |= P_MAXTOK;
    }
    if (sent & P_THINK) {
        if ((sent & S_DISABLED) &&
            (strstr(m, "disabled") || strstr(m, "budget 0") ||
             strstr(m, "thinking mode") || strstr(m, "cannot be disabled")))
            add |= P_NO_OFF;
        else if ((sent & S_ADAPTIVE) && strstr(m, "display"))
            add |= P_NODISPLAY;
        else if ((sent & S_ADAPTIVE) &&
                 (strstr(m, "adaptive") || strstr(m, "'enabled'")))
            add |= P_LEGACY;
        else if (strstr(m, "think") || strstr(m, "budget"))
            add |= P_THINK;
    }
    g_free(m);
    return add & ~quirks & Q_MASK;
}

/* "max_tokens: 64000 > 8192, which is the maximum allowed…" or
 * "This model supports at most 16384 completion tokens" -> 8192/16384.
 * The limit is the largest stand-alone number below what we sent;
 * digits glued to '-', '.', '_' or letters (model names, dates) are
 * ignored. 0 when the message is not a "too large" complaint. */
static int rejection_cap(const char *msg, int sent_max)
{
    char *m = g_ascii_strdown(msg ? msg : "", -1);
    const char *p;
    long best = 0;

    if (sent_max <= 0 ||
        !(strstr(m, "max_tokens") || strstr(m, "max_completion_tokens") ||
          strstr(m, "max_output_tokens") || strstr(m, "maxoutputtokens") ||
          strstr(m, "output tokens") || strstr(m, "completion tokens")) ||
        !(strchr(m, '>') || strstr(m, "too large") || strstr(m, "maximum") ||
          strstr(m, "at most") || strstr(m, "exceed") ||
          strstr(m, "less than") || strstr(m, "<="))) {
        g_free(m);
        return 0;
    }
    for (p = m; *p; ) {
        const char *q;

        if (!g_ascii_isdigit(*p)) {
            p++;
            continue;
        }
        q = p;
        while (g_ascii_isdigit(*q))
            q++;
        if (!(p > m && (g_ascii_isalnum(p[-1]) || strchr("-._", p[-1]))) &&
            !(*q && (g_ascii_isalnum(*q) || strchr("-._", *q)))) {
            long v = strtol(p, NULL, 10);

            if (v >= 256 && v < sent_max && v > best)
                best = v;
        }
        p = q;
    }
    g_free(m);
    return (int)best;
}

static ProvRequest *prov_build_request(ProvId id, const char *host,
                                       const char *model, const char *key,
                                       gboolean stream, const GenParams *gp,
                                       GPtrArray *hist)
{
    ProvRequest *r = g_new0(ProvRequest, 1);
    JsonObject *root = json_object_new();
    JsonNode *node;
    GString *system = g_string_new(NULL);
    char **h = NULL;
    char *base = strip_host(host, prov_get(id)->default_host);
    const guint q = gp->quirks;
    const int cap = gp->max_cap;
    const int maxreq = (cap > 0 && gp->max_tokens > cap) ? cap : gp->max_tokens;
    const gboolean temp_on = gp->use_temp && !(q & P_TEMP) && gp->temp >= 0.0;
    const gboolean topp_on = gp->use_topp && !(q & P_TOPP) &&
                             gp->top_p > 0.0 && gp->top_p <= 1.0;
    const gboolean max_on  = gp->use_maxtok && !(q & P_MAXTOK) && maxreq > 0;
    const gboolean seed_on = gp->use_seed && !(q & P_SEED) && gp->seed >= 0;
    const char *effort = (q & P_EFFORT) ? NULL : gp->effort;
    const int think = (q & P_THINK) ? THINK_DEFAULT
                    : (gp->think == THINK_OFF && (q & P_NO_OFF)) ? THINK_DEFAULT
                    : gp->think;
    guint sent = 0;

    if (!model || !*model)
        model = prov_get(id)->default_model;
    header_add(&h, "Content-Type: application/json");
    header_add(&h, "Expect:");           /* skip 100-continue pauses */

    switch (id) {
    case PROV_OPENAI:
        if (key && *key)                  /* LM Studio/llama.cpp run keyless */
            header_add(&h, "Authorization: Bearer %s", key);
        r->url = g_strdup_printf("%s/chat/completions", base);
        json_object_set_string_member(root, "model", model);
        json_object_set_array_member(root, "messages", openai_messages(hist));
        json_object_set_boolean_member(root, "stream", stream);
        if (max_on) {
            json_object_set_int_member(root,
                (q & P_MCT) ? "max_completion_tokens" : "max_tokens", maxreq);
            sent |= P_MAXTOK;
            r->max_sent = maxreq;
        }
        if (temp_on) {
            json_object_set_double_member(root, "temperature", gp->temp);
            sent |= P_TEMP;
        }
        if (topp_on) {
            json_object_set_double_member(root, "top_p", gp->top_p);
            sent |= P_TOPP;
        }
        if (seed_on) {
            json_object_set_int_member(root, "seed", gp->seed);
            sent |= P_SEED;
        }
        if (effort) {                     /* OpenAI tops out at "high" */
            json_object_set_string_member(root, "reasoning_effort",
                (!strcmp(effort, "xhigh") || !strcmp(effort, "max"))
                    ? "high" : effort);
            sent |= P_EFFORT;
        }
        if (stream && strstr(base, "openai.com")) {
            JsonObject *so = json_object_new();
            json_object_set_boolean_member(so, "include_usage", TRUE);
            json_object_set_object_member(root, "stream_options", so);
        }
        break;

    case PROV_ANTHROPIC: {
        /* max_tokens is mandatory; thinking counts against it */
        int maxtok = max_on ? maxreq
                   : (think == THINK_ON ? ANTHROPIC_MAX_DEFAULT_THINK
                                        : ANTHROPIC_MAX_DEFAULT);

        if (cap > 0 && maxtok > cap)
            maxtok = cap;
        if (!stream && maxtok > ANTHROPIC_MAX_NONSTREAM)
            maxtok = ANTHROPIC_MAX_NONSTREAM;   /* API rule: long jobs must stream */
        if (key && *key)
            header_add(&h, "x-api-key: %s", key);
        header_add(&h, "anthropic-version: 2023-06-01");
        r->url = g_strdup_printf("%s/messages", base);
        json_object_set_string_member(root, "model", model);
        json_object_set_array_member(root, "messages",
                                     anthropic_messages(hist, system));
        if (system->len)
            json_object_set_string_member(root, "system", system->str);
        if (max_on)
            sent |= P_MAXTOK;
        if (think == THINK_ON) {
            JsonObject *th = json_object_new();

            if (q & P_LEGACY) {           /* pre-4.6 models: fixed budget */
                int budget = MAX(1024, MIN(maxtok / 2, 8192));

                if (maxtok <= budget)
                    maxtok = budget + 4096;
                json_object_set_string_member(th, "type", "enabled");
                json_object_set_int_member(th, "budget_tokens", budget);
            } else {                      /* current models: adaptive */
                json_object_set_string_member(th, "type", "adaptive");
                if (!(q & P_NODISPLAY))
                    json_object_set_string_member(th, "display", "summarized");
                sent |= S_ADAPTIVE;
            }
            json_object_set_object_member(root, "thinking", th);
            sent |= P_THINK;
        } else if (think == THINK_OFF) {
            JsonObject *th = json_object_new();

            json_object_set_string_member(th, "type", "disabled");
            json_object_set_object_member(root, "thinking", th);
            sent |= P_THINK | S_DISABLED;
        }
        json_object_set_int_member(root, "max_tokens", maxtok);
        r->max_sent = maxtok;
        json_object_set_boolean_member(root, "stream", stream);
        /* no sampling with thinking on; and one of temperature / top_p,
         * since Claude 4.5 and later reject the two together */
        if (think != THINK_ON) {
            if (temp_on) {
                json_object_set_double_member(root, "temperature",
                                              CLAMP(gp->temp, 0.0, 1.0));
                sent |= P_TEMP;
            } else if (topp_on) {
                json_object_set_double_member(root, "top_p", gp->top_p);
                sent |= P_TOPP;
            }
        }
        if (effort) {
            JsonObject *oc = json_object_new();

            json_object_set_string_member(oc, "effort", effort);
            json_object_set_object_member(root, "output_config", oc);
            sent |= P_EFFORT;
        }
        break;
    }

    case PROV_GOOGLE: {
        /* thinkingConfig exists on 2.5 / 3.x model names only */
        gboolean tcfg = strstr(model, "2.5") || strstr(model, "-3");
        JsonObject *gc = json_object_new();

        if (key && *key)
            header_add(&h, "x-goog-api-key: %s", key);
        r->url = g_strdup_printf("%s/models/%s:%s%s", base, model,
                                 stream ? "streamGenerateContent"
                                        : "generateContent",
                                 stream ? "?alt=sse" : "");
        json_object_set_array_member(root, "contents",
                                     gemini_contents(hist, system));
        if (system->len) {
            JsonObject *si = json_object_new();
            JsonArray *parts = json_array_new();
            JsonObject *p = json_object_new();

            json_object_set_string_member(p, "text", system->str);
            json_array_add_object_element(parts, p);
            json_object_set_array_member(si, "parts", parts);
            json_object_set_object_member(root, "systemInstruction", si);
        }
        if (temp_on) {
            json_object_set_double_member(gc, "temperature", gp->temp);
            sent |= P_TEMP;
        }
        if (topp_on) {
            json_object_set_double_member(gc, "topP", gp->top_p);
            sent |= P_TOPP;
        }
        if (max_on) {
            json_object_set_int_member(gc, "maxOutputTokens", maxreq);
            sent |= P_MAXTOK;
            r->max_sent = maxreq;
        }
        if (seed_on) {
            json_object_set_int_member(gc, "seed", gp->seed);
            sent |= P_SEED;
        }
        /* Pro models cannot switch thinking off (budget 0 -> HTTP 400);
         * only Flash / Flash-Lite accept it. */
        if (tcfg && (think == THINK_ON ||
                     (think == THINK_OFF && strstr(model, "flash")))) {
            JsonObject *tc = json_object_new();

            json_object_set_int_member(tc, "thinkingBudget",
                                       think == THINK_ON ? -1 : 0);
            if (think == THINK_ON)
                json_object_set_boolean_member(tc, "includeThoughts", TRUE);
            json_object_set_object_member(gc, "thinkingConfig", tc);
            sent |= P_THINK | (think == THINK_OFF ? S_DISABLED : 0);
        }
        json_object_set_object_member(root, "generationConfig", gc);
        break;
    }

    default: {                           /* Ollama, canonical format */
        JsonObject *opt = json_object_new();

        r->url = g_strdup_printf("%s/api/chat", base);
        json_object_set_string_member(root, "model", model);
        json_object_set_array_member(root, "messages", ollama_messages(hist));
        json_object_set_boolean_member(root, "stream", stream);
        if (think != THINK_DEFAULT) {
            json_object_set_boolean_member(root, "think", think == THINK_ON);
            sent |= P_THINK | (think == THINK_OFF ? S_DISABLED : 0);
        }
        if (temp_on) {
            json_object_set_double_member(opt, "temperature", gp->temp);
            sent |= P_TEMP;
        }
        if (topp_on) {
            json_object_set_double_member(opt, "top_p", gp->top_p);
            sent |= P_TOPP;
        }
        if (max_on) {
            json_object_set_int_member(opt, "num_predict", maxreq);
            sent |= P_MAXTOK;
            r->max_sent = maxreq;
        }
        if (seed_on) {
            json_object_set_int_member(opt, "seed", gp->seed);
            sent |= P_SEED;
        }
        json_object_set_object_member(root, "options", opt);
        break;
    }
    }

    node = json_node_new(JSON_NODE_OBJECT);
    json_node_take_object(node, root);
    r->body = j_ser(node);
    json_node_unref(node);
    r->headers = h;
    r->sent = sent;
    g_string_free(system, TRUE);
    g_free(base);
    return r;
}

/* ================================================================== */
/*  reply parsing (stream + whole body)                               */
/* ================================================================== */

typedef struct {
    ProvId      id;
    gint64      in_tok, out_tok;
    gboolean    truncated;      /* stopped by the output-token limit */
    char       *error;
    JsonParser *jp;             /* reused for every streamed line */
} ProvParser;

static ProvParser *prov_parser_new(ProvId id)
{
    ProvParser *p = g_new0(ProvParser, 1);

    p->id = id;
    p->in_tok = p->out_tok = -1;
    p->jp = json_parser_new();
    return p;
}

static void prov_parser_free(ProvParser *p)
{
    if (!p)
        return;
    g_object_unref(p->jp);
    g_free(p->error);
    g_free(p);
}

static const char *prov_parser_error(const ProvParser *p)
{
    return p ? p->error : NULL;
}

static gboolean prov_parser_truncated(const ProvParser *p)
{
    return p ? p->truncated : FALSE;
}

static void prov_parser_usage(const ProvParser *p, gint64 *in_tok, gint64 *out_tok)
{
    if (in_tok)
        *in_tok = p ? p->in_tok : -1;
    if (out_tok)
        *out_tok = p ? p->out_tok : -1;
}

/* Gemini: text vs. thought is decided per part by the "thought" flag */
static void gemini_collect(JsonObject *o, GString *text, GString *think,
                           gboolean *trunc)
{
    JsonArray *cand = j_arr(o, "candidates");

    if (cand && json_array_get_length(cand) > 0) {
        JsonObject *c0 = json_array_get_object_element(cand, 0);
        JsonArray *parts = j_arr(j_obj(c0, "content"), "parts");
        const char *fr = j_str(c0, "finishReason");
        guint k;

        if (fr && trunc && !strcmp(fr, "MAX_TOKENS"))
            *trunc = TRUE;
        if (parts)
            for (k = 0; k < json_array_get_length(parts); k++) {
                JsonObject *pt = json_array_get_object_element(parts, k);
                const char *t = j_str(pt, "text");

                if (!t || !*t)
                    continue;
                if (j_bool(pt, "thought")) {
                    if (think)
                        g_string_append(think, t);
                } else if (text) {
                    g_string_append(text, t);
                }
            }
    }
}

/* Feed one line (NDJSON line or SSE "data:" line). */
static gboolean prov_parser_feed(ProvParser *p, const char *line,
                                 GString *text, GString *think)
{
    const char *s = line ? line : "";
    JsonParser *jp;
    JsonNode *rn;
    JsonObject *o;
    char *e;

    while (*s == ' ' || *s == '\t' || *s == '\r')
        s++;
    if (!*s)
        return TRUE;

    if (p->id != PROV_OLLAMA) {
        if (*s == ':')
            return TRUE;                    /* SSE keep-alive    */
        if (!g_str_has_prefix(s, "data:"))
            return TRUE;                    /* "event:" lines   */
        s += 5;
        while (*s == ' ')
            s++;
        if (!*s)
            return TRUE;
        if (!strcmp(s, "[DONE]"))
            return TRUE;                    /* OpenAI terminator */
    }

    jp = p->jp;
    if (!json_parser_load_from_data(jp, s, -1, NULL))
        return TRUE;                       /* tolerate noise */
    rn = json_parser_get_root(jp);
    if (!rn || !JSON_NODE_HOLDS_OBJECT(rn))
        return TRUE;
    o = json_node_get_object(rn);

    e = err_text(o);
    if (e) {
        g_free(p->error);
        p->error = e;
        return FALSE;
    }

    switch (p->id) {
    case PROV_OPENAI: {
        JsonArray *ch = j_arr(o, "choices");
        JsonObject *u = j_obj(o, "usage");

        if (ch && json_array_get_length(ch) > 0) {
            JsonObject *c0 = json_array_get_object_element(ch, 0);
            JsonObject *d = j_obj(c0, "delta");
            const char *t = j_str(d, "content");
            const char *r = j_str(d, "reasoning_content");
            const char *fr = j_str(c0, "finish_reason");

            if (!r)
                r = j_str(d, "reasoning");   /* OpenRouter/DeepSeek */
            if (t && *t && text)
                g_string_append(text, t);
            if (r && *r && think)
                g_string_append(think, r);
            if (fr && !strcmp(fr, "length"))
                p->truncated = TRUE;
        }
        if (u) {
            p->in_tok  = j_int(u, "prompt_tokens", p->in_tok);
            p->out_tok = j_int(u, "completion_tokens", p->out_tok);
        }
        break;
    }
    case PROV_ANTHROPIC: {
        const char *type = j_str(o, "type");

        if (type && !strcmp(type, "message_start")) {
            JsonObject *u = j_obj(j_obj(o, "message"), "usage");
            p->in_tok = j_int(u, "input_tokens", p->in_tok);
        } else if (type && !strcmp(type, "content_block_delta")) {
            JsonObject *d = j_obj(o, "delta");
            const char *dt = j_str(d, "type");

            if (dt && !strcmp(dt, "text_delta")) {
                const char *t = j_str(d, "text");
                if (t && text)
                    g_string_append(text, t);
            } else if (dt && !strcmp(dt, "thinking_delta")) {
                const char *t = j_str(d, "thinking");
                if (t && think)
                    g_string_append(think, t);
            }
        } else if (type && !strcmp(type, "message_delta")) {
            JsonObject *u = j_obj(o, "usage");
            const char *sr = j_str(j_obj(o, "delta"), "stop_reason");

            p->out_tok = j_int(u, "output_tokens", p->out_tok);
            if (sr && !strcmp(sr, "max_tokens"))
                p->truncated = TRUE;
        }
        break;
    }
    case PROV_GOOGLE: {
        JsonObject *um = j_obj(o, "usageMetadata");

        gemini_collect(o, text, think, &p->truncated);
        if (!j_arr(o, "candidates")) {
            const char *br = j_str(j_obj(o, "promptFeedback"), "blockReason");
            if (br) {
                g_free(p->error);
                p->error = g_strdup_printf("request blocked: %s", br);
                return FALSE;
            }
        }
        if (um) {
            p->in_tok  = j_int(um, "promptTokenCount", p->in_tok);
            p->out_tok = j_int(um, "candidatesTokenCount", p->out_tok);
        }
        break;
    }
    default: {                              /* Ollama NDJSON */
        const char *resp = j_str(o, "response");
        JsonObject *m = j_obj(o, "message");

        if (resp && *resp && text)
            g_string_append(text, resp);
        if (m) {
            const char *t  = j_str(m, "content");
            const char *th = j_str(m, "thinking");
            if (t && *t && text)
                g_string_append(text, t);
            if (th && *th && think)
                g_string_append(think, th);
        }
        if (j_bool(o, "done")) {
            const char *dr = j_str(o, "done_reason");

            p->in_tok  = j_int(o, "prompt_eval_count", p->in_tok);
            p->out_tok = j_int(o, "eval_count", p->out_tok);
            if (dr && !strcmp(dr, "length"))
                p->truncated = TRUE;
        }
        break;
    }
    }
    return TRUE;
}

/* whole-body parse for stream = FALSE */
static gboolean prov_parse_body(ProvId id, const char *body, gsize len,
                                GString *text, GString *think,
                                gint64 *in_tok, gint64 *out_tok,
                                gboolean *truncated, char **error)
{
    JsonParser *jp;
    JsonNode *rn;
    JsonObject *o;
    char *e;

    if (in_tok)
        *in_tok = -1;
    if (out_tok)
        *out_tok = -1;
    if (truncated)
        *truncated = FALSE;

    jp = json_parser_new();
    if (!json_parser_load_from_data(jp, body ? body : "", (gssize)len, NULL)) {
        g_object_unref(jp);
        if (error)
            *error = g_strdup("invalid JSON in response body");
        return FALSE;
    }
    rn = json_parser_get_root(jp);
    if (!rn || !JSON_NODE_HOLDS_OBJECT(rn)) {
        g_object_unref(jp);
        if (error)
            *error = g_strdup("unexpected response shape");
        return FALSE;
    }
    o = json_node_get_object(rn);

    e = err_text(o);
    if (e) {
        g_object_unref(jp);
        if (error)
            *error = e;
        else
            g_free(e);
        return FALSE;
    }

    switch (id) {
    case PROV_OPENAI: {
        JsonArray *ch = j_arr(o, "choices");
        JsonObject *c0 = NULL, *m = NULL, *u = j_obj(o, "usage");
        const char *t, *r, *fr;

        if (ch && json_array_get_length(ch) > 0) {
            c0 = json_array_get_object_element(ch, 0);
            m = j_obj(c0, "message");
        }
        t = j_str(m, "content");
        r = j_str(m, "reasoning_content");
        if (!r)
            r = j_str(m, "reasoning");
        fr = j_str(c0, "finish_reason");
        if (t && text)
            g_string_append(text, t);
        if (r && think)
            g_string_append(think, r);
        if (fr && truncated && !strcmp(fr, "length"))
            *truncated = TRUE;
        if (u) {
            if (in_tok)
                *in_tok = j_int(u, "prompt_tokens", -1);
            if (out_tok)
                *out_tok = j_int(u, "completion_tokens", -1);
        }
        break;
    }
    case PROV_ANTHROPIC: {
        JsonArray *c = j_arr(o, "content");
        JsonObject *u = j_obj(o, "usage");
        const char *sr = j_str(o, "stop_reason");
        guint k;

        if (c)
            for (k = 0; k < json_array_get_length(c); k++) {
                JsonObject *b = json_array_get_object_element(c, k);
                const char *ty = j_str(b, "type");

                if (ty && !strcmp(ty, "text")) {
                    const char *t = j_str(b, "text");
                    if (t && text)
                        g_string_append(text, t);
                } else if (ty && !strcmp(ty, "thinking")) {
                    const char *t = j_str(b, "thinking");
                    if (t && think)
                        g_string_append(think, t);
                }
            }
        if (sr && truncated && !strcmp(sr, "max_tokens"))
            *truncated = TRUE;
        if (u) {
            if (in_tok)
                *in_tok = j_int(u, "input_tokens", -1);
            if (out_tok)
                *out_tok = j_int(u, "output_tokens", -1);
        }
        break;
    }
    case PROV_GOOGLE: {
        JsonObject *um = j_obj(o, "usageMetadata");
        JsonObject *pf = j_obj(o, "promptFeedback");
        const char *br = pf ? j_str(pf, "blockReason") : NULL;

        gemini_collect(o, text, think, truncated);
        if (br) {
            g_object_unref(jp);
            if (error)
                *error = g_strdup_printf("request blocked: %s", br);
            return FALSE;
        }
        if (um) {
            if (in_tok)
                *in_tok = j_int(um, "promptTokenCount", -1);
            if (out_tok)
                *out_tok = j_int(um, "candidatesTokenCount", -1);
        }
        break;
    }
    default: {                              /* Ollama */
        const char *resp = j_str(o, "response");
        const char *dr = j_str(o, "done_reason");
        JsonObject *m = j_obj(o, "message");

        if (resp && text)
            g_string_append(text, resp);
        if (m) {
            const char *t  = j_str(m, "content");
            const char *th = j_str(m, "thinking");
            if (t && text)
                g_string_append(text, t);
            if (th && think)
                g_string_append(think, th);
        }
        if (dr && truncated && !strcmp(dr, "length"))
            *truncated = TRUE;
        if (in_tok)
            *in_tok = j_int(o, "prompt_eval_count", -1);
        if (out_tok)
            *out_tok = j_int(o, "eval_count", -1);
        break;
    }
    }
    g_object_unref(jp);
    return TRUE;
}

/* pull a {"error":…} out of an HTTP error body */
static char *body_error(const char *body, gsize len)
{
    JsonParser *p = json_parser_new();
    char *msg = NULL;

    if (json_parser_load_from_data(p, body ? body : "", (gssize)len, NULL)) {
        JsonNode *rn = json_parser_get_root(p);

        if (rn && JSON_NODE_HOLDS_OBJECT(rn))
            msg = err_text(json_node_get_object(rn));
    }
    g_object_unref(p);
    return msg;
}

/* ================================================================== */
/*  model inventory                                                   */
/* ================================================================== */

typedef struct {
    char  *url;
    char **headers;
} ProvListReq;

static void prov_list_req_free(ProvListReq *r)
{
    char **h;

    if (!r)
        return;
    if (r->headers)
        for (h = r->headers; *h; h++)
            g_free(*h);
    g_free(r->headers);
    g_free(r->url);
    g_free(r);
}

static ProvListReq *prov_list_models(ProvId id, const char *host, const char *key)
{
    ProvListReq *r = g_new0(ProvListReq, 1);
    char **h = NULL;
    char *base = strip_host(host, prov_get(id)->default_host);

    header_add(&h, "Accept: application/json");
    switch (id) {
    case PROV_OPENAI:
        r->url = g_strdup_printf("%s/models", base);
        if (key && *key)
            header_add(&h, "Authorization: Bearer %s", key);
        break;
    case PROV_ANTHROPIC:
        r->url = g_strdup_printf("%s/models?limit=1000", base);
        if (key && *key)
            header_add(&h, "x-api-key: %s", key);
        header_add(&h, "anthropic-version: 2023-06-01");
        break;
    case PROV_GOOGLE:
        r->url = g_strdup_printf("%s/models?pageSize=1000", base);
        if (key && *key)
            header_add(&h, "x-goog-api-key: %s", key);
        break;
    default:
        r->url = g_strdup_printf("%s/api/tags", base);
        break;
    }
    r->headers = h;
    g_free(base);
    return r;
}

static gint cmp_str(gconstpointer a, gconstpointer b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static char **finish_models(GPtrArray *a)
{
    g_ptr_array_sort(a, cmp_str);
    g_ptr_array_add(a, NULL);
    return (char **)g_ptr_array_free(a, FALSE);   /* g_strfreev() it */
}

static char **prov_parse_models(ProvId id, const char *body, gsize len)
{
    GPtrArray *out = g_ptr_array_new_with_free_func(g_free);
    JsonParser *jp = json_parser_new();
    JsonNode *rn;
    JsonObject *o;
    JsonArray *arr = NULL;
    guint i, k;

    if (json_parser_load_from_data(jp, body ? body : "", (gssize)len, NULL)) {
        rn = json_parser_get_root(jp);
        if (rn && JSON_NODE_HOLDS_OBJECT(rn)) {
            o = json_node_get_object(rn);
            arr = j_arr(o, "models");
            if (!arr)
                arr = j_arr(o, "data");   /* openai / anthropic */
        }
    }
    if (arr)
        for (i = 0; i < json_array_get_length(arr); i++) {
            JsonObject *m = json_array_get_object_element(arr, i);
            const char *name = (id == PROV_OPENAI || id == PROV_ANTHROPIC)
                               ? j_str(m, "id") : j_str(m, "name");

            if (!name || !*name)
                continue;
            if (id == PROV_GOOGLE) {
                JsonArray *meth = j_arr(m, "supportedGenerationMethods");
                gboolean chat = FALSE;

                if (meth)
                    for (k = 0; k < json_array_get_length(meth); k++)
                        if (!g_strcmp0(json_array_get_string_element(meth, k),
                                    "generateContent")) {
                            chat = TRUE;
                            break;
                        }
                if (!chat)
                    continue;
                if (g_str_has_prefix(name, "models/"))
                    name += strlen("models/");
            }
            if (id == PROV_OPENAI) {      /* skip non-chat models */
                static const char *skip[] = {
                    "dall-e", "whisper", "tts", "embedding", "moderation",
                    "transcribe", "realtime", "audio", "image", NULL
                };
                const char **sp;
                gboolean bad = FALSE;

                for (sp = skip; *sp; sp++)
                    if (strstr(name, *sp)) {
                        bad = TRUE;
                        break;
                    }
                if (bad)
                    continue;
            }
            g_ptr_array_add(out, g_strdup(name));
        }
    g_object_unref(jp);
    return finish_models(out);
}

/* ================================================================== */
/*  application state                                                 */
/* ================================================================== */

typedef struct App App;
typedef struct ModelsJob ModelsJob;

typedef struct {
    char    *name;       /* basename: display and <file name="..."> */
    gboolean image;
    char    *data;       /* base64 (image) or UTF-8 text             */
    gsize    bytes;      /* size on disk                             */
} Attachment;

typedef struct {
    App         *app;
    ProvRequest *req;
    ProvId       prov;
    gboolean     stream;
    GThread     *thread;       /* joined by worker_done_idle / shutdown */
    gint         refs;         /* g_atomic: done path + one queued flush */
    gint         cancel;       /* g_atomic access */
    gboolean     cancelled;    /* worker thread only, until done */
    gboolean     truncated;    /* ended because of the output-token limit */
    ProvParser  *parser;
    GString     *pending;      /* stream line buffer */
    GString     *full;         /* body; streams keep the first ERR_BODY_CAP */
    GString     *line_txt, *line_thk;   /* per-line scratch (worker) */
    char        *error;
    gint64       in_tok, out_tok;
    long         http_code;
    char        *qkey;         /* "provider/model" the quirks belong to */
    guint        quirks;       /* quirks in force for this request */
    guint        retries;      /* auto-adapt resends so far */

    /* worker -> main loop handoff, coalesced into one idle at a time */
    GMutex       q_lock;
    GString     *q_text, *q_think;
    gboolean     q_scheduled;
} Worker;

struct ModelsJob {
    App     *app;
    ProvId   id;
    char    *url;
    char   **headers;
    char    *err;
    char   **names;
    GThread *thread;
    gint     cancel;           /* g_atomic access */
};

struct App {
    GtkWindow     *win;

    /* provider pad */
    GtkWidget     *prov_dd, *key_e, *host_e, *remember_cb;
    ProvId         prov_id;
    char          *keys[PROV_COUNT];
    char          *hosts[PROV_COUNT];
    char          *pmodel[PROV_COUNT];  /* last model name per provider */

    /* model + parameters */
    GtkWidget     *model_entry, *model_dd;
    GtkStringList *model_list;
    GPtrArray     *models;           /* char* */
    GtkWidget     *temp_e, *topp_e, *maxtok_e, *seed_e;
    GtkWidget     *temp_cb, *topp_cb, *maxtok_cb, *seed_cb;   /* send it? */
    GtkWidget     *think_dd, *effort_dd, *adapt_cb, *cont_cb;
    GtkWidget     *quirk_label, *quirk_reset;
    GHashTable    *quirks;            /* "provider/model" -> learned P_* */
    GHashTable    *caps;              /* "provider/model" -> output limit */
    GtkWidget     *stream_cb, *newchat_btn;
    GtkTextBuffer *sys_buf;

    /* chat */
    GtkWidget     *chat_sw;           /* scrolled window holding chat_box */
    GtkWidget     *chat_box;          /* vertical list of bubbles */
    GtkAdjustment *chat_adj;
    gboolean       follow;            /* keep the view pinned to the end */
    guint          drag_scroll_id;    /* timer for mouse-select auto-scroll */
    double         drag_scroll_step;  /* px per tick (+down / -up) */
    double         drag_x, drag_y;    /* last pointer pos in chat_sw coords */
    GtkTextBuffer *input_buf;
    GtkWidget     *send_btn, *stop_btn, *spinner, *status;

    /* state */
    GPtrArray     *history;           /* Msg* */
    GString       *reply_acc;
    gboolean       busy;
    GtkWidget     *cur_box;           /* reply bubble being streamed */
    GtkWidget     *cur_anchor;        /* the thinking toggle goes after this */
    GtkTextBuffer *cur_think_buf, *cur_text_buf;   /* NULL until needed */
    guint          cont_count;        /* auto-continues for this turn */
    gboolean       continuing;        /* current request is a continuation */
    Worker        *worker;
    ModelsJob     *models_job;        /* model list being loaded, or NULL */
    gint64         total_in, total_out, requests;
    gboolean       quiet;             /* programmatic entry updates */
    gboolean       closing;           /* window gone: idles skip the UI */
    guint          save_id;           /* debounced config save */

    /* attachments + saving */
    GPtrArray     *att;               /* Attachment*, sent with next turn */
    GtkWidget     *att_label, *att_clear_btn, *code_btn;

    /* config staging (loaded before widgets exist) */
    int            cfg_provider;
    char          *cfg_model;
    double         cfg_temp, cfg_topp;
    gint64         cfg_maxtok, cfg_seed;
    gboolean       cfg_stream, cfg_remember, cfg_adapt, cfg_cont;
    gboolean       cfg_use_temp, cfg_use_topp, cfg_use_maxtok, cfg_use_seed;
    int            cfg_think, cfg_effort;
};

/* ================================================================== */
/*  small UI helpers                                                  */
/* ================================================================== */

static void margins(GtkWidget *w, int m)
{
    gtk_widget_set_margin_start(w, m);
    gtk_widget_set_margin_end(w, m);
    gtk_widget_set_margin_top(w, m);
    gtk_widget_set_margin_bottom(w, m);
}

static char *buffer_text(GtkTextBuffer *b)
{
    GtkTextIter s, e;

    gtk_text_buffer_get_bounds(b, &s, &e);
    return gtk_text_buffer_get_text(b, &s, &e, FALSE);
}

static const char *current_model(App *app)
{
    return gtk_editable_get_text(GTK_EDITABLE(app->model_entry));
}

/* ------------------------------------------------------------------ */
/*  transcript: bubbles in a vertical box                             */
/* ------------------------------------------------------------------ */

static const char CHAT_CSS[] =
    ".chat { padding: 10px; }\n"
    ".chat-window { border-radius: 6px; }\n"
    ".bubble-user { background-color: alpha(#3584e4, 0.18);"
    "  border-radius: 14px 14px 4px 14px; padding: 8px 12px; }\n"
    ".bubble-asst { background-color: alpha(#808080, 0.12);"
    "  border-radius: 14px 14px 14px 4px; padding: 8px 12px; }\n"
    ".bubble-user textview, .bubble-user textview text,"
    ".bubble-asst textview, .bubble-asst textview text"
    "  { background-color: transparent; }\n"
    ".think-toggle { padding: 1px 8px; min-height: 0; font-size: smaller; }\n"
    ".think-view { font-style: italic; opacity: 0.7; }\n"
    ".caption { font-size: smaller; font-weight: bold; }\n"
    ".note { font-size: smaller; }\n"
    ".note-error { color: #c01c28; }\n"
    ".cont-marker { font-size: smaller; font-style: italic; opacity: 0.7; }\n"
    ".composer > label { font-size: smaller; font-weight: bold; opacity: 0.75; }\n"
    ".composer-input { border-radius: 6px; }\n"
    ".composer-input textview { padding: 4px; }\n";

static void load_css(void)
{
    GtkCssProvider *p = gtk_css_provider_new();

#if GTK_CHECK_VERSION(4, 12, 0)
    gtk_css_provider_load_from_string(p, CHAT_CSS);
#else
    gtk_css_provider_load_from_data(p, CHAT_CSS, -1);
#endif
    gtk_style_context_add_provider_for_display(gdk_display_get_default(),
        GTK_STYLE_PROVIDER(p), GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(p);
}

/* follow the end unless the user has scrolled up */
static void on_adj_value(GtkAdjustment *a, gpointer data)
{
    App *app = data;

    app->follow = gtk_adjustment_get_value(a) + gtk_adjustment_get_page_size(a)
                  >= gtk_adjustment_get_upper(a) - 24.0;
}

static void on_adj_bounds(GObject *o, GParamSpec *ps, gpointer data)
{
    App *app = data;
    GtkAdjustment *a = GTK_ADJUSTMENT(o);

    (void)ps;
    if (app->follow)
        gtk_adjustment_set_value(a, gtk_adjustment_get_upper(a) -
                                    gtk_adjustment_get_page_size(a));
}

static void chat_follow(App *app)
{
    app->follow = TRUE;
    gtk_adjustment_set_value(app->chat_adj,
                             gtk_adjustment_get_upper(app->chat_adj) -
                             gtk_adjustment_get_page_size(app->chat_adj));
}

/* ------------------------------------------------------------------ */
/*  mouse drag-selection auto-scroll                                  */
/* ------------------------------------------------------------------ */

static void drag_scroll_stop(App *app)
{
    if (app->drag_scroll_id) {
        g_source_remove(app->drag_scroll_id);
        app->drag_scroll_id = 0;
    }
    app->drag_scroll_step = 0.0;
}

/* As the outer scrolled window scrolls under a stationary mouse cursor,
 * extend the active bubble's text selection to the new character under
 * the pointer. */
static void extend_selection_at_pointer(App *app)
{
    GtkWidget *focus;
    graphene_point_t pt, tv_pt;
    GtkTextView *tv;
    GtkTextBuffer *buf;
    GtkTextIter iter;
    int bx, by;

    if (!app->win || !app->chat_sw)
        return;
    focus = gtk_window_get_focus(app->win);
    if (!focus)
        return;
    if (!GTK_IS_TEXT_VIEW(focus))
        focus = gtk_widget_get_ancestor(focus, GTK_TYPE_TEXT_VIEW);
    if (!focus || !gtk_widget_is_ancestor(focus, app->chat_box))
        return;

    pt = GRAPHENE_POINT_INIT((float)app->drag_x, (float)app->drag_y);
    if (!gtk_widget_compute_point(app->chat_sw, focus, &pt, &tv_pt))
        return;

    tv = GTK_TEXT_VIEW(focus);
    gtk_text_view_window_to_buffer_coords(tv, GTK_TEXT_WINDOW_WIDGET,
                                          (int)tv_pt.x, (int)tv_pt.y,
                                          &bx, &by);
    gtk_text_view_get_iter_at_location(tv, &iter, bx, by);
    buf = gtk_text_view_get_buffer(tv);
    gtk_text_buffer_move_mark_by_name(buf, "insert", &iter);
}

static gboolean drag_scroll_tick(gpointer data)
{
    App *app = data;
    GtkAdjustment *a = app->chat_adj;
    double val, lo, hi, nv;

    if (app->closing || app->drag_scroll_step == 0.0) {
        app->drag_scroll_id = 0;
        return G_SOURCE_REMOVE;
    }
    val = gtk_adjustment_get_value(a);
    lo  = gtk_adjustment_get_lower(a);
    hi  = MAX(lo, gtk_adjustment_get_upper(a) - gtk_adjustment_get_page_size(a));
    nv  = CLAMP(val + app->drag_scroll_step, lo, hi);
    if (nv != val) {
        gtk_adjustment_set_value(a, nv);
        extend_selection_at_pointer(app);
    }
    return G_SOURCE_CONTINUE;
}

static void on_chat_motion(GtkEventControllerMotion *ctrl,
                           double x, double y, gpointer data)
{
    App *app = data;
    GdkEvent *ev = gtk_event_controller_get_current_event(
                       GTK_EVENT_CONTROLLER(ctrl));
    GdkModifierType state = ev ? gdk_event_get_modifier_state(ev) : 0;
    double h, step = 0.0;

    if (!(state & GDK_BUTTON1_MASK)) {
        drag_scroll_stop(app);
        return;
    }

    app->drag_x = x;
    app->drag_y = y;
    h = (double)gtk_widget_get_height(app->chat_sw);
    if (h <= 0.0)
        return;

    if (y > h - AUTO_SCROLL_EDGE) {
        double d = (y - (h - AUTO_SCROLL_EDGE)) / AUTO_SCROLL_EDGE;
        step = AUTO_SCROLL_SPEED * CLAMP(d, 0.05, 2.5);
    } else if (y < AUTO_SCROLL_EDGE) {
        double d = (AUTO_SCROLL_EDGE - y) / AUTO_SCROLL_EDGE;
        step = -AUTO_SCROLL_SPEED * CLAMP(d, 0.05, 2.5);
    }

    app->drag_scroll_step = step;
    if (step != 0.0 && !app->drag_scroll_id)
        app->drag_scroll_id = g_timeout_add(16, drag_scroll_tick, app);
    else if (step == 0.0)
        drag_scroll_stop(app);
}

static void on_chat_btn_released(GtkGestureClick *g, int n_press,
                                 double x, double y, gpointer data)
{
    (void)g;
    (void)n_press;
    (void)x;
    (void)y;
    drag_scroll_stop((App *)data);
}

static void chat_add(App *app, GtkWidget *w)
{
    gtk_box_append(GTK_BOX(app->chat_box), w);
    chat_follow(app);
}

static void chat_clear(App *app)
{
    GtkWidget *c;

    drag_scroll_stop(app);
    while ((c = gtk_widget_get_first_child(app->chat_box)) != NULL)
        gtk_box_remove(GTK_BOX(app->chat_box), c);
    app->cur_box = NULL;
    app->cur_anchor = NULL;
    app->cur_think_buf = app->cur_text_buf = NULL;
}

static GtkWidget *wrap_label(const char *text)
{
    GtkWidget *l = gtk_label_new(text);

    gtk_label_set_wrap(GTK_LABEL(l), TRUE);
    gtk_label_set_wrap_mode(GTK_LABEL(l), PANGO_WRAP_WORD_CHAR);
    gtk_label_set_xalign(GTK_LABEL(l), 0.0f);
    gtk_label_set_selectable(GTK_LABEL(l), TRUE);
    return l;
}

static GtkWidget *bubble_text_view(GtkTextBuffer **buf, const char *css)
{
    GtkWidget *tv = gtk_text_view_new();
    GtkTextBuffer *b = gtk_text_view_get_buffer(GTK_TEXT_VIEW(tv));

    gtk_text_view_set_editable(GTK_TEXT_VIEW(tv), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(tv), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(tv), GTK_WRAP_WORD_CHAR);
    if (css)
        gtk_widget_add_css_class(tv, css);
    if (buf)
        *buf = b;
    return tv;
}

/* small bold caption at the top of a bubble ("You", model name) */
static GtkWidget *bubble_caption(const char *text)
{
    GtkWidget *cap = gtk_label_new(text);

    gtk_widget_add_css_class(cap, "caption");
    gtk_widget_add_css_class(cap, "dim-label");
    gtk_widget_set_halign(cap, GTK_ALIGN_START);
    gtk_label_set_selectable(GTK_LABEL(cap), TRUE);
    return cap;
}

/* status / error lines inside the transcript */
static void chat_note(App *app, const char *text, gboolean error)
{
    GtkWidget *l = wrap_label(text);

    gtk_widget_add_css_class(l, "note");
    gtk_widget_add_css_class(l, error ? "note-error" : "dim-label");
    gtk_widget_set_halign(l, GTK_ALIGN_START);
    chat_add(app, l);
}

/* your turn: a full-width tinted bubble using GtkTextView for smooth
 * mouse selection */
static void chat_add_user(App *app, const char *text, GPtrArray *att)
{
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    guint i;

    gtk_widget_add_css_class(box, "bubble-user");
    gtk_widget_set_halign(box, GTK_ALIGN_FILL);
    gtk_widget_set_hexpand(box, TRUE);
    gtk_box_append(GTK_BOX(box), bubble_caption("You"));
    if (text && *text) {
        GtkTextBuffer *b = NULL;
        GtkWidget *tv = bubble_text_view(&b, NULL);

        gtk_text_buffer_set_text(b, text, -1);
        gtk_box_append(GTK_BOX(box), tv);
    }
    if (att && att->len) {
        GString *s = g_string_new(NULL);
        GtkWidget *l;

        for (i = 0; i < att->len; i++)
            g_string_append_printf(s, "%s%s", i ? " · " : "📎 ",
                ((Attachment *)g_ptr_array_index(att, i))->name);
        l = wrap_label(s->str);
        gtk_widget_add_css_class(l, "dim-label");
        gtk_box_append(GTK_BOX(box), l);
        g_string_free(s, TRUE);
    }
    chat_add(app, box);
}

/* reply: full-width bubble */
static void asst_begin(App *app)
{
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    const char *model = current_model(app);
    GtkWidget *cap = bubble_caption(model && *model ? model : "assistant");

    gtk_widget_add_css_class(box, "bubble-asst");
    gtk_widget_set_halign(box, GTK_ALIGN_FILL);
    gtk_widget_set_hexpand(box, TRUE);
    gtk_box_append(GTK_BOX(box), cap);
    app->cur_box = box;
    app->cur_anchor = cap;
    chat_add(app, box);
}

static void on_think_toggle(GtkButton *b, gpointer data)
{
    GtkRevealer *rev = data;
    gboolean show = !gtk_revealer_get_reveal_child(rev);

    gtk_revealer_set_reveal_child(rev, show);
    gtk_button_set_label(b, show ? "✕  Thinking" : "＋  Thinking");
}

/* collapsed thinking section, placed right under the caption (or under
 * the continuation marker for a continued reply) */
static void asst_think(App *app)
{
    GtkWidget *btn = gtk_button_new_with_label("＋  Thinking");
    GtkWidget *rev = gtk_revealer_new();
    GtkWidget *tv = bubble_text_view(&app->cur_think_buf, "think-view");
    GtkWidget *anchor = app->cur_anchor ? app->cur_anchor
                                        : gtk_widget_get_first_child(app->cur_box);

    gtk_widget_add_css_class(btn, "flat");
    gtk_widget_add_css_class(btn, "think-toggle");
    gtk_widget_set_halign(btn, GTK_ALIGN_START);
    gtk_revealer_set_transition_type(GTK_REVEALER(rev),
                                     GTK_REVEALER_TRANSITION_TYPE_SLIDE_DOWN);
    gtk_revealer_set_reveal_child(GTK_REVEALER(rev), FALSE);
    gtk_revealer_set_child(GTK_REVEALER(rev), tv);
    g_signal_connect(btn, "clicked", G_CALLBACK(on_think_toggle), rev);
    gtk_box_insert_child_after(GTK_BOX(app->cur_box), btn, anchor);
    gtk_box_insert_child_after(GTK_BOX(app->cur_box), rev, btn);
}

static void asst_text(App *app)
{
    gtk_box_append(GTK_BOX(app->cur_box),
                   bubble_text_view(&app->cur_text_buf, NULL));
}

/* the reply hit the token limit and is being continued: mark the seam
 * and start a fresh text (and thinking) section below it */
static void asst_continue_marker(App *app)
{
    GtkWidget *l = gtk_label_new("⟳ continued — the previous part hit the "
                                 "output-token limit");

    gtk_widget_add_css_class(l, "cont-marker");
    gtk_widget_add_css_class(l, "dim-label");
    gtk_widget_set_halign(l, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(app->cur_box), l);
    app->cur_anchor = l;
    app->cur_think_buf = app->cur_text_buf = NULL;
}

/* append, dropping the newlines models like to open with */
static void buf_append(GtkTextBuffer *b, const char *s)
{
    GtkTextIter end;

    if (gtk_text_buffer_get_char_count(b) == 0)
        while (*s == '\n' || *s == '\r')
            s++;
    if (!*s)
        return;
    gtk_text_buffer_get_end_iter(b, &end);
    gtk_text_buffer_insert(b, &end, s, -1);
}

/* ...and the trailing whitespace they close with */
static void buf_rstrip(GtkTextBuffer *b)
{
    GtkTextIter s, e;

    gtk_text_buffer_get_end_iter(b, &e);
    s = e;
    while (gtk_text_iter_backward_char(&s))
        if (!g_unichar_isspace(gtk_text_iter_get_char(&s))) {
            gtk_text_iter_forward_char(&s);
            break;
        }
    if (!gtk_text_iter_equal(&s, &e))
        gtk_text_buffer_delete(b, &s, &e);
}

static void set_status(App *app, const char *fmt, ...)
{
    va_list ap;
    char *s;

    va_start(ap, fmt);
    s = g_strdup_vprintf(fmt, ap);
    va_end(ap);
    gtk_label_set_text(GTK_LABEL(app->status), s);
    g_free(s);
}

static void show_error(App *app, const char *msg)
{
    GtkAlertDialog *d = gtk_alert_dialog_new("%s", msg);

    gtk_alert_dialog_show(d, app->win);
    g_object_unref(d);
}

static double entry_double(GtkWidget *e)
{
    const char *t = gtk_editable_get_text(GTK_EDITABLE(e));

    return (t && *t) ? g_ascii_strtod(t, NULL) : 0.0;
}

static gint64 entry_ll(GtkWidget *e, gint64 dflt)
{
    const char *t = gtk_editable_get_text(GTK_EDITABLE(e));

    return (t && *t) ? g_ascii_strtoll(t, NULL, 10) : dflt;
}

static void show_text_window(App *app, const char *title, const char *text)
{
    GtkWidget *win = gtk_window_new();
    GtkWidget *sw = gtk_scrolled_window_new();
    GtkWidget *tv = gtk_text_view_new();
    GtkTextBuffer *buf = gtk_text_buffer_new(NULL);

    gtk_text_buffer_set_text(buf, text, -1);
    gtk_text_view_set_buffer(GTK_TEXT_VIEW(tv), buf);
    gtk_text_view_set_editable(GTK_TEXT_VIEW(tv), FALSE);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(tv), TRUE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(tv), GTK_WRAP_WORD_CHAR);
    g_object_unref(buf);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
                                    GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(sw), tv);
    gtk_window_set_child(GTK_WINDOW(win), sw);
    gtk_window_set_title(GTK_WINDOW(win), title);
    gtk_window_set_default_size(GTK_WINDOW(win), 780, 560);
    gtk_window_set_transient_for(GTK_WINDOW(win), app->win);
    gtk_window_present(GTK_WINDOW(win));
}

/* ================================================================== */
/*  config persistence                                                */
/* ================================================================== */

static double key_dbl(GKeyFile *kf, const char *g, const char *k, double d)
{
    return g_key_file_has_key(kf, g, k, NULL)
           ? g_key_file_get_double(kf, g, k, NULL) : d;
}

static gint64 key_i64(GKeyFile *kf, const char *g, const char *k, gint64 d)
{
    return g_key_file_has_key(kf, g, k, NULL)
           ? g_key_file_get_int64(kf, g, k, NULL) : d;
}

static gboolean key_bool(GKeyFile *kf, const char *g, const char *k, gboolean d)
{
    return g_key_file_has_key(kf, g, k, NULL)
           ? g_key_file_get_boolean(kf, g, k, NULL) : d;
}

/* "provider/model" -> n as an integer key under `group` */
static void save_int_table(GKeyFile *kf, const char *group, GHashTable *t)
{
    GHashTableIter it;
    gpointer k, v;

    g_hash_table_iter_init(&it, t);
    while (g_hash_table_iter_next(&it, &k, &v))
        if (!strpbrk(k, "=[]\n"))         /* not a valid key name */
            g_key_file_set_integer(kf, group, k, (int)GPOINTER_TO_UINT(v));
}

static void load_int_table(GKeyFile *kf, const char *group, GHashTable *t,
                           guint mask)
{
    gsize n = 0, k;
    char **keys = g_key_file_get_keys(kf, group, &n, NULL);

    for (k = 0; keys && k < n; k++) {
        int raw = g_key_file_get_integer(kf, group, keys[k], NULL);
        guint v = raw > 0 ? ((guint)raw & mask) : 0;

        if (v)
            g_hash_table_replace(t, g_strdup(keys[k]), GUINT_TO_POINTER(v));
    }
    g_strfreev(keys);
}

static void config_save(App *app)
{
    GKeyFile *kf = g_key_file_new();
    char *dir = g_build_filename(g_get_user_config_dir(), "ollama-gui", NULL);
    char *path = g_build_filename(dir, "config.ini", NULL);
    gboolean remember;
    int i;

    remember = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->remember_cb));
    g_free(app->pmodel[app->prov_id]);
    app->pmodel[app->prov_id] = g_strdup(current_model(app));

    g_key_file_set_integer(kf, "ui", "provider", (int)app->prov_id);
    g_key_file_set_string(kf, "ui", "model", current_model(app));
    g_key_file_set_double(kf, "gen", "temperature", entry_double(app->temp_e));
    g_key_file_set_double(kf, "gen", "top_p", entry_double(app->topp_e));
    g_key_file_set_int64(kf, "gen", "max_tokens", entry_ll(app->maxtok_e, 0));
    g_key_file_set_int64(kf, "gen", "seed", entry_ll(app->seed_e, -1));
    g_key_file_set_boolean(kf, "ui", "stream",
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->stream_cb)));
    g_key_file_set_boolean(kf, "gen", "use_temperature",
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->temp_cb)));
    g_key_file_set_boolean(kf, "gen", "use_top_p",
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->topp_cb)));
    g_key_file_set_boolean(kf, "gen", "use_max_tokens",
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->maxtok_cb)));
    g_key_file_set_boolean(kf, "gen", "use_seed",
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->seed_cb)));
    g_key_file_set_integer(kf, "gen", "thinking",
        (int)gtk_drop_down_get_selected(GTK_DROP_DOWN(app->think_dd)));
    g_key_file_set_integer(kf, "gen", "effort",
        (int)gtk_drop_down_get_selected(GTK_DROP_DOWN(app->effort_dd)));
    g_key_file_set_boolean(kf, "gen", "auto_adapt",
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->adapt_cb)));
    g_key_file_set_boolean(kf, "gen", "auto_continue",
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->cont_cb)));
    save_int_table(kf, "quirks", app->quirks);
    save_int_table(kf, "caps", app->caps);
    g_key_file_set_boolean(kf, "auth", "remember", remember);
    for (i = 0; i < PROV_COUNT; i++) {
        const char *nm = prov_get((ProvId)i)->id_name;

        if (app->hosts[i] && *app->hosts[i])
            g_key_file_set_string(kf, "hosts", nm, app->hosts[i]);
        if (app->pmodel[i] && *app->pmodel[i])
            g_key_file_set_string(kf, "models", nm, app->pmodel[i]);
        if (remember && app->keys[i] && *app->keys[i])
            g_key_file_set_string(kf, "keys", nm, app->keys[i]);
    }
    if (!remember)
        g_key_file_remove_group(kf, "keys", NULL);   /* purge stale keys */
    g_mkdir_with_parents(dir, 0700);
    if (g_key_file_save_to_file(kf, path, NULL))
        g_chmod(path, 0600);                         /* keep it private */
    g_key_file_free(kf);
    g_free(path);
    g_free(dir);
}

static gboolean save_timeout(gpointer data)
{
    App *app = data;

    app->save_id = 0;
    if (!app->closing)
        config_save(app);
    return G_SOURCE_REMOVE;
}

/* one disk write per burst of keystrokes instead of one per keystroke */
static void config_save_later(App *app)
{
    if (app->closing)
        return;
    if (app->save_id)
        g_source_remove(app->save_id);
    app->save_id = g_timeout_add(500, save_timeout, app);
}

static void config_load(App *app)
{
    GKeyFile *kf = g_key_file_new();
    char *path = g_build_filename(g_get_user_config_dir(),
                                  "ollama-gui", "config.ini", NULL);
    int i;

    app->cfg_provider = 0;
    app->cfg_temp = 0.8;
    app->cfg_topp = 0.9;
    app->cfg_maxtok = 16384;
    app->cfg_seed = 42;
    app->cfg_stream = TRUE;
    app->cfg_think = THINK_DEFAULT;
    app->cfg_effort = 0;
    app->cfg_remember = FALSE;
    app->cfg_adapt = TRUE;
    app->cfg_cont = TRUE;
    app->cfg_use_temp = TRUE;
    app->cfg_use_topp = FALSE;
    app->cfg_use_maxtok = FALSE;
    app->cfg_use_seed = FALSE;

    if (g_key_file_load_from_file(kf, path, G_KEY_FILE_NONE, NULL)) {
        app->cfg_provider = g_key_file_get_integer(kf, "ui", "provider", NULL);
        if (app->cfg_provider < 0 || app->cfg_provider >= PROV_COUNT)
            app->cfg_provider = 0;
        app->cfg_model = g_key_file_get_string(kf, "ui", "model", NULL);
        app->cfg_temp  = key_dbl(kf, "gen", "temperature", 0.8);
        app->cfg_topp  = key_dbl(kf, "gen", "top_p", 0.9);
        app->cfg_maxtok = key_i64(kf, "gen", "max_tokens", 16384);
        app->cfg_seed  = key_i64(kf, "gen", "seed", 42);
        app->cfg_stream = key_bool(kf, "ui", "stream", TRUE);
        /* older configs: ui/think boolean, and every parameter always sent */
        app->cfg_think = g_key_file_has_key(kf, "gen", "thinking", NULL)
            ? (int)key_i64(kf, "gen", "thinking", THINK_DEFAULT)
            : (key_bool(kf, "ui", "think", FALSE) ? THINK_ON : THINK_DEFAULT);
        if (app->cfg_think < THINK_DEFAULT || app->cfg_think > THINK_OFF)
            app->cfg_think = THINK_DEFAULT;
        app->cfg_effort = (int)key_i64(kf, "gen", "effort", 0);
        if (app->cfg_effort < 0 || app->cfg_effort >= (int)G_N_ELEMENTS(EFFORTS))
            app->cfg_effort = 0;
        app->cfg_adapt = key_bool(kf, "gen", "auto_adapt", TRUE);
        app->cfg_cont = key_bool(kf, "gen", "auto_continue", TRUE);
        app->cfg_use_temp = key_bool(kf, "gen", "use_temperature", TRUE);
        app->cfg_use_topp = key_bool(kf, "gen", "use_top_p", FALSE);
        /* a missing switch (old config) means: let the model run to its
         * own maximum rather than cutting it off at a small value */
        app->cfg_use_maxtok = key_bool(kf, "gen", "use_max_tokens", FALSE);
        app->cfg_use_seed = key_bool(kf, "gen", "use_seed", app->cfg_seed >= 0);
        load_int_table(kf, "quirks", app->quirks, Q_MASK);
        load_int_table(kf, "caps", app->caps, 0x7fffffffu);
        app->cfg_remember = key_bool(kf, "auth", "remember", FALSE);
        for (i = 0; i < PROV_COUNT; i++) {
            const char *nm = prov_get((ProvId)i)->id_name;

            app->keys[i]   = g_key_file_get_string(kf, "keys", nm, NULL);
            app->hosts[i]  = g_key_file_get_string(kf, "hosts", nm, NULL);
            app->pmodel[i] = g_key_file_get_string(kf, "models", nm, NULL);
        }
    }
    g_key_file_free(kf);
    g_free(path);
}

/* ------------------------------------------------------------------ */
/*  per-model parameter quirks and output-token caps                  */
/* ------------------------------------------------------------------ */

static char *quirk_key(ProvId id, const char *model)
{
    return g_strdup_printf("%s/%s", prov_get(id)->id_name, model ? model : "");
}

static guint model_quirks(App *app, ProvId id, const char *model)
{
    char *k = quirk_key(id, model);
    guint q = builtin_quirks(id, model) |
              GPOINTER_TO_UINT(g_hash_table_lookup(app->quirks, k));

    g_free(k);
    return q;
}

static guint model_cap(App *app, ProvId id, const char *model)
{
    char *k = quirk_key(id, model);
    guint c = GPOINTER_TO_UINT(g_hash_table_lookup(app->caps, k));

    g_free(k);
    return c ? c : builtin_cap(id, model);
}

static char *quirk_names(guint q)
{
    static const struct { guint bit; const char *name; } N[] = {
        { P_TEMP, "temperature" },   { P_TOPP, "top-p" },
        { P_MAXTOK, "max tokens" },  { P_SEED, "seed" },
        { P_THINK, "thinking" },     { P_EFFORT, "effort" },
        { P_NO_OFF, "thinking off" },
        { P_LEGACY, "adaptive thinking (budget used)" },
        { P_MCT, "max_tokens (max_completion_tokens used)" },
        { P_NODISPLAY, "thinking display" },
    };
    GString *s = g_string_new(NULL);
    guint i;

    for (i = 0; i < G_N_ELEMENTS(N); i++)
        if (q & N[i].bit)
            g_string_append_printf(s, "%s%s", s->len ? " · " : "", N[i].name);
    return g_string_free(s, FALSE);
}

static void quirks_refresh(App *app)
{
    const char *model = current_model(app);
    guint q = model_quirks(app, app->prov_id, model);
    guint cap = model_cap(app, app->prov_id, model);
    char *k = quirk_key(app->prov_id, model);
    GString *s = g_string_new(NULL);

    if (q) {
        char *names = quirk_names(q);

        g_string_append_printf(s, "Not sent / adapted for this model: %s",
                               names);
        g_free(names);
    }
    if (cap)
        g_string_append_printf(s, "%sOutput limit for this model: %u tokens",
                               s->len ? "\n" : "", cap);
    gtk_label_set_text(GTK_LABEL(app->quirk_label), s->str);
    gtk_widget_set_visible(app->quirk_label, s->len > 0);
    gtk_widget_set_visible(app->quirk_reset,
                           g_hash_table_contains(app->quirks, k) ||
                           g_hash_table_contains(app->caps, k));
    g_string_free(s, TRUE);
    g_free(k);
}

static void quirks_learn(App *app, const char *key, guint add)
{
    guint old = GPOINTER_TO_UINT(g_hash_table_lookup(app->quirks, key));

    g_hash_table_replace(app->quirks, g_strdup(key),
                         GUINT_TO_POINTER((old | add) & Q_MASK));
    quirks_refresh(app);
    config_save(app);
}

static void caps_learn(App *app, const char *key, guint cap)
{
    g_hash_table_replace(app->caps, g_strdup(key), GUINT_TO_POINTER(cap));
    quirks_refresh(app);
    config_save(app);
}

static void gen_params(App *app, const char *model, GenParams *gp)
{
    guint t = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->think_dd));
    guint e = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->effort_dd));

    memset(gp, 0, sizeof *gp);
    gp->use_temp   = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->temp_cb));
    gp->use_topp   = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->topp_cb));
    gp->use_maxtok = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->maxtok_cb));
    gp->use_seed   = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->seed_cb));
    gp->temp       = entry_double(app->temp_e);
    gp->top_p      = entry_double(app->topp_e);
    gp->max_tokens = (int)entry_ll(app->maxtok_e, 0);
    gp->max_cap    = (int)model_cap(app, app->prov_id, model);
    gp->seed       = entry_ll(app->seed_e, -1);
    gp->think      = (t <= THINK_OFF) ? (int)t : THINK_DEFAULT;
    gp->effort     = (e < G_N_ELEMENTS(EFFORTS)) ? EFFORTS[e] : NULL;
    gp->quirks     = model_quirks(app, app->prov_id, model);
}

/* gtk_editable_set_text() deletes, then inserts, emitting "changed" for
 * each step. The delete step used to run on_key_changed/on_host_changed,
 * which freed app->keys[id] / app->hosts[id] -- the very string still
 * being inserted (use-after-free) -- and could save a blank key. Copy
 * first and keep the handlers quiet while the entries are refilled. */
static void provider_apply(App *app)
{
    guint i = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->prov_dd));
    ProvId id = (i == GTK_INVALID_LIST_POSITION || (int)i >= PROV_COUNT)
                ? PROV_OLLAMA : (ProvId)i;
    ProvId old = app->prov_id;
    char *key = g_strdup(app->keys[id] ? app->keys[id] : "");
    char *host = g_strdup((app->hosts[id] && *app->hosts[id])
                          ? app->hosts[id] : prov_get(id)->default_host);
    char *model = NULL;

    if (id != old) {   /* the model name and the listed models belong to the old one */
        guint cnt = g_list_model_get_n_items(G_LIST_MODEL(app->model_list));

        g_free(app->pmodel[old]);
        app->pmodel[old] = g_strdup(current_model(app));
        g_ptr_array_set_size(app->models, 0);
        if (cnt)
            gtk_string_list_splice(app->model_list, 0, cnt, NULL);
        model = g_strdup(app->pmodel[id] ? app->pmodel[id]
                                         : prov_get(id)->default_model);
    }
    app->prov_id = id;
    app->quiet = TRUE;
    gtk_editable_set_text(GTK_EDITABLE(app->key_e), key);
    gtk_editable_set_text(GTK_EDITABLE(app->host_e), host);
    app->quiet = FALSE;
    if (model) {
        gtk_editable_set_text(GTK_EDITABLE(app->model_entry), model);
        g_free(model);
    }
    gtk_widget_set_sensitive(app->key_e, id != PROV_OLLAMA);
    set_status(app, "Provider: %s", prov_get(id)->label);
    quirks_refresh(app);
    g_free(key);
    g_free(host);
}

static void apply_config(App *app)
{
    char buf[32];
    const char *m;

    /* pre-set the provider so provider_apply() sees no switch and does
     * not file the (still empty) model entry under the wrong provider */
    app->prov_id = (ProvId)app->cfg_provider;
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app->prov_dd),
                               (guint)app->cfg_provider);
    provider_apply(app);                       /* also for the no-change case */
    m = app->pmodel[app->prov_id];
    if ((!m || !*m) && app->cfg_model && *app->cfg_model)
        m = app->cfg_model;                    /* older configs */
    if (m && *m)
        gtk_editable_set_text(GTK_EDITABLE(app->model_entry), m);
    g_free(app->cfg_model);
    app->cfg_model = NULL;
    g_snprintf(buf, sizeof buf, "%.2f", app->cfg_temp);
    gtk_editable_set_text(GTK_EDITABLE(app->temp_e), buf);
    g_snprintf(buf, sizeof buf, "%.2f", app->cfg_topp);
    gtk_editable_set_text(GTK_EDITABLE(app->topp_e), buf);
    g_snprintf(buf, sizeof buf, "%lld", (long long)app->cfg_maxtok);
    gtk_editable_set_text(GTK_EDITABLE(app->maxtok_e), buf);
    g_snprintf(buf, sizeof buf, "%lld", (long long)app->cfg_seed);
    gtk_editable_set_text(GTK_EDITABLE(app->seed_e), buf);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->stream_cb), app->cfg_stream);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->temp_cb), app->cfg_use_temp);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->topp_cb), app->cfg_use_topp);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->maxtok_cb),
                                app->cfg_use_maxtok);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->seed_cb), app->cfg_use_seed);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app->think_dd), (guint)app->cfg_think);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app->effort_dd),
                               (guint)app->cfg_effort);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->adapt_cb), app->cfg_adapt);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->cont_cb), app->cfg_cont);
    quirks_refresh(app);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->remember_cb),
                                app->cfg_remember);
}

/* keep history[0] in sync with the system-prompt buffer */
static void sync_system(App *app)
{
    char *sys = buffer_text(app->sys_buf);
    Msg *first = app->history->len ? g_ptr_array_index(app->history, 0) : NULL;
    gboolean had = first && !g_strcmp0(first->role, "system");

    if (had) {
        g_free(first->text);
        first->text = sys;                     /* takes ownership */
        sys = NULL;
        if (!first->text || !*first->text)
            g_ptr_array_remove_index(app->history, 0);
    } else if (sys && *sys) {
        g_ptr_array_insert(app->history, 0, msg_new("system", sys));
    }
    g_free(sys);
}

/* ================================================================== */
/*  attachments                                                       */
/* ================================================================== */

#define ATT_IMG_MAX ((gsize)100u * 1024u * 1024u)   /* 100 MiB per image     */
#define ATT_TXT_MAX ((gsize)64u  * 1024u * 1024u)   /* 64 MiB per text file  */

static void attachment_free(gpointer p)
{
    Attachment *a = p;

    g_free(a->name);
    g_free(a->data);
    g_free(a);
}

/* formats all four providers take as-is; others are re-encoded */
static gboolean mime_passthrough(const char *mime)
{
    return !g_strcmp0(mime, "image/png")  || !g_strcmp0(mime, "image/jpeg") ||
           !g_strcmp0(mime, "image/gif")  || !g_strcmp0(mime, "image/webp");
}

static Attachment *attachment_load(GFile *f, char **why)
{
    char *name = g_file_get_basename(f), *data = NULL, *ctype, *mime;
    gsize len = 0;
    GError *err = NULL;
    Attachment *a = NULL;

    if (!g_file_load_contents(f, NULL, &data, &len, NULL, &err)) {
        *why = g_strdup_printf("%s: %s", name, err->message);
        g_error_free(err);
        g_free(name);
        return NULL;
    }
    ctype = g_content_type_guess(name, (const guchar *)data,
                                 MIN(len, (gsize)4096), NULL);
    mime = g_content_type_get_mime_type(ctype);

    if (mime && g_str_has_prefix(mime, "image/") &&
        strcmp(mime, "image/svg+xml") != 0) {
        if (len > ATT_IMG_MAX) {
            *why = g_strdup_printf("%s: image larger than %u MiB",
                                   name, (unsigned)(ATT_IMG_MAX >> 20));
        } else if (mime_passthrough(mime)) {
            a = g_new0(Attachment, 1);
            a->data = g_base64_encode((const guchar *)data, len);
        } else {                                 /* BMP, TIFF, ... -> PNG */
            GBytes *in = g_bytes_new(data, len);
            GdkTexture *t = gdk_texture_new_from_bytes(in, &err);

            g_bytes_unref(in);
            if (t) {
                GBytes *png = gdk_texture_save_to_png_bytes(t);
                gsize n = 0;
                gconstpointer d = g_bytes_get_data(png, &n);

                a = g_new0(Attachment, 1);
                a->data = g_base64_encode(d, n);
                g_bytes_unref(png);
                g_object_unref(t);
            } else {
                *why = g_strdup_printf("%s: unsupported image (%s)",
                                       name, err->message);
                g_error_free(err);
            }
        }
        if (a)
            a->image = TRUE;
    } else if (len > ATT_TXT_MAX) {
        *why = g_strdup_printf("%s: text file larger than %u MiB",
                               name, (unsigned)(ATT_TXT_MAX >> 20));
    } else if (memchr(data, '\0', len)) {
        *why = g_strdup_printf("%s: binary file (neither an image nor text)",
                               name);
    } else if (!g_utf8_validate(data, (gssize)len, NULL)) {
        /* Large logs / legacy encodings: sanitize invalid byte sequences
         * to U+FFFD instead of rejecting the file. */
        a = g_new0(Attachment, 1);
        a->data = g_utf8_make_valid(data, (gssize)len);
    } else {
        a = g_new0(Attachment, 1);
        a->data = data;                          /* takes ownership */
        data = NULL;
    }
    if (a) {
        a->name = name;
        a->bytes = len;
        name = NULL;
    }
    g_free(name);
    g_free(data);
    g_free(ctype);
    g_free(mime);
    return a;
}

static void att_refresh(App *app)
{
    GString *s = g_string_new(NULL);
    guint i;

    for (i = 0; i < app->att->len; i++) {
        Attachment *a = g_ptr_array_index(app->att, i);
        char *sz = g_format_size(a->bytes);

        g_string_append_printf(s, "%s%s (%s%s)", i ? " · " : "📎 ",
                               a->name, a->image ? "image, " : "", sz);
        g_free(sz);
    }
    gtk_label_set_text(GTK_LABEL(app->att_label),
                       s->len ? s->str : "no attachments");
    gtk_widget_set_tooltip_text(app->att_label, s->len ? s->str : NULL);
    gtk_widget_set_sensitive(app->att_clear_btn, app->att->len > 0);
    g_string_free(s, TRUE);
}

/* the user turn as the model sees it: typed text, inlined text files,
 * images as separate base64 parts */
static Msg *compose_user_msg(GPtrArray *att, const char *input)
{
    GString *t = g_string_new(input);
    Msg *m;
    guint i;

    for (i = 0; i < att->len; i++) {
        Attachment *a = g_ptr_array_index(att, i);
        gsize n;

        if (a->image)
            continue;
        n = strlen(a->data);
        if (t->len)
            g_string_append(t, "\n\n");
        g_string_append_printf(t, "<file name=\"%s\">\n%s%s</file>",
                               a->name, a->data,
                               (n && a->data[n - 1] == '\n') ? "" : "\n");
    }
    m = msg_new("user", t->str);
    g_string_free(t, TRUE);
    for (i = 0; i < att->len; i++) {
        Attachment *a = g_ptr_array_index(att, i);

        if (!a->image)
            continue;
        if (!m->images)
            m->images = g_ptr_array_new_with_free_func(g_free);
        g_ptr_array_add(m->images, g_strdup(a->data));
    }
    return m;
}

/* deep copy for the request preview, base64 shortened to a stub that
 * keeps the magic bytes image_mime() looks at */
static Msg *preview_copy(const Msg *src)
{
    Msg *m = msg_new(src->role, src->text);
    guint k;

    if (src->images && src->images->len) {
        m->images = g_ptr_array_new_with_free_func(g_free);
        for (k = 0; k < src->images->len; k++) {
            const char *b64 = g_ptr_array_index(src->images, k);
            char *sz = g_format_size(strlen(b64));

            g_ptr_array_add(m->images,
                            g_strdup_printf("%.12s…[%s base64]", b64, sz));
            g_free(sz);
        }
    }
    return m;
}

/* ================================================================== */
/*  fenced code blocks                                                */
/* ================================================================== */

typedef struct {
    char *lang;          /* lower-case info word, may be NULL */
    char *name;          /* file name from the info string, may be NULL */
    char *code;
} CodeBlock;

static void code_block_free(gpointer p)
{
    CodeBlock *b = p;

    g_free(b->lang);
    g_free(b->name);
    g_free(b->code);
    g_free(b);
}

static const struct { const char *lang, *ext; } LANG_EXT[] = {
    { "python", "py" },   { "py", "py" },        { "c", "c" },
    { "h", "h" },         { "cpp", "cpp" },      { "c++", "cpp" },
    { "cxx", "cpp" },     { "cc", "cpp" },       { "hpp", "hpp" },
    { "julia", "jl" },    { "jl", "jl" },        { "bash", "sh" },
    { "sh", "sh" },       { "shell", "sh" },     { "zsh", "sh" },
    { "console", "sh" },  { "javascript", "js" },{ "js", "js" },
    { "typescript", "ts" },{ "ts", "ts" },       { "json", "json" },
    { "html", "html" },   { "css", "css" },      { "rust", "rs" },
    { "rs", "rs" },       { "go", "go" },        { "java", "java" },
    { "kotlin", "kt" },   { "kt", "kt" },        { "r", "R" },
    { "matlab", "m" },    { "octave", "m" },     { "sql", "sql" },
    { "yaml", "yaml" },   { "yml", "yaml" },     { "toml", "toml" },
    { "xml", "xml" },     { "cmake", "cmake" },  { "markdown", "md" },
    { "md", "md" },       { "latex", "tex" },    { "tex", "tex" },
    { "ini", "ini" },     { "swift", "swift" },  { "ruby", "rb" },
    { "rb", "rb" },       { "perl", "pl" },      { "lua", "lua" },
    { "fortran", "f90" }, { "glsl", "glsl" },    { "diff", "diff" },
    { "patch", "patch" }, { "csv", "csv" },      { "powershell", "ps1" },
    { "ps1", "ps1" },
};

/* basename only, harmless characters only; NULL if nothing usable */
static char *safe_name(const char *s)
{
    char *b = g_path_get_basename(s), *p;

    for (p = b; *p; p++)
        if (!g_ascii_isalnum(*p) && !strchr("._-+", *p))
            *p = '_';
    if (!*b || !strcmp(b, ".") || !strcmp(b, "..")) {
        g_free(b);
        return NULL;
    }
    return b;
}

static gboolean is_name_key(const char *k)
{
    return !g_ascii_strcasecmp(k, "title") || !g_ascii_strcasecmp(k, "file") ||
           !g_ascii_strcasecmp(k, "filename") || !g_ascii_strcasecmp(k, "name");
}

/* ```python | ```{.python} | ```python:src/main.py | ```c title="x.c"
 * | ```main.c */
static void parse_info(const char *info, char **lang, char **name)
{
    char *s = g_strstrip(g_strdup(info));
    char **tok = g_strsplit_set(s, " \t", -1);
    guint i;

    *lang = NULL;
    *name = NULL;
    for (i = 0; tok[i]; i++) {
        char *t = tok[i], *eq = strchr(t, '=');
        gsize n;

        if (!*t)
            continue;
        if (eq) {
            char *v = eq + 1;

            *eq = '\0';
            if (*name || !is_name_key(t))
                continue;
            if (*v == '"' || *v == '\'')
                v++;
            n = strlen(v);
            if (n && (v[n - 1] == '"' || v[n - 1] == '\''))
                v[n - 1] = '\0';
            *name = safe_name(v);
        } else if (!*lang) {
            char *colon;

            while (*t == '{' || *t == '.')
                t++;
            n = strlen(t);
            if (n && t[n - 1] == '}')
                t[n - 1] = '\0';
            colon = strchr(t, ':');
            if (colon) {
                *colon = '\0';
                if (!*name && colon[1])
                    *name = safe_name(colon + 1);
            }
            if (*t)
                *lang = g_ascii_strdown(t, -1);
        }
    }
    if (*lang && strchr(*lang, '.') && !*name) {   /* ```main.c */
        char *ext = g_strdup(strrchr(*lang, '.') + 1);

        *name = safe_name(*lang);
        g_free(*lang);
        *lang = ext;
    }
    g_strfreev(tok);
    g_free(s);
}

/* CommonMark-style fences: 3+ backticks or tildes, indented <= 3 spaces,
 * closed by a run of the same character at least as long. An unclosed
 * block (reply stopped mid-stream) is kept. */
static GPtrArray *extract_code(const char *md)
{
    GPtrArray *out = g_ptr_array_new_with_free_func(code_block_free);
    const char *p = md ? md : "";
    GString *body = NULL;
    char fch = 0, *lang = NULL, *name = NULL;
    gsize flen = 0;

    while (*p) {
        const char *eol = strchr(p, '\n');
        gsize llen = eol ? (gsize)(eol - p) : strlen(p);
        gsize i = 0, run = 0;
        char c;

        while (i < 3 && i < llen && p[i] == ' ')
            i++;
        c = (i < llen) ? p[i] : '\0';
        if (c == '`' || c == '~')
            while (i + run < llen && p[i + run] == c)
                run++;

        if (!body) {
            if (run >= 3) {
                char *info = g_strndup(p + i + run, llen - i - run);

                if (!(c == '`' && strchr(info, '`'))) {   /* not inline */
                    parse_info(info, &lang, &name);
                    fch = c;
                    flen = run;
                    body = g_string_new(NULL);
                }
                g_free(info);
            }
        } else {
            gsize j = i + run;

            while (j < llen && (p[j] == ' ' || p[j] == '\t' || p[j] == '\r'))
                j++;
            if (c == fch && run >= flen && j == llen) {   /* closing fence */
                CodeBlock *b = g_new0(CodeBlock, 1);

                b->lang = lang;
                b->name = name;
                b->code = g_string_free(body, FALSE);
                lang = name = NULL;
                body = NULL;
                if (*b->code)
                    g_ptr_array_add(out, b);
                else
                    code_block_free(b);
            } else {
                g_string_append_len(body, p, (gssize)llen);
                g_string_append_c(body, '\n');
            }
        }
        if (!eol)
            break;
        p = eol + 1;
    }
    if (body && body->len) {
        CodeBlock *b = g_new0(CodeBlock, 1);

        b->lang = lang;
        b->name = name;
        b->code = g_string_free(body, FALSE);
        g_ptr_array_add(out, b);
    } else {
        if (body)
            g_string_free(body, TRUE);
        g_free(lang);
        g_free(name);
    }
    return out;
}

static char *code_file_name(const CodeBlock *b, guint idx, guint total)
{
    const char *ext = "txt";
    guint i;

    if (b->name)
        return g_strdup(b->name);
    if (b->lang) {
        if (!strcmp(b->lang, "make") || !strcmp(b->lang, "makefile"))
            return g_strdup("Makefile");
        if (!strcmp(b->lang, "dockerfile"))
            return g_strdup("Dockerfile");
        for (i = 0; i < G_N_ELEMENTS(LANG_EXT); i++)
            if (!strcmp(LANG_EXT[i].lang, b->lang)) {
                ext = LANG_EXT[i].ext;
                break;
            }
    }
    return total > 1 ? g_strdup_printf("snippet-%02u.%s", idx + 1, ext)
                     : g_strdup_printf("snippet.%s", ext);
}

/* code blocks of the most recent assistant reply that has any */
static GPtrArray *last_code_blocks(App *app)
{
    guint i = app->history->len;

    while (i-- > 0) {
        Msg *m = g_ptr_array_index(app->history, i);
        GPtrArray *b;

        if (g_strcmp0(m->role, "assistant"))
            continue;
        b = extract_code(m->text);
        if (b->len)
            return b;
        g_ptr_array_unref(b);
    }
    return NULL;
}

/* ================================================================== */
/*  file dialogs: attach, save chat, save code                        */
/* ================================================================== */

/* TRUE when the dialog produced nothing; silent if the user cancelled */
static gboolean dialog_aborted(App *app, GError *err)
{
    if (!err)
        return app->closing;
    if (!app->closing &&
        !g_error_matches(err, GTK_DIALOG_ERROR, GTK_DIALOG_ERROR_DISMISSED) &&
        !g_error_matches(err, GTK_DIALOG_ERROR, GTK_DIALOG_ERROR_CANCELLED))
        show_error(app, err->message);
    g_error_free(err);
    return TRUE;
}

static gboolean write_text(GFile *f, const char *data, char **why)
{
    GError *err = NULL;

    if (g_file_replace_contents(f, data, strlen(data), NULL, FALSE,
                                G_FILE_CREATE_NONE, NULL, NULL, &err))
        return TRUE;
    *why = g_strdup(err->message);
    g_error_free(err);
    return FALSE;
}

static void attach_done(GObject *src, GAsyncResult *res, gpointer data)
{
    App *app = data;
    GError *err = NULL;
    GListModel *files = gtk_file_dialog_open_multiple_finish(
                            GTK_FILE_DIALOG(src), res, &err);
    GString *bad;
    guint i, n, added = 0;

    if (dialog_aborted(app, err) || !files) {
        if (files)
            g_object_unref(files);
        return;
    }
    bad = g_string_new(NULL);
    n = g_list_model_get_n_items(files);
    for (i = 0; i < n; i++) {
        GFile *f = g_list_model_get_item(files, i);
        char *why = NULL;
        Attachment *a = attachment_load(f, &why);

        if (a) {
            g_ptr_array_add(app->att, a);
            added++;
        } else {
            g_string_append_printf(bad, "%s\n", why);
            g_free(why);
        }
        g_object_unref(f);
    }
    g_object_unref(files);
    att_refresh(app);
    set_status(app, "%u file%s attached", added, added == 1 ? "" : "s");
    if (bad->len)
        show_error(app, bad->str);
    g_string_free(bad, TRUE);
}

static void add_filter(GListStore *s, const char *label,
                       const char *mime, const char *pattern)
{
    GtkFileFilter *f = gtk_file_filter_new();

    gtk_file_filter_set_name(f, label);
    if (mime)
        gtk_file_filter_add_mime_type(f, mime);
    if (pattern)
        gtk_file_filter_add_pattern(f, pattern);
    g_list_store_append(s, f);
    g_object_unref(f);
}

static void on_attach(GtkButton *b, gpointer data)
{
    App *app = data;
    GtkFileDialog *d = gtk_file_dialog_new();
    GListStore *flt = g_list_store_new(GTK_TYPE_FILE_FILTER);

    (void)b;
    add_filter(flt, "All files", NULL, "*");
    add_filter(flt, "Images", "image/*", NULL);
    add_filter(flt, "Text", "text/*", NULL);
    gtk_file_dialog_set_title(d, "Attach images or text files");
    gtk_file_dialog_set_filters(d, G_LIST_MODEL(flt));
    g_object_unref(flt);
    gtk_file_dialog_open_multiple(d, app->win, NULL, attach_done, app);
    g_object_unref(d);                     /* the async task holds a ref */
}

static void on_att_clear(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    g_ptr_array_set_size(app->att, 0);
    att_refresh(app);
}

static char *chat_markdown(App *app)
{
    GString *s = g_string_new(NULL);
    GDateTime *now = g_date_time_new_now_local();
    char *ts = g_date_time_format(now, "%Y-%m-%d %H:%M");
    guint i;

    g_string_append_printf(s, "# Chat transcript\n\n*%s · %s · %s*\n",
        prov_get(app->prov_id)->label, current_model(app), ts);
    for (i = 0; i < app->history->len; i++) {
        Msg *m = g_ptr_array_index(app->history, i);
        const char *who = !g_strcmp0(m->role, "system")    ? "System"
                        : !g_strcmp0(m->role, "assistant") ? "Assistant"
                                                           : "User";

        g_string_append_printf(s, "\n## %s\n\n%s\n", who, m->text ? m->text : "");
        if (m->images && m->images->len)
            g_string_append_printf(s, "\n*(%u image%s attached)*\n",
                                   m->images->len,
                                   m->images->len == 1 ? "" : "s");
    }
    g_free(ts);
    g_date_time_unref(now);
    return g_string_free(s, FALSE);
}

/* canonical history, images included: a faithful archive */
static char *chat_json(App *app)
{
    JsonObject *root = json_object_new();
    JsonNode *node = json_node_new(JSON_NODE_OBJECT);
    GDateTime *now = g_date_time_new_now_local();
    char *ts = g_date_time_format_iso8601(now), *out;

    json_object_set_string_member(root, "provider",
                                  prov_get(app->prov_id)->id_name);
    json_object_set_string_member(root, "model", current_model(app));
    json_object_set_string_member(root, "saved", ts);
    json_object_set_array_member(root, "messages",
                                 ollama_messages(app->history));
    json_node_take_object(node, root);
    out = node_to_string(node, TRUE);
    json_node_unref(node);
    g_free(ts);
    g_date_time_unref(now);
    return out;
}

static void save_chat_done(GObject *src, GAsyncResult *res, gpointer data)
{
    App *app = data;
    GError *err = NULL;
    GFile *f = gtk_file_dialog_save_finish(GTK_FILE_DIALOG(src), res, &err);
    char *base, *lower, *body, *why = NULL;

    if (dialog_aborted(app, err) || !f) {
        if (f)
            g_object_unref(f);
        return;
    }
    base = g_file_get_basename(f);
    lower = g_ascii_strdown(base, -1);
    body = g_str_has_suffix(lower, ".json") ? chat_json(app)
                                            : chat_markdown(app);
    if (write_text(f, body, &why)) {
        set_status(app, "chat saved: %s", base);
    } else {
        show_error(app, why);
        g_free(why);
    }
    g_free(body);
    g_free(lower);
    g_free(base);
    g_object_unref(f);
}

static void on_save_chat(GtkButton *b, gpointer data)
{
    App *app = data;
    GtkFileDialog *d;
    GDateTime *now;
    char *name;
    guint i, turns = 0;

    (void)b;
    sync_system(app);
    for (i = 0; i < app->history->len; i++)
        if (g_strcmp0(((Msg *)g_ptr_array_index(app->history, i))->role,
                      "system"))
            turns++;
    if (!turns) {
        set_status(app, "nothing to save yet");
        return;
    }
    now = g_date_time_new_now_local();
    name = g_date_time_format(now, "chat-%Y%m%d-%H%M.md");
    d = gtk_file_dialog_new();
    gtk_file_dialog_set_title(d, "Save chat (.md = Markdown, .json = full history)");
    gtk_file_dialog_set_initial_name(d, name);
    gtk_file_dialog_save(d, app->win, NULL, save_chat_done, app);
    g_object_unref(d);
    g_free(name);
    g_date_time_unref(now);
}

typedef struct {
    App       *app;
    GPtrArray *blocks;    /* CodeBlock* */
} CodeJob;

static void code_job_free(CodeJob *j)
{
    g_ptr_array_unref(j->blocks);
    g_free(j);
}

/* scripts with a shebang are made executable */
static gboolean write_code(GFile *f, const CodeBlock *b, char **why)
{
    char *path;

    if (!write_text(f, b->code, why))
        return FALSE;
    if (g_str_has_prefix(b->code, "#!") && (path = g_file_get_path(f))) {
        g_chmod(path, 0755);
        g_free(path);
    }
    return TRUE;
}

/* name.ext, name-2.ext, name-3.ext, ... : never overwrite in folder mode */
static GFile *unique_child(GFile *dir, const char *name)
{
    GFile *f = g_file_get_child(dir, name);
    const char *dot = strrchr(name, '.');
    int stem = (dot && dot != name) ? (int)(dot - name) : (int)strlen(name);
    guint k;

    for (k = 2; k < 1000 && g_file_query_exists(f, NULL); k++) {
        char *alt = g_strdup_printf("%.*s-%u%s", stem, name, k, name + stem);

        g_object_unref(f);
        f = g_file_get_child(dir, alt);
        g_free(alt);
    }
    return f;
}

static void save_code_file_done(GObject *src, GAsyncResult *res, gpointer data)
{
    CodeJob *job = data;
    App *app = job->app;
    GError *err = NULL;
    GFile *f = gtk_file_dialog_save_finish(GTK_FILE_DIALOG(src), res, &err);
    char *why = NULL, *base;

    if (dialog_aborted(app, err) || !f) {
        if (f)
            g_object_unref(f);
        code_job_free(job);
        return;
    }
    base = g_file_get_basename(f);
    if (write_code(f, g_ptr_array_index(job->blocks, 0), &why)) {
        set_status(app, "code saved: %s", base);
    } else {
        show_error(app, why);
        g_free(why);
    }
    g_free(base);
    g_object_unref(f);
    code_job_free(job);
}

static void save_code_dir_done(GObject *src, GAsyncResult *res, gpointer data)
{
    CodeJob *job = data;
    App *app = job->app;
    GError *err = NULL;
    GFile *dir = gtk_file_dialog_select_folder_finish(GTK_FILE_DIALOG(src),
                                                      res, &err);
    GString *bad;
    char *dname;
    guint i, ok = 0;

    if (dialog_aborted(app, err) || !dir) {
        if (dir)
            g_object_unref(dir);
        code_job_free(job);
        return;
    }
    bad = g_string_new(NULL);
    for (i = 0; i < job->blocks->len; i++) {
        CodeBlock *b = g_ptr_array_index(job->blocks, i);
        char *name = code_file_name(b, i, job->blocks->len), *why = NULL;
        GFile *f = unique_child(dir, name);

        if (write_code(f, b, &why)) {
            ok++;
        } else {
            g_string_append_printf(bad, "%s: %s\n", name, why);
            g_free(why);
        }
        g_object_unref(f);
        g_free(name);
    }
    dname = g_file_get_basename(dir);
    set_status(app, "%u of %u code files saved in %s",
               ok, job->blocks->len, dname);
    if (bad->len)
        show_error(app, bad->str);
    g_free(dname);
    g_string_free(bad, TRUE);
    g_object_unref(dir);
    code_job_free(job);
}

static void on_save_code(GtkButton *b, gpointer data)
{
    App *app = data;
    GPtrArray *blocks = last_code_blocks(app);
    GtkFileDialog *d;
    CodeJob *job;

    (void)b;
    if (!blocks) {
        set_status(app, "no code blocks in the replies");
        gtk_widget_set_sensitive(app->code_btn, FALSE);
        return;
    }
    job = g_new0(CodeJob, 1);
    job->app = app;
    job->blocks = blocks;
    d = gtk_file_dialog_new();
    if (blocks->len == 1) {
        char *name = code_file_name(g_ptr_array_index(blocks, 0), 0, 1);

        gtk_file_dialog_set_title(d, "Save code");
        gtk_file_dialog_set_initial_name(d, name);
        gtk_file_dialog_save(d, app->win, NULL, save_code_file_done, job);
        g_free(name);
    } else {
        char *title = g_strdup_printf("Folder for %u code files", blocks->len);

        gtk_file_dialog_set_title(d, title);
        gtk_file_dialog_select_folder(d, app->win, NULL,
                                      save_code_dir_done, job);
        g_free(title);
    }
    g_object_unref(d);
}

/* ================================================================== */
/*  worker thread: streaming / non-streaming request                  */
/* ================================================================== */

static void worker_unref(Worker *wd)
{
    if (!g_atomic_int_dec_and_test(&wd->refs))
        return;
    prov_parser_free(wd->parser);
    prov_request_free(wd->req);
    g_string_free(wd->pending, TRUE);
    g_string_free(wd->full, TRUE);
    g_string_free(wd->line_txt, TRUE);
    g_string_free(wd->line_thk, TRUE);
    g_string_free(wd->q_text, TRUE);
    g_string_free(wd->q_think, TRUE);
    g_mutex_clear(&wd->q_lock);
    g_free(wd->error);
    g_free(wd->qkey);
    g_free(wd);
}

static Worker *worker_new(App *app, ProvRequest *req, ProvId prov,
                          gboolean stream)
{
    Worker *wd = g_new0(Worker, 1);

    wd->app = app;
    wd->req = req;
    wd->prov = prov;
    wd->stream = stream;
    wd->refs = 1;                          /* dropped by worker_done_idle */
    wd->parser = prov_parser_new(prov);
    wd->pending = g_string_new(NULL);
    wd->full = g_string_new(NULL);
    wd->line_txt = g_string_new(NULL);
    wd->line_thk = g_string_new(NULL);
    wd->q_text = g_string_new(NULL);
    wd->q_think = g_string_new(NULL);
    g_mutex_init(&wd->q_lock);
    return wd;
}

/* main thread: render everything queued since the last flush */
static void worker_drain(Worker *wd)
{
    App *app = wd->app;
    char *think = NULL, *text = NULL;

    g_mutex_lock(&wd->q_lock);
    if (wd->q_think->len) {
        think = g_strndup(wd->q_think->str, wd->q_think->len);
        g_string_truncate(wd->q_think, 0);
    }
    if (wd->q_text->len) {
        text = g_strndup(wd->q_text->str, wd->q_text->len);
        g_string_truncate(wd->q_text, 0);
    }
    wd->q_scheduled = FALSE;
    g_mutex_unlock(&wd->q_lock);

    if ((think || text) && !app->closing) {
        if (!app->cur_box)
            asst_begin(app);
        if (think) {
            if (!app->cur_think_buf)
                asst_think(app);
            buf_append(app->cur_think_buf, think);
        }
        if (text) {
            if (!app->cur_text_buf)
                asst_text(app);
            buf_append(app->cur_text_buf, text);
            g_string_append(app->reply_acc, text);
        }
    }
    g_free(think);
    g_free(text);
}

static gboolean flush_idle(gpointer data)
{
    Worker *wd = data;

    worker_drain(wd);
    worker_unref(wd);
    return G_SOURCE_REMOVE;
}

/* worker thread: queue a delta; at most one flush idle is ever pending */
static void push_delta(Worker *wd, const GString *think, const GString *text)
{
    gboolean schedule;

    g_mutex_lock(&wd->q_lock);
    if (think && think->len)
        g_string_append_len(wd->q_think, think->str, (gssize)think->len);
    if (text && text->len)
        g_string_append_len(wd->q_text, text->str, (gssize)text->len);
    schedule = !wd->q_scheduled;
    wd->q_scheduled = TRUE;
    g_mutex_unlock(&wd->q_lock);
    if (schedule) {
        g_atomic_int_inc(&wd->refs);
        g_idle_add(flush_idle, wd);
    }
}

static void start_request(App *app, guint retries);

/* The finished reply goes into the history. A continuation is merged
 * into the assistant turn it extends and the nudge that asked for it
 * is dropped, so the history (and "Save chat"/"Save code") see one
 * uninterrupted reply. */
static void history_absorb_reply(App *app)
{
    guint n = app->history->len;
    Msg *last = n ? g_ptr_array_index(app->history, n - 1) : NULL;

    if (app->continuing && last && !g_strcmp0(last->role, "user") &&
        !g_strcmp0(last->text, CONTINUE_PROMPT)) {
        g_ptr_array_remove_index(app->history, n - 1);
        n--;
        last = n ? g_ptr_array_index(app->history, n - 1) : NULL;
        if (last && !g_strcmp0(last->role, "assistant")) {
            char *joined = g_strconcat(last->text, app->reply_acc->str, NULL);

            g_free(last->text);
            last->text = joined;
            return;
        }
    }
    g_ptr_array_add(app->history, msg_new("assistant", app->reply_acc->str));
}

static gboolean worker_done_idle(gpointer data)
{
    Worker *wd = data;
    App *app = wd->app;
    gboolean was_cont, truncated, cont_on;

    if (wd->thread) {                      /* it is returning right now */
        g_thread_join(wd->thread);
        wd->thread = NULL;
    }
    worker_drain(wd);                      /* whatever is still queued */
    if (app->worker == wd)
        app->worker = NULL;
    app->busy = FALSE;
    if (app->closing) {
        worker_unref(wd);
        return G_SOURCE_REMOVE;
    }
    was_cont = app->continuing;

    /* the API refused a parameter or the output-token value: remember
     * that for this model and resend the same turn (each pass must learn
     * something; the retry counter bounds the loop) */
    if (wd->error && !wd->cancelled && !app->reply_acc->len &&
        (wd->http_code == 400 || wd->http_code == 422) &&
        wd->retries < MAX_ADAPT_RETRIES &&
        gtk_check_button_get_active(GTK_CHECK_BUTTON(app->adapt_cb))) {
        int cap = rejection_cap(wd->error, wd->req->max_sent);
        char *note = NULL;

        if (cap > 0) {
            note = g_strdup_printf("⟳ adapting request (output limit %d "
                                   "tokens for this model): %.160s",
                                   cap, wd->error);
            caps_learn(app, wd->qkey, (guint)cap);
        } else {
            guint add = classify_rejection(wd->prov, wd->error,
                                           wd->req->sent, wd->quirks);

            if (add) {
                char *names = quirk_names(add);

                note = g_strdup_printf("⟳ adapting request (%s): %.160s",
                                       names, wd->error);
                quirks_learn(app, wd->qkey, add);
                g_free(names);
            }
        }
        if (note) {
            chat_note(app, note, FALSE);
            g_free(note);
            start_request(app, wd->retries + 1);
            worker_unref(wd);
            return G_SOURCE_REMOVE;
        }
    }

    if (app->reply_acc->len) {
        Msg *reply;

        history_absorb_reply(app);
        reply = g_ptr_array_index(app->history, app->history->len - 1);
        if (!gtk_widget_get_sensitive(app->code_btn)) {
            GPtrArray *cb = extract_code(reply->text);

            gtk_widget_set_sensitive(app->code_btn, cb->len > 0);
            g_ptr_array_unref(cb);
        }
        app->requests++;
        if (wd->in_tok > 0)
            app->total_in += wd->in_tok;
        if (wd->out_tok > 0)
            app->total_out += wd->out_tok;
    } else if (app->history->len &&
               !g_strcmp0(((Msg *)g_ptr_array_index(app->history,
                                app->history->len - 1))->role, "user")) {
        /* the failed turn — or the continuation nudge, which leaves the
         * partial reply in place */
        g_ptr_array_remove_index(app->history, app->history->len - 1);
    }

    if (app->cur_text_buf)
        buf_rstrip(app->cur_text_buf);
    if (app->cur_think_buf)
        buf_rstrip(app->cur_think_buf);

    /* cut off by the token limit: keep going in the same bubble */
    truncated = wd->truncated && !wd->error && !wd->cancelled &&
                app->reply_acc->len > 0;
    cont_on = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->cont_cb));
    if (truncated && cont_on && app->cur_box &&
        app->cont_count < MAX_CONTINUES) {
        app->cont_count++;
        asst_continue_marker(app);
        g_ptr_array_add(app->history, msg_new("user", CONTINUE_PROMPT));
        app->continuing = TRUE;
        start_request(app, 0);
        set_status(app, "token limit reached — continuing (%u of %u)…",
                   app->cont_count, MAX_CONTINUES);
        worker_unref(wd);
        return G_SOURCE_REMOVE;
    }

    gtk_spinner_stop(GTK_SPINNER(app->spinner));
    gtk_widget_set_sensitive(app->send_btn, TRUE);
    gtk_widget_set_sensitive(app->stop_btn, FALSE);
    app->continuing = FALSE;
    app->cur_box = NULL;
    app->cur_anchor = NULL;
    app->cur_think_buf = app->cur_text_buf = NULL;

    if (wd->error) {
        char *note = g_strdup_printf(
            app->reply_acc->len ? "✕ %s"
            : was_cont ? "✕ %s — the continuation failed; the partial reply "
                         "was kept"
                       : "✕ %s — this turn was not kept in the history",
            wd->error);

        chat_note(app, note, TRUE);
        g_free(note);
        set_status(app, "failed: %.100s", wd->error);
    } else if (wd->cancelled) {
        chat_note(app, "stopped", FALSE);
        set_status(app, "stopped");
    } else if (app->reply_acc->len) {
        if (truncated) {
            char *note = cont_on
                ? g_strdup_printf("✂ still cut off by the token limit after "
                                  "%u continuations", app->cont_count)
                : g_strdup("✂ reply cut off by the token limit — raise "
                           "Max tokens or enable Auto-continue");

            chat_note(app, note, TRUE);
            g_free(note);
        }
        set_status(app,
            "done%s · in %lld / out %lld tok · totals %lld/%lld · replies %lld",
            app->cont_count ? " (continued)" : "",
            (long long)wd->in_tok, (long long)wd->out_tok,
            (long long)app->total_in, (long long)app->total_out,
            (long long)app->requests);
    } else {
        set_status(app, "empty reply");
    }
    worker_unref(wd);
    return G_SOURCE_REMOVE;
}

static void worker_feed_line(Worker *wd, const char *line)
{
    g_string_truncate(wd->line_txt, 0);
    g_string_truncate(wd->line_thk, 0);
    if (!prov_parser_feed(wd->parser, line, wd->line_txt, wd->line_thk)) {
        g_free(wd->error);
        wd->error = g_strdup(prov_parser_error(wd->parser));
    } else if (wd->line_txt->len || wd->line_thk->len) {
        push_delta(wd, wd->line_thk, wd->line_txt);
    }
}

static size_t worker_write(char *ptr, size_t size, size_t nmemb, void *ud)
{
    Worker *wd = ud;
    gsize total = size * nmemb, start = 0;
    char *nl;

    if (g_atomic_int_get(&wd->cancel)) {
        wd->cancelled = TRUE;
        return 0;                          /* abort the transfer */
    }
    if (!wd->stream) {                     /* parsed as a whole at the end */
        g_string_append_len(wd->full, ptr, (gssize)total);
        return total;
    }
    if (wd->full->len < ERR_BODY_CAP)      /* bounded copy for error reports */
        g_string_append_len(wd->full, ptr,
                            (gssize)MIN(total, ERR_BODY_CAP - wd->full->len));

    /* split complete lines in place; shift the buffer once per chunk */
    g_string_append_len(wd->pending, ptr, (gssize)total);
    while (!wd->error &&
           (nl = memchr(wd->pending->str + start, '\n',
                        wd->pending->len - start)) != NULL) {
        gsize len = (gsize)(nl - (wd->pending->str + start));

        *nl = '\0';
        if (len && nl[-1] == '\r')
            nl[-1] = '\0';
        worker_feed_line(wd, wd->pending->str + start);
        start += len + 1;
    }
    if (start)
        g_string_erase(wd->pending, 0, (gssize)start);
    return wd->error ? 0 : total;          /* surface the error, stop reading */
}

/* lets Stop abort promptly even while no bytes are arriving */
static int worker_progress(void *ud, curl_off_t dltotal, curl_off_t dlnow,
                           curl_off_t ultotal, curl_off_t ulnow)
{
    Worker *wd = ud;

    (void)dltotal;
    (void)dlnow;
    (void)ultotal;
    (void)ulnow;
    if (g_atomic_int_get(&wd->cancel)) {
        wd->cancelled = TRUE;
        return 1;                          /* CURLE_ABORTED_BY_CALLBACK */
    }
    return 0;
}

static gpointer worker_thread(gpointer data)
{
    Worker *wd = data;
    struct curl_slist *hl = NULL;
    CURL *c;
    CURLcode rc;
    long code = 0;
    char **h;

    c = curl_easy_init();
    if (!c) {
        wd->error = g_strdup("libcurl initialisation failed");
        goto finish;
    }
    for (h = wd->req->headers; h && *h; h++)
        hl = curl_slist_append(hl, *h);
    curl_easy_setopt(c, CURLOPT_URL, wd->req->url);
    if (hl)
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, hl);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, wd->req->body);
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE,
                     (curl_off_t)strlen(wd->req->body));
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, worker_write);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, wd);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, worker_progress);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, wd);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(c, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "ollama-gui/1.1");
    rc = curl_easy_perform(c);
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    wd->http_code = code;
    curl_slist_free_all(hl);
    curl_easy_cleanup(c);

    /* a final stream line without a trailing newline */
    if (wd->stream && rc == CURLE_OK && code < 400 && !wd->error &&
        wd->pending->len)
        worker_feed_line(wd, wd->pending->str);

    if (!wd->error && !wd->cancelled) {
        if (rc != CURLE_OK) {
            wd->error = g_strdup_printf("network error: %s",
                                        curl_easy_strerror(rc));
        } else if (code >= 400) {
            wd->error = body_error(wd->full->str, wd->full->len);
            if (!wd->error)
                wd->error = wd->full->len
                    ? g_strdup_printf("HTTP %ld: %.300s", code, wd->full->str)
                    : g_strdup_printf("HTTP %ld (empty body)", code);
        } else if (!wd->stream) {
            GString *txt = g_string_new(NULL);
            GString *thk = g_string_new(NULL);
            char *err = NULL;

            if (prov_parse_body(wd->prov, wd->full->str, wd->full->len,
                                txt, thk, &wd->in_tok, &wd->out_tok,
                                &wd->truncated, &err))
                push_delta(wd, thk, txt);
            else
                wd->error = err;
            g_string_free(txt, TRUE);
            g_string_free(thk, TRUE);
        } else {
            prov_parser_usage(wd->parser, &wd->in_tok, &wd->out_tok);
            wd->truncated = prov_parser_truncated(wd->parser);
        }
    }
finish:
    g_idle_add(worker_done_idle, wd);
    return NULL;
}

/* ================================================================== */
/*  model-list loading                                                */
/* ================================================================== */

static size_t models_write_cb(char *ptr, size_t s, size_t n, void *ud)
{
    g_string_append_len((GString *)ud, ptr, s * n);
    return s * n;
}

static int models_progress(void *ud, curl_off_t dltotal, curl_off_t dlnow,
                           curl_off_t ultotal, curl_off_t ulnow)
{
    ModelsJob *j = ud;

    (void)dltotal;
    (void)dlnow;
    (void)ultotal;
    (void)ulnow;
    return g_atomic_int_get(&j->cancel) ? 1 : 0;
}

static gboolean models_done_idle(gpointer data)
{
    ModelsJob *j = data;
    App *app = j->app;

    if (j->thread) {
        g_thread_join(j->thread);
        j->thread = NULL;
    }
    if (app->models_job == j)
        app->models_job = NULL;

    if (app->closing) {
        /* window already destroyed: just free the job */
    } else if (j->err) {
        show_error(app, j->err);
        set_status(app, "model list failed (%s)", prov_get(j->id)->label);
    } else if (j->names && *j->names) {
        guint cnt = g_list_model_get_n_items(G_LIST_MODEL(app->model_list));
        guint sel = GTK_INVALID_LIST_POSITION, k = 0;
        char *keep = g_strdup(current_model(app));
        char **n;

        /* refill without touching the model the user typed; re-select
         * it if it is in the list, fill the entry only when it is empty */
        app->quiet = TRUE;
        g_ptr_array_set_size(app->models, 0);
        if (cnt)
            gtk_string_list_splice(app->model_list, 0, cnt, NULL);
        for (n = j->names; *n; n++, k++) {
            g_ptr_array_add(app->models, g_strdup(*n));
            gtk_string_list_append(app->model_list, *n);
            if (!strcmp(*n, keep))
                sel = k;
        }
        app->quiet = FALSE;
        if (sel != GTK_INVALID_LIST_POSITION) {
            gtk_drop_down_set_selected(GTK_DROP_DOWN(app->model_dd), sel);
        } else if (!*keep) {
            gtk_drop_down_set_selected(GTK_DROP_DOWN(app->model_dd), 0);
            gtk_editable_set_text(GTK_EDITABLE(app->model_entry), j->names[0]);
        }
        g_free(keep);
        set_status(app, "%u models loaded (%s)",
                   app->models->len, prov_get(j->id)->label);
    } else {
        set_status(app, "no models returned");
    }
    g_strfreev(j->names);
    g_strfreev(j->headers);
    g_free(j->url);
    g_free(j->err);
    g_free(j);
    return G_SOURCE_REMOVE;
}

static gpointer models_thread(gpointer data)
{
    ModelsJob *j = data;
    GString *body = g_string_new(NULL);
    CURL *c = curl_easy_init();
    struct curl_slist *h = NULL;
    char **hh;
    long code = 0;
    CURLcode rc;

    if (!c) {
        j->err = g_strdup("curl init failed");
        goto out;
    }
    for (hh = j->headers; hh && *hh; hh++)
        h = curl_slist_append(h, *hh);
    curl_easy_setopt(c, CURLOPT_URL, j->url);
    if (h)
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, models_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, body);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, models_progress);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, j);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 30L);
    rc = curl_easy_perform(c);
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    if (rc != CURLE_OK)
        j->err = g_strdup_printf("network error: %s", curl_easy_strerror(rc));
    else if (code == 200)
        j->names = prov_parse_models(j->id, body->str, body->len);
    else if (!(j->err = body_error(body->str, body->len)))
        j->err = g_strdup_printf("HTTP %ld: %.200s", code, body->str);
    curl_slist_free_all(h);
    curl_easy_cleanup(c);
out:
    g_string_free(body, TRUE);
    g_idle_add(models_done_idle, j);
    return NULL;
}

/* ================================================================== */
/*  signal handlers                                                   */
/* ================================================================== */

static void on_load_models(GtkButton *b, gpointer data)
{
    App *app = data;
    ProvListReq *r;
    ModelsJob *j;

    (void)b;
    if (app->models_job) {
        set_status(app, "model list is already loading…");
        return;
    }
    r = prov_list_models(app->prov_id,
            gtk_editable_get_text(GTK_EDITABLE(app->host_e)),
            gtk_editable_get_text(GTK_EDITABLE(app->key_e)));
    j = g_new0(ModelsJob, 1);
    j->app = app;
    j->id = app->prov_id;
    j->url = g_strdup(r->url);
    j->headers = r->headers ? g_strdupv(r->headers) : NULL;
    prov_list_req_free(r);
    app->models_job = j;
    set_status(app, "loading model list…");
    j->thread = g_thread_new("prov-models", models_thread, j);
}

static void on_preview(GtkButton *b, gpointer data)
{
    App *app = data;
    char *input = buffer_text(app->input_buf);
    GPtrArray *hist = g_ptr_array_new_with_free_func(msg_free);
    ProvRequest *r;
    GString *s = g_string_new(NULL);
    char **h;
    guint i;

    (void)b;
    sync_system(app);
    for (i = 0; i < app->history->len; i++)
        g_ptr_array_add(hist, preview_copy(g_ptr_array_index(app->history, i)));
    if ((input && *input) || app->att->len) {
        Msg *next = compose_user_msg(app->att, input);

        g_ptr_array_add(hist, preview_copy(next));
        msg_free(next);
    }
    {
        const char *model = current_model(app);
        GenParams gp;

        gen_params(app, model, &gp);
        r = prov_build_request(app->prov_id,
                gtk_editable_get_text(GTK_EDITABLE(app->host_e)), model,
                gtk_editable_get_text(GTK_EDITABLE(app->key_e)),
                gtk_check_button_get_active(GTK_CHECK_BUTTON(app->stream_cb)),
                &gp, hist);
    }
    g_ptr_array_unref(hist);

    g_string_append_printf(s, "POST %s\n", r->url);
    for (h = r->headers; h && *h; h++) {          /* mask secrets */
        const char *colon = strchr(*h, ':');

        if (colon && (g_ascii_strncasecmp(*h, "Authorization:", 14) == 0 ||
                      strstr(*h, "api-key")))
            g_string_append_printf(s, "%.*s: ***\n", (int)(colon - *h), *h);
        else
            g_string_append_printf(s, "%s\n", *h);
    }
    g_string_append_c(s, '\n');
    {
        JsonParser *p = json_parser_new();

        if (json_parser_load_from_data(p, r->body, -1, NULL)) {
            char *pretty = node_to_string(json_parser_get_root(p), TRUE);
            g_string_append(s, pretty);
            g_free(pretty);
        } else {
            g_string_append(s, r->body);
        }
        g_object_unref(p);
    }
    show_text_window(app, "Request preview", s->str);
    g_string_free(s, TRUE);
    g_free(input);
    prov_request_free(r);
}

/* build the request from the current history and settings, and run it;
 * bubble state (cur_box & co.) is set up by the caller */
static void start_request(App *app, guint retries)
{
    const char *model = current_model(app);
    gboolean stream = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->stream_cb));
    GenParams gp;
    ProvRequest *req;
    Worker *wd;

    gen_params(app, model, &gp);
    req = prov_build_request(app->prov_id,
            gtk_editable_get_text(GTK_EDITABLE(app->host_e)),
            model, app->keys[app->prov_id], stream, &gp, app->history);

    wd = worker_new(app, req, app->prov_id, stream);
    wd->qkey = quirk_key(app->prov_id, model);
    wd->quirks = gp.quirks;
    wd->retries = retries;
    app->worker = wd;
    app->busy = TRUE;
    g_string_truncate(app->reply_acc, 0);

    gtk_spinner_start(GTK_SPINNER(app->spinner));
    gtk_widget_set_sensitive(app->send_btn, FALSE);
    gtk_widget_set_sensitive(app->stop_btn, TRUE);
    set_status(app, app->continuing ? "continuing…"
               : stream ? "streaming…" : "waiting for reply…");
    wd->thread = g_thread_new("chat", worker_thread, wd);
}

static void on_send(GtkButton *btn, gpointer data)
{
    App *app = data;
    char *input = buffer_text(app->input_buf);
    const char *model = current_model(app);

    (void)btn;
    if (app->busy || !input || (!*input && app->att->len == 0)) {
        g_free(input);
        return;
    }
    if (!model || !*model) {
        show_error(app, "Pick or type a model name first.");
        g_free(input);
        return;
    }

    /* remember host/key for this provider (in-memory, maybe on disk) */
    g_free(app->hosts[app->prov_id]);
    app->hosts[app->prov_id] =
        g_strdup(gtk_editable_get_text(GTK_EDITABLE(app->host_e)));
    g_free(app->keys[app->prov_id]);
    app->keys[app->prov_id] =
        g_strdup(gtk_editable_get_text(GTK_EDITABLE(app->key_e)));

    sync_system(app);

    /* transcript + canonical history */
    chat_add_user(app, input, app->att);
    g_ptr_array_add(app->history, compose_user_msg(app->att, input));
    g_ptr_array_set_size(app->att, 0);     /* attachments go with this turn */
    att_refresh(app);
    gtk_text_buffer_set_text(app->input_buf, "", -1);
    g_free(input);

    app->cur_box = NULL;
    app->cur_anchor = NULL;
    app->cur_think_buf = app->cur_text_buf = NULL;
    app->cont_count = 0;
    app->continuing = FALSE;
    config_save(app);
    start_request(app, 0);
}

static void on_stop(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    if (app->worker) {
        g_atomic_int_set(&app->worker->cancel, 1);
        set_status(app, "stopping…");
    }
}

static gboolean on_input_key(GtkEventControllerKey *c, guint keyval,
                             guint keycode, GdkModifierType state, gpointer data)
{
    (void)c;
    (void)keycode;
    if ((state & GDK_CONTROL_MASK) &&
        (keyval == GDK_KEY_Return || keyval == GDK_KEY_KP_Enter)) {
        on_send(NULL, data);
        return TRUE;
    }
    return FALSE;
}

static void on_new_chat(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    if (app->busy)
        return;
    g_ptr_array_set_size(app->history, 0);
    chat_clear(app);
    gtk_widget_set_sensitive(app->code_btn, FALSE);
    g_string_truncate(app->reply_acc, 0);
    app->cont_count = 0;
    app->continuing = FALSE;
    set_status(app, "new conversation");
}

static void on_prov_selected(GObject *o, GParamSpec *ps, gpointer data)
{
    (void)o;
    (void)ps;
    provider_apply((App *)data);
}

static void on_key_changed(GtkEditable *e, gpointer data)
{
    App *app = data;

    if (app->quiet)                        /* provider_apply is refilling */
        return;
    g_free(app->keys[app->prov_id]);
    app->keys[app->prov_id] = g_strdup(gtk_editable_get_text(e));
    if (gtk_check_button_get_active(GTK_CHECK_BUTTON(app->remember_cb)))
        config_save_later(app);
}

static void on_host_changed(GtkEditable *e, gpointer data)
{
    App *app = data;

    if (app->quiet)                        /* provider_apply is refilling */
        return;
    g_free(app->hosts[app->prov_id]);
    app->hosts[app->prov_id] = g_strdup(gtk_editable_get_text(e));
    if (gtk_check_button_get_active(GTK_CHECK_BUTTON(app->remember_cb)))
        config_save_later(app);
}

static void on_remember_toggled(GtkCheckButton *cb, gpointer data)
{
    (void)cb;
    config_save((App *)data);
}

static void on_model_changed(GtkEditable *e, gpointer data)
{
    (void)e;
    quirks_refresh((App *)data);
}

static void on_quirk_reset(GtkButton *b, gpointer data)
{
    App *app = data;
    char *k = quirk_key(app->prov_id, current_model(app));

    (void)b;
    g_hash_table_remove(app->quirks, k);
    g_hash_table_remove(app->caps, k);
    g_free(k);
    quirks_refresh(app);
    config_save(app);
    set_status(app, "learned parameter limits cleared for this model");
}

static void on_model_selected(GObject *o, GParamSpec *ps, gpointer data)
{
    App *app = data;
    guint i = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->model_dd));

    (void)o;
    (void)ps;
    if (app->quiet || i == GTK_INVALID_LIST_POSITION || i >= app->models->len)
        return;
    gtk_editable_set_text(GTK_EDITABLE(app->model_entry),
                          g_ptr_array_index(app->models, i));
}

static void on_model_activate(GtkEntry *e, gpointer data)
{
    App *app = data;
    const char *t = gtk_editable_get_text(GTK_EDITABLE(e));
    guint i;

    if (!t || !*t)
        return;
    for (i = 0; i < app->models->len; i++)
        if (!strcmp(g_ptr_array_index(app->models, i), t)) {
            gtk_drop_down_set_selected(GTK_DROP_DOWN(app->model_dd), i);
            return;
        }
    g_ptr_array_add(app->models, g_strdup(t));
    gtk_string_list_append(app->model_list, t);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app->model_dd),
                               (guint)app->models->len - 1);
}

/* ================================================================== */
/*  UI construction                                                   */
/* ================================================================== */

static void build_settings_pane(App *app, GtkWidget *left)
{
    GtkWidget *f, *grid, *w, *hbox, *sw, *inner;
    GtkStringList *provs;
    int i;

    /* ---- provider pad ---- */
    f = gtk_frame_new("Provider / API key");
    grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 8);
    margins(grid, 8);
    gtk_frame_set_child(GTK_FRAME(f), grid);

    provs = gtk_string_list_new(NULL);
    for (i = 0; i < PROV_COUNT; i++)
        gtk_string_list_append(provs, prov_get((ProvId)i)->label);
    app->prov_dd = gtk_drop_down_new(G_LIST_MODEL(provs), NULL);
    g_object_unref(provs);

    app->key_e = gtk_password_entry_new();
    gtk_password_entry_set_show_peek_icon(GTK_PASSWORD_ENTRY(app->key_e), TRUE);
    app->host_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->host_e), "https://…");

    w = gtk_label_new("Provider");
    gtk_widget_set_halign(w, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), w, 0, 0, 1, 1);
    gtk_widget_set_hexpand(app->prov_dd, TRUE);
    gtk_grid_attach(GTK_GRID(grid), app->prov_dd, 1, 0, 1, 1);

    w = gtk_label_new("API key");
    gtk_widget_set_halign(w, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), w, 0, 1, 1, 1);
    gtk_widget_set_hexpand(app->key_e, TRUE);
    gtk_grid_attach(GTK_GRID(grid), app->key_e, 1, 1, 1, 1);

    w = gtk_label_new("Host");
    gtk_widget_set_halign(w, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), w, 0, 2, 1, 1);
    gtk_widget_set_hexpand(app->host_e, TRUE);
    gtk_grid_attach(GTK_GRID(grid), app->host_e, 1, 2, 1, 1);

    hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    {
        GtkWidget *load = gtk_button_new_with_label("Load models");
        GtkWidget *prev = gtk_button_new_with_label("Preview request");

        gtk_box_append(GTK_BOX(hbox), load);
        gtk_box_append(GTK_BOX(hbox), prev);
        g_signal_connect(load, "clicked", G_CALLBACK(on_load_models), app);
        g_signal_connect(prev, "clicked", G_CALLBACK(on_preview), app);
    }
    gtk_grid_attach(GTK_GRID(grid), hbox, 1, 3, 1, 1);

    app->remember_cb = gtk_check_button_new_with_label(
            "Remember keys on disk (chmod 600)");
    gtk_grid_attach(GTK_GRID(grid), app->remember_cb, 0, 4, 2, 1);
    gtk_box_append(GTK_BOX(left), f);

    /* ---- model ---- */
    f = gtk_frame_new("Model");
    grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 8);
    margins(grid, 8);
    gtk_frame_set_child(GTK_FRAME(f), grid);

    app->model_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->model_entry),
        "llama3.1 · gpt-4o-mini · claude-sonnet-4-5 · gemini-2.5-flash");
    gtk_widget_set_hexpand(app->model_entry, TRUE);
    gtk_grid_attach(GTK_GRID(grid), app->model_entry, 0, 0, 1, 1);

    app->model_list = gtk_string_list_new(NULL);
    app->model_dd = gtk_drop_down_new(G_LIST_MODEL(app->model_list), NULL);
    gtk_widget_set_hexpand(app->model_dd, TRUE);
    gtk_grid_attach(GTK_GRID(grid), app->model_dd, 0, 1, 1, 1);
    gtk_box_append(GTK_BOX(left), f);

    /* ---- system prompt ---- */
    f = gtk_frame_new("System prompt");
    app->sys_buf = gtk_text_buffer_new(NULL);
    w = gtk_text_view_new_with_buffer(app->sys_buf);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(w), GTK_WRAP_WORD);
    sw = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
                                    GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(sw), w);
    gtk_widget_set_size_request(sw, -1, 90);
    inner = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    margins(inner, 8);
    gtk_box_append(GTK_BOX(inner), sw);
    gtk_frame_set_child(GTK_FRAME(f), inner);
    gtk_box_append(GTK_BOX(left), f);

    /* ---- parameters ---- */
    f = gtk_frame_new("Parameters");
    grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 8);
    margins(grid, 8);
    gtk_frame_set_child(GTK_FRAME(f), grid);
    {
        struct {
            const char *lbl, *tip;
            GtkWidget **cb, **e;
            const char *dflt;
        } rows[] = {
            { "Temperature", "Sampling temperature. Unchecked: not sent.",
              &app->temp_cb, &app->temp_e, "0.8" },
            { "Top-p", "Nucleus sampling. Unchecked: not sent. Anthropic gets "
              "temperature or top-p, never both.",
              &app->topp_cb, &app->topp_e, "0.9" },
            { "Max tokens", "Unchecked (recommended): OpenAI, Gemini and "
              "Ollama run to the model maximum; Anthropic requires a value "
              "and gets 32000 (64000 with thinking on). Values above a "
              "model's limit are learned and clamped automatically.",
              &app->maxtok_cb, &app->maxtok_e, "16384" },
            { "Seed", "Fixed seed for repeatable sampling. Not sent to "
              "Anthropic, which has no seed.",
              &app->seed_cb, &app->seed_e, "42" },
        };
        static const char *const think_opts[] = {
            "Model default (not sent)", "On", "Off", NULL
        };
        static const char *const effort_opts[] = {
            "Model default (not sent)", "low", "medium", "high", "xhigh", "max",
            NULL
        };
        int r;

        for (r = 0; r < 4; r++) {
            *rows[r].cb = gtk_check_button_new_with_label(rows[r].lbl);
            gtk_widget_set_tooltip_text(*rows[r].cb, rows[r].tip);
            gtk_grid_attach(GTK_GRID(grid), *rows[r].cb, 0, r, 1, 1);
            *rows[r].e = gtk_entry_new();
            gtk_editable_set_text(GTK_EDITABLE(*rows[r].e), rows[r].dflt);
            gtk_widget_set_hexpand(*rows[r].e, TRUE);
            gtk_grid_attach(GTK_GRID(grid), *rows[r].e, 1, r, 1, 1);
            g_object_bind_property(*rows[r].cb, "active", *rows[r].e,
                                   "sensitive", G_BINDING_SYNC_CREATE);
        }
        gtk_check_button_set_active(GTK_CHECK_BUTTON(app->temp_cb), TRUE);

        w = gtk_label_new("Thinking");
        gtk_widget_set_halign(w, GTK_ALIGN_START);
        gtk_grid_attach(GTK_GRID(grid), w, 0, 4, 1, 1);
        app->think_dd = gtk_drop_down_new_from_strings(think_opts);
        gtk_widget_set_tooltip_text(app->think_dd,
            "Off is never sent to models whose thinking cannot be disabled "
            "(Claude Fable / Mythos).");
        gtk_grid_attach(GTK_GRID(grid), app->think_dd, 1, 4, 1, 1);

        w = gtk_label_new("Effort");
        gtk_widget_set_halign(w, GTK_ALIGN_START);
        gtk_grid_attach(GTK_GRID(grid), w, 0, 5, 1, 1);
        app->effort_dd = gtk_drop_down_new_from_strings(effort_opts);
        gtk_widget_set_tooltip_text(app->effort_dd,
            "Anthropic: output_config.effort. OpenAI: reasoning_effort "
            "(xhigh/max sent as high). Not used for Gemini or Ollama.");
        gtk_grid_attach(GTK_GRID(grid), app->effort_dd, 1, 5, 1, 1);

        app->adapt_cb = gtk_check_button_new_with_label(
            "Auto-adapt: drop parameters the API rejects, then retry");
        gtk_check_button_set_active(GTK_CHECK_BUTTON(app->adapt_cb), TRUE);
        gtk_grid_attach(GTK_GRID(grid), app->adapt_cb, 0, 6, 2, 1);

        app->cont_cb = gtk_check_button_new_with_label(
            "Auto-continue a reply cut off by the token limit");
        gtk_widget_set_tooltip_text(app->cont_cb,
            "When a reply stops because of the output-token limit, send "
            "\"continue\" and append the result to the same reply "
            "(up to 8 times per turn).");
        gtk_check_button_set_active(GTK_CHECK_BUTTON(app->cont_cb), TRUE);
        gtk_grid_attach(GTK_GRID(grid), app->cont_cb, 0, 7, 2, 1);

        hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
        app->quirk_label = gtk_label_new(NULL);
        gtk_label_set_wrap(GTK_LABEL(app->quirk_label), TRUE);
        gtk_label_set_xalign(GTK_LABEL(app->quirk_label), 0.0f);
        gtk_widget_add_css_class(app->quirk_label, "dim-label");
        gtk_widget_add_css_class(app->quirk_label, "note");
        gtk_widget_set_hexpand(app->quirk_label, TRUE);
        app->quirk_reset = gtk_button_new_with_label("Reset");
        gtk_widget_set_tooltip_text(app->quirk_reset,
            "Forget what was learned for this model");
        gtk_widget_set_valign(app->quirk_reset, GTK_ALIGN_CENTER);
        gtk_box_append(GTK_BOX(hbox), app->quirk_label);
        gtk_box_append(GTK_BOX(hbox), app->quirk_reset);
        gtk_widget_set_visible(app->quirk_label, FALSE);
        gtk_widget_set_visible(app->quirk_reset, FALSE);
        gtk_grid_attach(GTK_GRID(grid), hbox, 0, 8, 2, 1);
    }
    hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    app->stream_cb = gtk_check_button_new_with_label("Stream");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->stream_cb), TRUE);
    app->newchat_btn = gtk_button_new_with_label("New chat");
    gtk_box_append(GTK_BOX(hbox), app->stream_cb);
    gtk_box_append(GTK_BOX(hbox), app->newchat_btn);
    gtk_grid_attach(GTK_GRID(grid), hbox, 0, 9, 2, 1);
    gtk_box_append(GTK_BOX(left), f);

    /* pad signal wiring */
    g_signal_connect(app->prov_dd, "notify::selected",
                     G_CALLBACK(on_prov_selected), app);
    g_signal_connect(app->key_e, "changed", G_CALLBACK(on_key_changed), app);
    g_signal_connect(app->host_e, "changed", G_CALLBACK(on_host_changed), app);
    g_signal_connect(app->remember_cb, "toggled",
                     G_CALLBACK(on_remember_toggled), app);
    g_signal_connect(app->model_dd, "notify::selected",
                     G_CALLBACK(on_model_selected), app);
    g_signal_connect(app->model_entry, "activate",
                     G_CALLBACK(on_model_activate), app);
    g_signal_connect(app->model_entry, "changed",
                     G_CALLBACK(on_model_changed), app);
    g_signal_connect(app->quirk_reset, "clicked",
                     G_CALLBACK(on_quirk_reset), app);
    g_signal_connect(app->newchat_btn, "clicked",
                     G_CALLBACK(on_new_chat), app);
}

static void build_chat_pane(App *app, GtkWidget *right)
{
    GtkWidget *sw, *cbox, *row, *arow, *scroll_in, *sep, *frame;
    GtkEventController *key, *motion;
    GtkGesture *click;

    /* ---- transcript ---- */
    app->chat_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_add_css_class(app->chat_box, "chat");

    sw = gtk_scrolled_window_new();
    app->chat_sw = sw;
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
                                    GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_has_frame(GTK_SCROLLED_WINDOW(sw), TRUE);
    gtk_widget_add_css_class(sw, "chat-window");
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(sw), app->chat_box);
    gtk_widget_set_vexpand(sw, TRUE);
    gtk_box_append(GTK_BOX(right), sw);
    app->chat_adj = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(sw));
    app->follow = TRUE;
    g_signal_connect(app->chat_adj, "value-changed",
                     G_CALLBACK(on_adj_value), app);
    g_signal_connect(app->chat_adj, "notify::upper",
                     G_CALLBACK(on_adj_bounds), app);
    g_signal_connect(app->chat_adj, "notify::page-size",
                     G_CALLBACK(on_adj_bounds), app);

    /* Capture mouse motion and button release on the scrolled window so
     * dragging a text selection to the top/bottom edge scrolls the chat. */
    motion = gtk_event_controller_motion_new();
    gtk_event_controller_set_propagation_phase(motion, GTK_PHASE_CAPTURE);
    g_signal_connect(motion, "motion", G_CALLBACK(on_chat_motion), app);
    gtk_widget_add_controller(sw, motion);

    click = gtk_gesture_click_new();
    gtk_event_controller_set_propagation_phase(GTK_EVENT_CONTROLLER(click),
                                               GTK_PHASE_CAPTURE);
    g_signal_connect(click, "released", G_CALLBACK(on_chat_btn_released), app);
    g_signal_connect_swapped(click, "stopped",
                             G_CALLBACK(drag_scroll_stop), app);
    gtk_widget_add_controller(sw, GTK_EVENT_CONTROLLER(click));

    /* ---- a visible break, then the composer in its own frame ---- */
    sep = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_widget_set_margin_top(sep, 6);
    gtk_widget_set_margin_bottom(sep, 2);
    gtk_box_append(GTK_BOX(right), sep);

    frame = gtk_frame_new("Your message  ·  Ctrl+Enter sends");
    gtk_widget_add_css_class(frame, "composer");
    cbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    margins(cbox, 8);
    gtk_frame_set_child(GTK_FRAME(frame), cbox);

    app->input_buf = gtk_text_buffer_new(NULL);
    {
        GtkWidget *iv = gtk_text_view_new_with_buffer(app->input_buf);

        gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(iv), GTK_WRAP_WORD);
        key = gtk_event_controller_key_new();
        g_signal_connect(key, "key-pressed", G_CALLBACK(on_input_key), app);
        gtk_widget_add_controller(iv, key);
        scroll_in = gtk_scrolled_window_new();
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll_in),
                                        GTK_POLICY_AUTOMATIC,
                                        GTK_POLICY_AUTOMATIC);
        gtk_scrolled_window_set_has_frame(GTK_SCROLLED_WINDOW(scroll_in), TRUE);
        gtk_widget_add_css_class(scroll_in, "composer-input");
        gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scroll_in), iv);
        gtk_widget_set_size_request(scroll_in, -1, 110);
    }

    row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    app->send_btn = gtk_button_new_with_label("Send");
    gtk_widget_add_css_class(app->send_btn, "suggested-action");
    app->stop_btn = gtk_button_new_with_label("Stop");
    gtk_widget_set_sensitive(app->stop_btn, FALSE);
    app->spinner = gtk_spinner_new();
    app->status = gtk_label_new("ready");
    gtk_label_set_ellipsize(GTK_LABEL(app->status), PANGO_ELLIPSIZE_END);
    gtk_widget_set_halign(app->status, GTK_ALIGN_START);
    gtk_widget_set_hexpand(app->status, TRUE);
    g_signal_connect(app->send_btn, "clicked", G_CALLBACK(on_send), app);
    g_signal_connect(app->stop_btn, "clicked", G_CALLBACK(on_stop), app);

    gtk_box_append(GTK_BOX(row), app->send_btn);
    gtk_box_append(GTK_BOX(row), app->stop_btn);
    gtk_box_append(GTK_BOX(row), app->spinner);
    gtk_box_append(GTK_BOX(row), app->status);
    {
        GtkWidget *save_btn = gtk_button_new_with_label("Save chat…");

        app->code_btn = gtk_button_new_with_label("Save code…");
        gtk_widget_set_sensitive(app->code_btn, FALSE);
        g_signal_connect(save_btn, "clicked", G_CALLBACK(on_save_chat), app);
        g_signal_connect(app->code_btn, "clicked",
                         G_CALLBACK(on_save_code), app);
        gtk_box_append(GTK_BOX(row), save_btn);
        gtk_box_append(GTK_BOX(row), app->code_btn);
    }

    /* attachment strip between the composer and the send row */
    arow = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    {
        GtkWidget *attach_btn = gtk_button_new_with_label("Attach…");

        app->att_clear_btn = gtk_button_new_with_label("Clear");
        app->att_label = gtk_label_new("no attachments");
        gtk_label_set_ellipsize(GTK_LABEL(app->att_label), PANGO_ELLIPSIZE_END);
        gtk_widget_set_halign(app->att_label, GTK_ALIGN_START);
        gtk_widget_set_hexpand(app->att_label, TRUE);
        gtk_widget_add_css_class(app->att_label, "dim-label");
        gtk_widget_set_sensitive(app->att_clear_btn, FALSE);
        g_signal_connect(attach_btn, "clicked", G_CALLBACK(on_attach), app);
        g_signal_connect(app->att_clear_btn, "clicked",
                         G_CALLBACK(on_att_clear), app);
        gtk_box_append(GTK_BOX(arow), attach_btn);
        gtk_box_append(GTK_BOX(arow), app->att_clear_btn);
        gtk_box_append(GTK_BOX(arow), app->att_label);
    }

    gtk_box_append(GTK_BOX(cbox), scroll_in);
    gtk_box_append(GTK_BOX(cbox), arow);
    gtk_box_append(GTK_BOX(cbox), row);
    gtk_box_append(GTK_BOX(right), frame);
}

/* ================================================================== */
/*  application lifecycle                                             */
/* ================================================================== */

/* Save while the widgets still exist. "shutdown" fires only after the
 * window has been destroyed, so config_save() there read freed widgets. */
static gboolean on_close_request(GtkWindow *w, gpointer data)
{
    App *app = data;

    (void)w;
    drag_scroll_stop(app);
    if (app->save_id) {
        g_source_remove(app->save_id);
        app->save_id = 0;
    }
    config_save(app);
    if (app->worker)
        g_atomic_int_set(&app->worker->cancel, 1);
    if (app->models_job)
        g_atomic_int_set(&app->models_job->cancel, 1);
    app->closing = TRUE;
    return FALSE;                          /* let the window close */
}

static void on_shutdown(GApplication *gapp, gpointer data)
{
    App *app = data;
    int i;

    (void)gapp;
    /* the main loop is over: let the background threads finish before
     * curl_global_cleanup() runs (their idles never fire; they leak a
     * few bytes, which is fine at exit) */
    if (app->worker) {
        g_atomic_int_set(&app->worker->cancel, 1);
        if (app->worker->thread) {
            g_thread_join(app->worker->thread);
            app->worker->thread = NULL;
        }
    }
    if (app->models_job) {
        g_atomic_int_set(&app->models_job->cancel, 1);
        if (app->models_job->thread) {
            g_thread_join(app->models_job->thread);
            app->models_job->thread = NULL;
        }
    }
    for (i = 0; i < PROV_COUNT; i++) {
        g_free(app->keys[i]);
        g_free(app->hosts[i]);
        g_free(app->pmodel[i]);
    }
}

static void activate(GtkApplication *gtkapp, gpointer user)
{
    App *app = g_new0(App, 1);
    GtkWidget *win, *paned, *left_sw, *left, *right;

    (void)user;
    app->history = g_ptr_array_new_with_free_func(msg_free);
    app->models = g_ptr_array_new_with_free_func(g_free);
    app->reply_acc = g_string_new(NULL);
    app->att = g_ptr_array_new_with_free_func(attachment_free);
    app->quirks = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    app->caps = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    config_load(app);

    win = gtk_application_window_new(gtkapp);
    app->win = GTK_WINDOW(win);
    gtk_window_set_title(app->win,
                         "ollama-gui — Ollama · OpenAI · Anthropic · Gemini");
    gtk_window_set_default_size(app->win, 1150, 780);

    paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(paned), 400);

    left_sw = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(left_sw),
                                    GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(left_sw, 340, -1);
    left = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    margins(left, 8);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(left_sw), left);
    gtk_paned_set_start_child(GTK_PANED(paned), left_sw);

    right = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    margins(right, 8);
    gtk_paned_set_end_child(GTK_PANED(paned), right);

    load_css();
    build_settings_pane(app, left);
    build_chat_pane(app, right);

    gtk_window_set_child(app->win, paned);
    gtk_window_present(app->win);

    apply_config(app);
    chat_note(app,
        "Pick a provider, paste its API key, load models, then send "
        "(Ctrl+Enter sends). Attach… adds images or text files to the "
        "next message. Replies cut off by the token limit are continued "
        "automatically.", FALSE);

    g_signal_connect(win, "close-request", G_CALLBACK(on_close_request), app);
    g_signal_connect(gtkapp, "shutdown", G_CALLBACK(on_shutdown), app);
}

int main(int argc, char **argv)
{
    GtkApplication *gtkapp;
    int status;

    curl_global_init(CURL_GLOBAL_DEFAULT);   /* not thread-safe if implicit */
    gtkapp = gtk_application_new("org.example.cloudchat", APP_FLAGS);
    g_signal_connect(gtkapp, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(gtkapp), argc, argv);
    g_object_unref(gtkapp);
    curl_global_cleanup();
    return status;
}
