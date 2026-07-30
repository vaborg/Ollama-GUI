/*
 * ollama_gui.c — desktop client for the Ollama REST API, GTK4 + libcurl + json-glib.
 *
 * Build dependencies (Debian/Ubuntu):
 *
 *   sudo apt install build-essential pkg-config libgtk-4-dev \
 *                    libjson-glib-dev libcurl4-openssl-dev
 *
 * Compile:
 *
 *   gcc -O2 -Wall -o ollama-gui ollama_gui.c \
 *       $(pkg-config --cflags --libs gtk4 json-glib-1.0 libcurl)
 *
 * Requires GTK 4.10 or newer (GtkFileDialog, GtkAlertDialog).
 *
 * Layout: request builder on the left, chat transcript in the centre,
 * composer at the bottom. Every request is assembled as a JsonNode, so
 * "Preview request" prints the exact bytes that will be POSTed.
 *
 * Streaming runs on a worker thread; the libcurl write callback splits the
 * NDJSON stream into lines and hands text to the GTK main loop through
 * g_idle_add(). No GTK call is ever made off the main thread.
 *
 * API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
 */

#include <gtk/gtk.h>
#include <json-glib/json-glib.h>
#include <curl/curl.h>
#include <glib/gstdio.h>
#include <stdarg.h>
#include <string.h>
#include <signal.h>

/* ------------------------------------------------------------------ */
/* constants                                                          */
/* ------------------------------------------------------------------ */

static const char *const LOAD_MODES[] = {
    "auto", "text", "image (base64)", "JSON", "raw -> base64", NULL
};
static const char *const OUT_MODES[] = {
    "free text", "JSON mode", "JSON schema", NULL
};
static const char *const ENDPOINTS[] = {
    "/api/chat", "/api/generate", NULL
};

#define PANE_LEFT_MIN   280
#define PANE_RIGHT_MIN  360

typedef enum {
    MODE_AUTO = 0,
    MODE_TEXT,
    MODE_IMAGE,
    MODE_JSON,
    MODE_B64
} LoadMode;

typedef struct {
    char    *path;
    LoadMode mode;
} Attachment;

typedef struct {
    char    *model;
    gboolean cloud;              /* billed: cloud tag, or host ollama.com */
    gint64   requests;
    gint64   prompt_tokens;
    gint64   eval_tokens;
} Usage;

typedef struct {
    GtkWindow     *win;
    GtkWidget     *host_entry;
    GtkWidget     *model_dd;
    GtkWidget     *endpoint_dd;
    GtkWidget     *format_dd;
    GtkWidget     *loadmode_dd;
    GtkStringList *model_list;

    GtkTextBuffer *sys_buf;
    GtkTextBuffer *schema_buf;
    GtkTextBuffer *chat_buf;
    GtkTextBuffer *input_buf;
    GtkTextBuffer *attach_buf;
    GtkWidget     *chat_view;
    GtkTextMark   *end_mark;

    GtkWidget *temp_e, *topp_e, *ctx_e, *npred_e, *seed_e, *keep_e, *limit_e;
    GtkWidget *stream_cb, *think_cb, *hist_cb, *ground_cb;
    GtkWidget *send_btn, *stop_btn, *spinner, *status;

    GPtrArray *models;       /* char*       */
    GPtrArray *attachments;  /* Attachment* */
    JsonArray *messages;     /* conversation history */

    char *last_response;
    char *last_payload;

    volatile gint cancel;
    gboolean      busy;

    GtkWidget *paned;
    GtkWidget *ptr_cb;

    GtkWidget *code_btn, *points_btn;

    GtkWidget *colour_cb;
    gboolean   tags_ready;

    GtkWidget   *serve_cmd_e, *signin_cmd_e, *apikey_entry, *remember_key_cb;
    GtkWidget   *serve_btn, *serve_stop_btn, *version_label;
    GtkTextBuffer *log_buf;
    GSubprocess *server_proc;

    GtkWidget     *pull_entry, *pull_btn, *pull_bar, *pull_status;
    GtkTextBuffer *models_buf;
    volatile gint  pull_cancel;
    gboolean       pulling;

    GtkWidget     *price_in_e, *price_out_e, *budget_e, *currency_e;
    GtkWidget     *usage_summary;
    GtkTextBuffer *usage_buf;
    GPtrArray     *usage;        /* Usage* */
    char          *period_start;

    char *cfg_host, *cfg_key, *cfg_price_in, *cfg_price_out;
    char *cfg_budget, *cfg_currency, *cfg_serve_cmd, *cfg_signin_cmd;
    gboolean cfg_remember_key;
} App;

static char *auth_header(App *app);
static void  usage_record(App *app, const char *model, gint64 in_tok,
                          gint64 out_tok);
static void  render_usage(App *app);
static void  save_state(App *app);
static void  load_state(App *app);
static void  start_inventory_load(App *app);
static void  start_version_probe(App *app);
static void  on_shutdown(GApplication *gapp, gpointer data);
static GtkWidget *build_left_pane(App *app, GtkWidget *request_page);

/* ------------------------------------------------------------------ */
/* small helpers                                                      */
/* ------------------------------------------------------------------ */

static char *buffer_text(GtkTextBuffer *b)
{
    GtkTextIter s, e;
    gtk_text_buffer_get_bounds(b, &s, &e);
    return gtk_text_buffer_get_text(b, &s, &e, FALSE);
}

static void chat_append(App *app, const char *s)
{
    GtkTextIter end;

    if (!s || !*s)
        return;
    gtk_text_buffer_get_end_iter(app->chat_buf, &end);
    gtk_text_buffer_insert(app->chat_buf, &end, s, -1);
    gtk_text_buffer_get_end_iter(app->chat_buf, &end);
    gtk_text_buffer_move_mark(app->chat_buf, app->end_mark, &end);
    gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(app->chat_view),
                                 app->end_mark, 0.0, TRUE, 0.0, 1.0);
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

static void attachment_free(gpointer p)
{
    Attachment *a = p;

    g_free(a->path);
    g_free(a);
}

static char *base_url(App *app)
{
    const char *h = gtk_editable_get_text(GTK_EDITABLE(app->host_entry));
    char *s = g_strstrip(g_strdup(h ? h : ""));
    gsize n = strlen(s);

    while (n > 0 && s[n - 1] == '/')
        s[--n] = '\0';
    return s;
}

static const char *selected_model(App *app)
{
    guint i = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->model_dd));

    if (i == GTK_INVALID_LIST_POSITION || i >= app->models->len)
        return "";
    return g_ptr_array_index(app->models, i);
}

static int selected_endpoint(App *app)
{
    guint i = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->endpoint_dd));

    return (i == GTK_INVALID_LIST_POSITION) ? 0 : (int)i;
}

static const char *obj_str(JsonObject *o, const char *key)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, key))
        return NULL;
    n = json_object_get_member(o, key);
    if (!JSON_NODE_HOLDS_VALUE(n))
        return NULL;
    return json_node_get_string(n);
}

static gint64 obj_int(JsonObject *o, const char *key, gint64 dflt)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, key))
        return dflt;
    n = json_object_get_member(o, key);
    if (!JSON_NODE_HOLDS_VALUE(n))
        return dflt;
    return json_node_get_int(n);
}

static gboolean obj_true(JsonObject *o, const char *key)
{
    JsonNode *n;

    if (!o || !json_object_has_member(o, key))
        return FALSE;
    n = json_object_get_member(o, key);
    if (!JSON_NODE_HOLDS_VALUE(n))
        return FALSE;
    return json_node_get_boolean(n);
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

static char *timestamp(void)
{
    GDateTime *t = g_date_time_new_now_local();
    char *s = g_date_time_format(t, "%H:%M:%S");

    g_date_time_unref(t);
    return s;
}

/* ------------------------------------------------------------------ */
/* attachments                                                        */
/* ------------------------------------------------------------------ */

static LoadMode resolve_mode(const Attachment *a)
{
    static const char *const img[] = {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", NULL
    };
    char *lower;
    LoadMode m = MODE_TEXT;
    int i;

    if (a->mode != MODE_AUTO)
        return a->mode;
    lower = g_ascii_strdown(a->path, -1);
    for (i = 0; img[i]; i++) {
        if (g_str_has_suffix(lower, img[i])) {
            m = MODE_IMAGE;
            break;
        }
    }
    if (m == MODE_TEXT && g_str_has_suffix(lower, ".json"))
        m = MODE_JSON;
    g_free(lower);
    return m;
}

static const char *mode_name(LoadMode m)
{
    switch (m) {
    case MODE_IMAGE: return "image";
    case MODE_JSON:  return "json";
    case MODE_B64:   return "base64";
    default:         return "text";
    }
}

static void refresh_attachment_list(App *app)
{
    GString *s;
    guint i;

    if (app->attachments->len == 0) {
        gtk_text_buffer_set_text(app->attach_buf, "(no files attached)", -1);
        return;
    }
    s = g_string_new(NULL);
    for (i = 0; i < app->attachments->len; i++) {
        Attachment *a = g_ptr_array_index(app->attachments, i);
        char *base = g_path_get_basename(a->path);

        g_string_append_printf(s, "%u. %s  [%s]\n", i + 1, base,
                               mode_name(resolve_mode(a)));
        g_free(base);
    }
    if (s->len)
        g_string_truncate(s, s->len - 1);
    gtk_text_buffer_set_text(app->attach_buf, s->str, -1);
    g_string_free(s, TRUE);
}

static gsize char_limit(App *app)
{
    const char *t = gtk_editable_get_text(GTK_EDITABLE(app->limit_e));
    gint64 v = t ? g_ascii_strtoll(t, NULL, 10) : 0;

    return (v > 0) ? (gsize)v : 20000;
}

/*
 * Fold every attachment into the message: text-like files become fenced
 * blocks inside "content", images become base64 strings for "images".
 */
static gboolean build_user_content(App *app, const char *text,
                                   GString *content, GPtrArray *images,
                                   GError **err)
{
    gsize limit = char_limit(app);
    guint i;

    if (text && *text)
        g_string_append(content, text);

    for (i = 0; i < app->attachments->len; i++) {
        Attachment *a = g_ptr_array_index(app->attachments, i);
        LoadMode m = resolve_mode(a);
        char *base = g_path_get_basename(a->path);
        char *data = NULL;
        gsize len = 0;

        if (!g_file_get_contents(a->path, &data, &len, err)) {
            g_free(base);
            return FALSE;
        }
        if (content->len)
            g_string_append(content, "\n\n");

        if (m == MODE_IMAGE) {
            g_ptr_array_add(images, g_base64_encode((const guchar *)data, len));
            g_string_append_printf(content, "[attached image: %s]", base);
        } else if (m == MODE_B64) {
            char *b64 = g_base64_encode((const guchar *)data, len);

            g_string_append_printf(content,
                                   "### %s (base64, %" G_GSIZE_FORMAT " bytes)\n%s",
                                   base, len, b64);
            g_free(b64);
        } else {
            const char *invalid = NULL;
            const char *dot;
            char *fence;
            gsize keep = len;
            gboolean truncated = FALSE;

            if (m == MODE_JSON) {
                JsonParser *p = json_parser_new();

                if (!json_parser_load_from_data(p, data, (gssize)len, err)) {
                    g_object_unref(p);
                    g_free(data);
                    g_free(base);
                    return FALSE;
                }
                g_object_unref(p);
            }
            if (!g_utf8_validate(data, (gssize)len, &invalid))
                keep = (gsize)(invalid - data);
            if (g_utf8_strlen(data, (gssize)keep) > (glong)limit) {
                const char *cut = g_utf8_offset_to_pointer(data, (glong)limit);

                keep = (gsize)(cut - data);
                truncated = TRUE;
            }
            dot = strrchr(base, '.');
            fence = (m == MODE_JSON) ? g_strdup("json")
                                     : (dot ? g_ascii_strdown(dot + 1, -1)
                                            : g_strdup(""));
            g_string_append_printf(content, "### %s\n```%s\n", base, fence);
            g_string_append_len(content, data, (gssize)keep);
            g_string_append(content, "\n```");
            if (truncated)
                g_string_append_printf(content,
                                       "\n[truncated at %" G_GSIZE_FORMAT " characters]",
                                       limit);
            g_free(fence);
        }
        g_free(data);
        g_free(base);
    }
    return TRUE;
}

/*
 * The settings/chat split is a GtkPaned, so the divider can always be
 * dragged by hand. On top of that, while the "Split follows mouse
 * pointer" toggle is active, the pointer's x-position inside the window
 * steers the divider: sliding left shrinks the settings pane, sliding
 * right shrinks the chat. Motion is eased (a quarter of the remaining
 * distance per event) so the panes glide instead of snapping.
 */
static void on_root_motion(GtkEventControllerMotion *ctrl,
                           double x, double y, gpointer data)
{
    App *app = data;
    int w, pos, diff, step;

    (void)ctrl;
    (void)y;
    if (!gtk_check_button_get_active(GTK_CHECK_BUTTON(app->ptr_cb)))
        return;

    w = gtk_widget_get_width(app->paned);
    if (w < PANE_LEFT_MIN + PANE_RIGHT_MIN)
        return;

    if (x < PANE_LEFT_MIN)
        x = PANE_LEFT_MIN;
    if (x > w - PANE_RIGHT_MIN)
        x = w - PANE_RIGHT_MIN;

    pos  = gtk_paned_get_position(GTK_PANED(app->paned));
    diff = (int)x - pos;
    if (diff > -2 && diff < 2)
        return;
    step = diff / 4;
    if (step == 0)
        step = diff > 0 ? 1 : -1;
    gtk_paned_set_position(GTK_PANED(app->paned), pos + step);
}

/* ------------------------------------------------------------------ */
/* extract code / scripts / points from the reply                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char *lang;
    char *code;
} CodeBlock;

static void code_block_free(gpointer p)
{
    CodeBlock *b = p;

    g_free(b->lang);
    g_free(b->code);
    g_free(b);
}

/* length of a markdown list marker at p: "- ", "* ", "+ ", "1. ", "2) " … */
static gsize list_marker_len(const char *p)
{
    gsize n = 0;

    if (g_str_has_prefix(p, "- ") || g_str_has_prefix(p, "* ") ||
        g_str_has_prefix(p, "+ "))
        return 2;
    while (g_ascii_isdigit(p[n]))
        n++;
    if (n && (p[n] == '.' || p[n] == ')') && p[n + 1] == ' ')
        return n + 2;
    return 0;
}

/* scan text for ```lang … ``` fences -> GPtrArray of CodeBlock* */
static GPtrArray *extract_code_blocks(const char *text)
{
    GPtrArray *out = g_ptr_array_new_with_free_func(code_block_free);
    GString *cur = NULL;
    char *lang = NULL;
    gboolean in = FALSE;
    g_auto(GStrv) lines = g_strsplit(text ? text : "", "\n", -1);
    guint i;

    for (i = 0; lines[i]; i++) {
        char *line = lines[i];
        const char *p;

        g_strchomp(line);
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;

        if (g_str_has_prefix(p, "```")) {
            if (!in) {                          /* opening fence */
                char *tok, *sp;

                g_free(lang);
                tok = g_strstrip(g_strdup(p + 3));
                sp = strpbrk(tok, " \t");       /* first token only */
                if (sp)
                    *sp = '\0';
                lang = tok;
                cur = g_string_new(NULL);
                in = TRUE;
            } else {                            /* closing fence */
                CodeBlock *b = g_new0(CodeBlock, 1);

                b->lang = g_strdup(lang ? lang : "");
                b->code = g_string_free(cur, FALSE);
                g_strchomp(b->code);
                g_ptr_array_add(out, b);
                cur = NULL;
                in = FALSE;
            }
            continue;
        }
        if (in) {
            g_string_append(cur, line);
            g_string_append_c(cur, '\n');
        }
    }
    if (in && cur) {                            /* unterminated fence: keep it */
        CodeBlock *b = g_new0(CodeBlock, 1);

        b->lang = g_strdup(lang ? lang : "");
        b->code = g_string_free(cur, FALSE);
        g_ptr_array_add(out, b);
    }
    g_free(lang);
    return out;
}

static const char *ext_for_lang(const char *lang)
{
    static const struct { const char *l; const char *e; } map[] = {
        { "c", "c" }, { "h", "h" },
        { "cpp", "cpp" }, { "cc", "cpp" }, { "c++", "cpp" }, { "cxx", "cpp" },
        { "python", "py" }, { "py", "py" },
        { "bash", "sh" }, { "sh", "sh" }, { "shell", "sh" }, { "zsh", "sh" },
        { "json", "json" },
        { "javascript", "js" }, { "js", "js" },
        { "typescript", "ts" }, { "ts", "ts" },
        { "html", "html" }, { "css", "css" }, { "xml", "xml" },
        { "yaml", "yml" }, { "yml", "yml" },
        { "sql", "sql" },
        { "rust", "rs" }, { "rs", "rs" },
        { "go", "go" }, { "java", "java" }, { "lua", "lua" },
        { "php", "php" }, { "markdown", "md" }, { "md", "md" },
        { "csharp", "cs" }, { "cs", "cs" },
        { "ruby", "rb" }, { "rb", "rb" },
        { "perl", "pl" }, { "swift", "swift" }, { "kotlin", "kt" },
        { "toml", "toml" }, { "ini", "ini" }, { "diff", "diff" },
    };
    char clean[16];
    gsize i, n = 0;

    if (!lang)
        return "txt";
    for (i = 0; lang[i] && n < sizeof(clean) - 1; i++)
        if (g_ascii_isalnum(lang[i]) || lang[i] == '+' || lang[i] == '#')
            clean[n++] = (char)g_ascii_tolower(lang[i]);
    clean[n] = '\0';
    if (!n)
        return "txt";
    for (i = 0; i < G_N_ELEMENTS(map); i++)
        if (strcmp(clean, map[i].l) == 0)
            return map[i].e;
    return "txt";
}

/* headings + bullet / numbered items, excluding fenced code */
static char *extract_points(const char *text)
{
    GString *out = g_string_new("# Key points extracted from the reply\n\n");
    gboolean in_code = FALSE;
    g_auto(GStrv) lines = g_strsplit(text ? text : "", "\n", -1);
    guint i;

    for (i = 0; lines[i]; i++) {
        const char *p = lines[i];

        while (*p == ' ' || *p == '\t')
            p++;
        if (g_str_has_prefix(p, "```")) {
            in_code = !in_code;
            continue;
        }
        if (in_code || !*p)
            continue;
        if (*p == '#' || list_marker_len(p) > 0) {
            g_string_append(out, p);
            g_string_append_c(out, '\n');
        }
    }
    return g_string_free(out, FALSE);
}

typedef enum {
    EXTRACT_CODE = 0,
    EXTRACT_POINTS
} ExtractKind;

typedef struct {
    App         *app;
    ExtractKind  kind;
} ExtractJob;

static void extraction_write(App *app, ExtractKind kind, const char *base_dir)
{
    char *src = (app->last_response && *app->last_response)
                    ? g_strdup(app->last_response)
                    : buffer_text(app->chat_buf);
    GDateTime *now = g_date_time_new_now_local();
    char *stamp = g_date_time_format(now, "%Y%m%d-%H%M%S");
    char *outdir = g_build_filename(base_dir, "ollama-extract", stamp, NULL);
    GError *err = NULL;
    int saved = 0, found = 0;

    g_date_time_unref(now);

    if (g_mkdir_with_parents(outdir, 0755) != 0) {
        show_error(app, "Cannot create output directory");
        goto done;
    }

    if (kind == EXTRACT_CODE) {
        GPtrArray *blocks = extract_code_blocks(src);
        guint i;

        found = (int)blocks->len;
        for (i = 0; i < blocks->len; i++) {
            CodeBlock *b = g_ptr_array_index(blocks, i);
            char *name = g_strdup_printf("code_%02u.%s", i + 1,
                                         ext_for_lang(b->lang));
            char *path = g_build_filename(outdir, name, NULL);

            if (g_file_set_contents(path, b->code, -1, &err))
                saved++;
            else {
                show_error(app, err->message);
                g_clear_error(&err);
            }
            g_free(path);
            g_free(name);
        }
        g_ptr_array_free(blocks, TRUE);
        if (!found)
            set_status(app, "No ``` code blocks found in the reply");
    } else {
        char *doc  = extract_points(src);
        char *path = g_build_filename(outdir, "points.md", NULL);

        found = 1;
        if (g_file_set_contents(path, doc, -1, &err))
            saved = 1;
        else {
            show_error(app, err->message);
            g_clear_error(&err);
        }
        g_free(path);
        g_free(doc);
    }
    if (saved)
        set_status(app, "%d file%s saved -> %s",
                   saved, saved == 1 ? "" : "s", outdir);

done:
    g_free(outdir);
    g_free(stamp);
    g_free(src);
}

static void on_extract_dir(GObject *src, GAsyncResult *res, gpointer data)
{
    ExtractJob *job = data;
    GFile *dir = gtk_file_dialog_select_folder_finish(GTK_FILE_DIALOG(src),
                                                      res, NULL);

    if (dir) {
        char *path = g_file_get_path(dir);

        if (path) {
            extraction_write(job->app, job->kind, path);
            g_free(path);
        }
        g_object_unref(dir);
    }
    g_free(job);
}

static void start_extract(App *app, ExtractKind kind, const char *title)
{
    ExtractJob *job = g_new0(ExtractJob, 1);
    GtkFileDialog *d = gtk_file_dialog_new();

    job->app = app;
    job->kind = kind;
    gtk_file_dialog_set_title(d, title);
    gtk_file_dialog_select_folder(d, app->win, NULL, on_extract_dir, job);
    g_object_unref(d);
}

static void on_extract_code(GtkButton *b, gpointer data)
{
    (void)b;
    start_extract(data, EXTRACT_CODE, "Save extracted code into folder");
}

static void on_extract_points(GtkButton *b, gpointer data)
{
    (void)b;
    start_extract(data, EXTRACT_POINTS, "Save reply outline into folder");
}

/* ------------------------------------------------------------------ */
/* colourful rendering of answers                                     */
/* ------------------------------------------------------------------ */

static void ensure_tags(App *app)
{
    if (app->tags_ready)
        return;
    app->tags_ready = TRUE;

    /* markdown-ish line tags */
    gtk_text_buffer_create_tag(app->chat_buf, "h1",
        "weight", PANGO_WEIGHT_BOLD, "scale", 1.4,  "foreground", "#d63384", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "h2",
        "weight", PANGO_WEIGHT_BOLD, "scale", 1.25, "foreground", "#6f42c1", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "h3",
        "weight", PANGO_WEIGHT_BOLD, "foreground", "#0d6efd", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "code",
        "family", "monospace", "paragraph-background", "#161b22",
        "foreground", "#e6edf3", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "fence",
        "family", "monospace", "paragraph-background", "#161b22",
        "foreground", "#7d8590", "scale", 0.85, NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "icode",
        "family", "monospace", "background", "#eef1f4",
        "foreground", "#c7254e", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "strong",
        "weight", PANGO_WEIGHT_BOLD, "foreground", "#b35900", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "ulink",
        "foreground", "#0d6efd", "underline", PANGO_UNDERLINE_SINGLE, NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "quote",
        "style", PANGO_STYLE_ITALIC, "foreground", "#1a7f37", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "bullet",
        "weight", PANGO_WEIGHT_BOLD, "foreground", "#8250df", NULL);
    /* app role headers */
    gtk_text_buffer_create_tag(app->chat_buf, "urole",
        "weight", PANGO_WEIGHT_BOLD, "foreground", "#1a7f37", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "arole",
        "weight", PANGO_WEIGHT_BOLD, "foreground", "#0d6efd", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "err",
        "weight", PANGO_WEIGHT_BOLD, "foreground", "#cf222e", NULL);
    /* inside fenced code — github-dark-ish palette */
    gtk_text_buffer_create_tag(app->chat_buf, "kw",  "foreground", "#ff7b72", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "str", "foreground", "#a5d6ff", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "num", "foreground", "#79c0ff", NULL);
    gtk_text_buffer_create_tag(app->chat_buf, "cmt", "foreground", "#8b949e",
        "style", PANGO_STYLE_ITALIC, NULL);
}

/*
 * Tag the [from, to) byte range of 'base' (the text that starts at the
 * 'block_start' iter). Byte offsets are converted to character offsets.
 */
static void tag_span(App *app, GtkTextIter *block_start, const char *base,
                     const char *from, const char *to, const char *tag)
{
    GtkTextIter a = *block_start, b = *block_start;

    gtk_text_iter_forward_chars(&a, (gint)g_utf8_strlen(base, (gssize)(from - base)));
    gtk_text_iter_forward_chars(&b, (gint)g_utf8_strlen(base, (gssize)(to - base)));
    gtk_text_buffer_apply_tag_by_name(app->chat_buf, tag, &a, &b);
}

static gboolean is_ident(char c)
{
    return g_ascii_isalnum(c) || c == '_';
}

static gboolean is_keyword(const char *s, gsize n)
{
    static const char *const kws[] = {
        /* C / C-like */
        "if", "else", "for", "while", "do", "switch", "case", "default",
        "break", "continue", "return", "goto", "sizeof", "typedef",
        "struct", "union", "enum", "static", "const", "volatile", "extern",
        "inline", "void", "char", "short", "int", "long", "float",
        "double", "signed", "unsigned", "bool", "true", "false", "NULL",
        /* python */
        "def", "class", "import", "from", "in", "is", "None", "and", "or",
        "not", "pass", "lambda", "with", "as", "try", "except", "finally",
        "raise", "yield", "global", "elif",
        /* rust / go / js / shell misc */
        "fn", "let", "mut", "pub", "impl", "match", "use", "func", "var",
        "new", "this", "function", "package", "local", "then", "fi",
        "done", "echo", "print", "public", "private", "async", "await",
        NULL
    };
    int i;

    for (i = 0; kws[i]; i++)
        if (strlen(kws[i]) == n && strncmp(kws[i], s, n) == 0)
            return TRUE;
    return FALSE;
}

/* lightweight tokenizer for strings / comments / numbers / keywords */
static void highlight_code(App *app, GtkTextIter *start, GtkTextIter *end)
{
    char *code = gtk_text_buffer_get_text(app->chat_buf, start, end, FALSE);
    const char *p = code;

    while (*p) {
        const char *to = NULL, *tag = NULL;

        if (p[0] == '/' && p[1] == '/') {
            to = strchr(p, '\n');
            if (!to)
                to = p + strlen(p);
            tag = "cmt";
        } else if (p[0] == '/' && p[1] == '*') {
            to = strstr(p + 2, "*/");
            to = to ? to + 2 : p + strlen(p);
            tag = "cmt";
        } else if (*p == '#') {                 /* python / shell comment */
            to = strchr(p, '\n');
            if (!to)
                to = p + strlen(p);
            tag = "cmt";
        } else if (*p == '"' || *p == '\'') {
            char q = *p;
            const char *s = p + 1;

            while (*s && *s != q) {
                if (*s == '\\' && s[1])
                    s++;
                s++;
            }
            to = (*s == q) ? s + 1 : s;
            tag = "str";
        } else if (g_ascii_isdigit(*p)) {
            const char *s = p;

            while (*s && (is_ident(*s) || *s == '.'))
                s++;
            to = s;
            tag = "num";
        } else if (g_ascii_isalpha(*p) || *p == '_') {
            const char *s = p;

            while (is_ident(*s))
                s++;
            if (is_keyword(p, (gsize)(s - p)))
                tag_span(app, start, code, p, s, "kw");
            p = s;
            continue;
        }
        if (tag) {
            tag_span(app, start, code, p, to, tag);
            p = to;
        } else {
            p++;
        }
    }
    g_free(code);
}

/* inline `code`, **bold** and URLs inside one line */
static void highlight_inline_md(App *app, GtkTextIter *line_start_after_indent,
                                const char *text)
{
    const char *p = text;

    while (*p) {
        if (*p == '`') {
            const char *q = strchr(p + 1, '`');

            if (q) {
                tag_span(app, line_start_after_indent, text, p + 1, q, "icode");
                p = q + 1;
                continue;
            }
        } else if (p[0] == '*' && p[1] == '*') {
            const char *q = strstr(p + 2, "**");

            if (q && q > p + 2) {
                tag_span(app, line_start_after_indent, text, p + 2, q, "strong");
                p = q + 2;
                continue;
            }
        } else if (g_str_has_prefix(p, "http://") ||
                   g_str_has_prefix(p, "https://")) {
            const char *q = p;

            while (*q && !g_ascii_isspace(*q) && *q != ')' &&
                   *q != '"' && *q != '<')
                q++;
            tag_span(app, line_start_after_indent, text, p, q, "ulink");
            p = q;
            continue;
        }
        p++;
    }
}

/*
 * Re-scan the whole chat buffer: clear all tags and apply the colourful
 * representation. Runs once per completed reply (cheap even for long
 * logs), so streaming stays plain and only the final render is decorated.
 */
static void re_render_chat(App *app)
{
    GtkTextIter line, s, e, fence_open = { 0 };
    gboolean in_code = FALSE;

    ensure_tags(app);
    gtk_text_buffer_get_bounds(app->chat_buf, &s, &e);
    gtk_text_buffer_remove_all_tags(app->chat_buf, &s, &e);

    gtk_text_buffer_get_start_iter(app->chat_buf, &line);
    while (TRUE) {
        GtkTextIter lend = line, t0;
        char *ltxt;
        const char *p;
        gsize mark;

        gtk_text_iter_forward_to_line_end(&lend);
        ltxt = gtk_text_buffer_get_text(app->chat_buf, &line, &lend, FALSE);
        p = ltxt;
        while (*p == ' ' || *p == '\t')
            p++;
        t0 = line;
        gtk_text_iter_forward_chars(&t0, (gint)(p - ltxt)); /* past indent */

        if (g_str_has_prefix(p, "```")) {
            gtk_text_buffer_apply_tag_by_name(app->chat_buf, "fence", &line, &lend);
            if (!in_code) {
                fence_open = line;
                in_code = TRUE;
            } else {
                GtkTextIter body = fence_open;

                gtk_text_iter_forward_line(&body);
                gtk_text_buffer_apply_tag_by_name(app->chat_buf, "code",
                                                  &body, &line);
                highlight_code(app, &body, &line);
                in_code = FALSE;
            }
        } else if (in_code) {
            /* tagged when the closing fence is reached */
        } else if (*p == '#') {
            int n = 1;

            while (p[n] == '#')
                n++;
            gtk_text_buffer_apply_tag_by_name(app->chat_buf,
                n == 1 ? "h1" : n == 2 ? "h2" : "h3", &line, &lend);
        } else if (g_str_has_prefix(p, "> ")) {
            gtk_text_buffer_apply_tag_by_name(app->chat_buf, "quote", &line, &lend);
        } else if ((mark = list_marker_len(p)) > 0) {
            GtkTextIter m1 = t0;

            gtk_text_iter_forward_chars(&m1, (gint)mark); /* ASCII markers */
            gtk_text_buffer_apply_tag_by_name(app->chat_buf, "bullet", &t0, &m1);
            highlight_inline_md(app, &m1, p + mark);
        } else if (g_str_has_prefix(p, "[You")) {
            gtk_text_buffer_apply_tag_by_name(app->chat_buf, "urole", &line, &lend);
        } else if (g_str_has_prefix(p, "[Assistant")) {
            gtk_text_buffer_apply_tag_by_name(app->chat_buf, "arole", &line, &lend);
        } else if (g_str_has_prefix(p, "[Error")) {
            gtk_text_buffer_apply_tag_by_name(app->chat_buf, "err", &line, &lend);
        } else if (*p) {
            highlight_inline_md(app, &t0, p);
        }

        g_free(ltxt);
        if (!gtk_text_iter_forward_line(&line))
            break;
    }
    if (in_code) {                      /* unterminated fence (mid-stream) */
        GtkTextIter body = fence_open, e2;

        gtk_text_iter_forward_line(&body);
        gtk_text_buffer_get_end_iter(app->chat_buf, &e2);
        gtk_text_buffer_apply_tag_by_name(app->chat_buf, "code", &body, &e2);
        highlight_code(app, &body, &e2);
    }
}

static void on_colour_toggled(GtkToggleButton *tb, gpointer data)
{
    App *app = data;

    if (gtk_check_button_get_active(GTK_CHECK_BUTTON(tb)))
        re_render_chat(app);
    else {
        GtkTextIter s, e;

        gtk_text_buffer_get_bounds(app->chat_buf, &s, &e);
        gtk_text_buffer_remove_all_tags(app->chat_buf, &s, &e);
    }
}

/* decorate once, when a reply is complete */
static void finish_render(App *app)
{
    if (gtk_check_button_get_active(GTK_CHECK_BUTTON(app->colour_cb)))
        re_render_chat(app);
}

/* ------------------------------------------------------------------ */
/* request building                                                    */
/* ------------------------------------------------------------------ */

static void add_number_option(JsonObject *o, const char *key,
                              GtkWidget *entry, gboolean as_int)
{
    const char *t = gtk_editable_get_text(GTK_EDITABLE(entry));
    char *s, *endp = NULL;

    if (!t)
        return;
    s = g_strstrip(g_strdup(t));
    if (*s) {
        if (as_int) {
            gint64 v = g_ascii_strtoll(s, &endp, 10);

            if (endp && *endp == '\0')
                json_object_set_int_member(o, key, v);
        } else {
            double v = g_ascii_strtod(s, &endp);

            if (endp && *endp == '\0')
                json_object_set_double_member(o, key, v);
        }
    }
    g_free(s);
}

static JsonObject *collect_options(App *app)
{
    JsonObject *o = json_object_new();

    add_number_option(o, "temperature", app->temp_e, FALSE);
    add_number_option(o, "top_p", app->topp_e, FALSE);
    add_number_option(o, "num_ctx", app->ctx_e, TRUE);
    add_number_option(o, "num_predict", app->npred_e, TRUE);
    add_number_option(o, "seed", app->seed_e, TRUE);
    if (json_object_get_size(o) == 0) {
        json_object_unref(o);
        return NULL;
    }
    return o;
}

static void add_string_array(JsonBuilder *b, const char *member,
                             GPtrArray *strings)
{
    guint i;

    json_builder_set_member_name(b, member);
    json_builder_begin_array(b);
    for (i = 0; i < strings->len; i++)
        json_builder_add_string_value(b, g_ptr_array_index(strings, i));
    json_builder_end_array(b);
}

/*
 * Returns the payload node (caller frees), and through umsg_out the user
 * message that was appended, so the caller can push it into the history.
 */
static JsonNode *build_payload(App *app, const char *text, int *endpoint_out,
                               JsonNode **umsg_out, GError **err)
{
    GString *content = g_string_new(NULL);
    GPtrArray *images = g_ptr_array_new_with_free_func(g_free);
    JsonNode *schema = NULL;
    const char *simple_format = NULL;
    JsonBuilder *b;
    JsonNode *umsg = NULL;
    JsonNode *root = NULL;
    JsonObject *opts;
    char *sys = NULL;
    char *keep = NULL;
    guint fsel;
    int ep = selected_endpoint(app);
    gboolean streaming = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->stream_cb));

    if (!build_user_content(app, text, content, images, err))
        goto fail;

    fsel = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->format_dd));
    if (fsel == 1) {
        simple_format = "json";
    } else if (fsel == 2) {
        char *raw = buffer_text(app->schema_buf);

        g_strstrip(raw);
        if (*raw == '\0') {
            simple_format = "json";
        } else {
            JsonParser *p = json_parser_new();

            if (!json_parser_load_from_data(p, raw, -1, err)) {
                g_object_unref(p);
                g_free(raw);
                goto fail;
            }
            schema = json_node_copy(json_parser_get_root(p));
            g_object_unref(p);
        }
        g_free(raw);
    }

    if (schema && gtk_check_button_get_active(GTK_CHECK_BUTTON(app->ground_cb))) {
        char *ss = node_to_string(schema, FALSE);

        g_string_append_printf(content,
                               "\n\nRespond with JSON matching this schema:\n%s",
                               ss);
        g_free(ss);
    }

    /* the user message */
    b = json_builder_new();
    json_builder_begin_object(b);
    json_builder_set_member_name(b, "role");
    json_builder_add_string_value(b, "user");
    json_builder_set_member_name(b, "content");
    json_builder_add_string_value(b, content->str);
    if (images->len)
        add_string_array(b, "images", images);
    json_builder_end_object(b);
    umsg = json_builder_get_root(b);
    g_object_unref(b);

    /* the payload */
    sys = buffer_text(app->sys_buf);
    g_strstrip(sys);
    b = json_builder_new();
    json_builder_begin_object(b);
    json_builder_set_member_name(b, "model");
    json_builder_add_string_value(b, selected_model(app));
    json_builder_set_member_name(b, "stream");
    json_builder_add_boolean_value(b, streaming);

    if (ep == 0) {
        json_builder_set_member_name(b, "messages");
        json_builder_begin_array(b);
        if (*sys) {
            json_builder_begin_object(b);
            json_builder_set_member_name(b, "role");
            json_builder_add_string_value(b, "system");
            json_builder_set_member_name(b, "content");
            json_builder_add_string_value(b, sys);
            json_builder_end_object(b);
        }
        if (gtk_check_button_get_active(GTK_CHECK_BUTTON(app->hist_cb))) {
            guint i, n = json_array_get_length(app->messages);

            for (i = 0; i < n; i++)
                json_builder_add_value(b,
                    json_node_copy(json_array_get_element(app->messages, i)));
        }
        json_builder_add_value(b, json_node_copy(umsg));
        json_builder_end_array(b);
    } else {
        json_builder_set_member_name(b, "prompt");
        json_builder_add_string_value(b, content->str);
        if (images->len)
            add_string_array(b, "images", images);
        if (*sys) {
            json_builder_set_member_name(b, "system");
            json_builder_add_string_value(b, sys);
        }
    }

    if (simple_format) {
        json_builder_set_member_name(b, "format");
        json_builder_add_string_value(b, simple_format);
    } else if (schema) {
        json_builder_set_member_name(b, "format");
        json_builder_add_value(b, json_node_copy(schema));
    }

    opts = collect_options(app);
    if (opts) {
        JsonNode *n = json_node_new(JSON_NODE_OBJECT);

        json_node_take_object(n, opts);
        json_builder_set_member_name(b, "options");
        json_builder_add_value(b, n);
    }

    keep = g_strstrip(g_strdup(gtk_editable_get_text(GTK_EDITABLE(app->keep_e))));
    if (*keep) {
        json_builder_set_member_name(b, "keep_alive");
        json_builder_add_string_value(b, keep);
    }
    if (gtk_check_button_get_active(GTK_CHECK_BUTTON(app->think_cb))) {
        json_builder_set_member_name(b, "think");
        json_builder_add_boolean_value(b, TRUE);
    }
    json_builder_end_object(b);
    root = json_builder_get_root(b);
    g_object_unref(b);

    *endpoint_out = ep;
    *umsg_out = umsg;
    g_free(sys);
    g_free(keep);
    if (schema)
        json_node_free(schema);
    g_string_free(content, TRUE);
    g_ptr_array_free(images, TRUE);
    return root;

fail:
    if (schema)
        json_node_free(schema);
    g_string_free(content, TRUE);
    g_ptr_array_free(images, TRUE);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* transport: worker thread + idle callbacks into the main loop        */
/* ------------------------------------------------------------------ */

typedef struct {
    App  *app;
    char *url;
    char *body;
    char *auth;                /* Authorization header, or NULL */
    char *model;               /* for the token ledger          */
    gint64 in_tokens;
    gint64 out_tokens;
    gboolean streaming;
    int   endpoint;
    GString *acc;      /* assistant text, for the history        */
    GString *pending;  /* partial NDJSON line                    */
    gboolean thinking_open;
    gboolean got_error;
    long  http_code;
} Req;

typedef struct {
    App  *app;
    char *text;
} UiText;

typedef struct {
    App  *app;
    char *response;
    char *status;
    char *model;
    gint64 in_tokens;
    gint64 out_tokens;
} UiDone;

static void ui_text_free(gpointer p)
{
    UiText *u = p;

    g_free(u->text);
    g_free(u);
}

static gboolean ui_append_cb(gpointer p)
{
    UiText *u = p;

    chat_append(u->app, u->text);
    return G_SOURCE_REMOVE;
}

static gboolean ui_status_cb(gpointer p)
{
    UiText *u = p;

    set_status(u->app, "%s", u->text);
    return G_SOURCE_REMOVE;
}

/* takes ownership of text */
static void post_text(App *app, char *text)
{
    UiText *u = g_new0(UiText, 1);

    u->app = app;
    u->text = text;
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, ui_append_cb, u, ui_text_free);
}

static void post_status(App *app, char *text)
{
    UiText *u = g_new0(UiText, 1);

    u->app = app;
    u->text = text;
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, ui_status_cb, u, ui_text_free);
}

static void ui_done_free(gpointer p)
{
    UiDone *d = p;

    g_free(d->response);
    g_free(d->status);
    g_free(d->model);
    g_free(d);
}

static gboolean ui_done_cb(gpointer p)
{
    UiDone *d = p;
    App *app = d->app;

    if (d->response && *d->response) {
        JsonBuilder *b = json_builder_new();

        g_free(app->last_response);
        app->last_response = g_strdup(d->response);

        json_builder_begin_object(b);
        json_builder_set_member_name(b, "role");
        json_builder_add_string_value(b, "assistant");
        json_builder_set_member_name(b, "content");
        json_builder_add_string_value(b, d->response);
        json_builder_end_object(b);
        json_array_add_element(app->messages, json_builder_get_root(b));
        g_object_unref(b);
    }

    /* fold this exchange into the ledger and persist it */
    if (d->in_tokens > 0 || d->out_tokens > 0) {
        usage_record(app, d->model, d->in_tokens, d->out_tokens);
        save_state(app);
    }

    chat_append(app, "\n");
    set_status(app, "%s", d->status ? d->status : "Done");

    app->busy = FALSE;
    gtk_spinner_stop(GTK_SPINNER(app->spinner));
    gtk_widget_set_sensitive(app->send_btn, TRUE);
    gtk_widget_set_sensitive(app->stop_btn, FALSE);

    finish_render(app);                 /* decorate the reply */
    return G_SOURCE_REMOVE;
}

/* one complete NDJSON object (also used whole for non-streaming bodies) */
static void process_chunk_line(Req *r, const char *line)
{
    JsonParser *p;
    JsonNode *root;
    JsonObject *o;
    const char *errm, *piece = NULL, *think = NULL;

    while (*line == ' ' || *line == '\t' || *line == '\r')
        line++;
    if (!*line)
        return;

    p = json_parser_new();
    if (!json_parser_load_from_data(p, line, -1, NULL)) {
        g_object_unref(p);
        return;
    }
    root = json_parser_get_root(p);
    if (!JSON_NODE_HOLDS_OBJECT(root)) {
        g_object_unref(p);
        return;
    }
    o = json_node_get_object(root);

    errm = obj_str(o, "error");
    if (errm) {
        r->got_error = TRUE;
        post_text(r->app, g_strdup_printf("\n[Error] %s\n", errm));
        g_object_unref(p);
        return;
    }

    if (r->endpoint == 0) {             /* /api/chat */
        if (json_object_has_member(o, "message")) {
            JsonNode *mn = json_object_get_member(o, "message");

            if (JSON_NODE_HOLDS_OBJECT(mn)) {
                JsonObject *m = json_node_get_object(mn);

                piece = obj_str(m, "content");
                think = obj_str(m, "thinking");
            }
        }
    } else {                            /* /api/generate */
        piece = obj_str(o, "response");
        think = obj_str(o, "thinking");
    }

    if (think && *think) {
        if (!r->thinking_open) {
            post_text(r->app, g_strdup("[thinking] "));
            r->thinking_open = TRUE;
        }
        post_text(r->app, g_strdup(think));
    }
    if (piece && *piece) {
        if (r->thinking_open) {
            post_text(r->app, g_strdup("\n[/thinking]\n"));
            r->thinking_open = FALSE;
        }
        g_string_append(r->acc, piece);
        post_text(r->app, g_strdup(piece));
    }

    if (obj_true(o, "done")) {
        gint64 total = obj_int(o, "total_duration", 0);
        gint64 evals = obj_int(o, "eval_count", 0);
        gint64 prompt = obj_int(o, "prompt_eval_count", 0);

        if (prompt > 0)
            r->in_tokens = prompt;
        if (evals > 0)
            r->out_tokens = evals;
        if (evals > 0)
            post_status(r->app,
                        g_strdup_printf("generated %" G_GINT64_FORMAT
                                        " tokens in %.1f s",
                                        evals, (double)total / 1e9));
    }
    g_object_unref(p);
}

static size_t write_cb(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    Req *r = userdata;
    size_t n = size * nmemb, consumed = 0;
    char *buf, *nl;

    if (g_atomic_int_get(&r->app->cancel))
        return 0;                       /* abort the transfer */

    g_string_append_len(r->pending, ptr, (gssize)n);

    if (!r->streaming)
        return n;                       /* parse the whole body at the end */

    buf = r->pending->str;
    while ((nl = memchr(buf + consumed, '\n', r->pending->len - consumed))) {
        size_t off = (size_t)(nl - buf);
        char *line = g_strndup(buf + consumed, off - consumed);

        process_chunk_line(r, line);
        g_free(line);
        consumed = off + 1;
    }
    if (consumed)
        g_string_erase(r->pending, 0, (gssize)consumed);
    return n;
}

static void req_free(Req *r)
{
    g_free(r->url);
    g_free(r->body);
    g_free(r->auth);
    g_free(r->model);
    if (r->acc)
        g_string_free(r->acc, TRUE);
    if (r->pending)
        g_string_free(r->pending, TRUE);
    g_free(r);
}

static gpointer req_worker(gpointer data)
{
    Req *r = data;
    CURL *curl;
    CURLcode cc;
    struct curl_slist *hdrs = NULL;
    long code = 0;
    double total = 0.0;
    UiDone *d;
    gboolean cancelled;

    curl = curl_easy_init();
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    if (r->auth)
        hdrs = curl_slist_append(hdrs, r->auth);
    curl_easy_setopt(curl, CURLOPT_URL, r->url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, r->body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(r->body));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, r);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    cc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME, &total);
    r->http_code = code;
    cancelled = g_atomic_int_get(&r->app->cancel) != 0;

    /* flush a trailing line that never saw '\n' */
    if (cc == CURLE_OK && r->streaming && r->pending->len > 0) {
        char *line = g_strndup(r->pending->str, r->pending->len);

        g_string_truncate(r->pending, 0);
        process_chunk_line(r, line);
        g_free(line);
    }
    /* non-streaming: the whole response body is one JSON object */
    if (cc == CURLE_OK && !r->streaming && r->pending->len > 0)
        process_chunk_line(r, r->pending->str);

    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);

    if (cc != CURLE_OK && !cancelled)
        post_text(r->app, g_strdup_printf("\n[Error] %s\n",
                                          curl_easy_strerror(cc)));

    d = g_new0(UiDone, 1);
    d->app = r->app;
    d->model = g_strdup(r->model ? r->model : "");
    d->in_tokens = r->in_tokens;
    d->out_tokens = r->out_tokens;
    if (r->acc->len)
        d->response = g_strdup(r->acc->str);
    if (cancelled)
        d->status = g_strdup("Stopped by user");
    else if (cc == CURLE_OK && r->got_error)
        d->status = g_strdup_printf("HTTP %ld — server reported an error", code);
    else if (cc == CURLE_OK)
        d->status = g_strdup_printf("HTTP %ld · %.1f s · %" G_GSIZE_FORMAT
                                    " chars", code, total, r->acc->len);
    else
        d->status = g_strdup_printf("Request failed: %s",
                                    curl_easy_strerror(cc));
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, ui_done_cb, d, ui_done_free);

    req_free(r);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* model list (/api/tags)                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    App       *app;
    char      *url;     /* built on the main thread */
    GPtrArray *names;   /* char*                    */
    char      *error;
} ModelsResult;

static void models_result_free(gpointer p)
{
    ModelsResult *r = p;

    if (r->names)
        g_ptr_array_free(r->names, TRUE);
    g_free(r->url);
    g_free(r->error);
    g_free(r);
}

static size_t get_write_cb(char *ptr, size_t size, size_t nmemb, void *ud)
{
    GString *s = ud;

    g_string_append_len(s, ptr, (gssize)(size * nmemb));
    return size * nmemb;
}

static gboolean ui_models_cb(gpointer p);

static gpointer models_worker(gpointer data)
{
    ModelsResult *r = data;
    CURL *curl = curl_easy_init();
    GString *body = g_string_new(NULL);
    CURLcode cc;
    long code = 0;

    curl_easy_setopt(curl, CURLOPT_URL, r->url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, get_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, body);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    cc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(curl);

    if (cc != CURLE_OK) {
        r->error = g_strdup(curl_easy_strerror(cc));
    } else if (code != 200) {
        r->error = g_strdup_printf("HTTP %ld", code);
    } else {
        JsonParser *p = json_parser_new();

        if (json_parser_load_from_data(p, body->str, (gssize)body->len, NULL)) {
            JsonNode *rootn = json_parser_get_root(p);

            if (JSON_NODE_HOLDS_OBJECT(rootn)) {
                JsonObject *root = json_node_get_object(rootn);

                if (json_object_has_member(root, "models")) {
                    JsonNode *mn = json_object_get_member(root, "models");

                    if (JSON_NODE_HOLDS_ARRAY(mn)) {
                        JsonArray *arr = json_node_get_array(mn);
                        guint i, n = json_array_get_length(arr);

                        for (i = 0; i < n; i++) {
                            JsonNode *el = json_array_get_element(arr, i);

                            if (JSON_NODE_HOLDS_OBJECT(el)) {
                                const char *name =
                                    obj_str(json_node_get_object(el), "name");

                                if (name)
                                    g_ptr_array_add(r->names, g_strdup(name));
                            }
                        }
                    }
                }
            }
        } else {
            r->error = g_strdup("Invalid JSON from /api/tags");
        }
        g_object_unref(p);
    }
    g_string_free(body, TRUE);
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, ui_models_cb, r,
                    models_result_free);
    return NULL;
}

static gboolean ui_models_cb(gpointer p)
{
    ModelsResult *r = p;
    App *app = r->app;
    guint i, n_old;
    const char **items;

    if (!r->names || r->names->len == 0) {
        set_status(app, "No models: %s",
                   r->error ? r->error : "empty /api/tags response");
        return G_SOURCE_REMOVE;
    }

    g_ptr_array_set_size(app->models, 0);       /* free old entries */
    items = g_new0(const char *, r->names->len + 1);
    for (i = 0; i < r->names->len; i++) {
        g_ptr_array_add(app->models,
                        g_strdup(g_ptr_array_index(r->names, i)));
        items[i] = g_ptr_array_index(app->models, i);
    }
    n_old = g_list_model_get_n_items(G_LIST_MODEL(app->model_list));
    gtk_string_list_splice(app->model_list, 0, n_old, items);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app->model_dd), 0);
    set_status(app, "%u model%s loaded",
               r->names->len, r->names->len == 1 ? "" : "s");
    g_free(items);
    return G_SOURCE_REMOVE;
}

static void start_models_load(App *app)
{
    ModelsResult *r = g_new0(ModelsResult, 1);
    char *base = base_url(app);

    r->app = app;
    r->url = g_strdup_printf("%s/api/tags", base);
    r->names = g_ptr_array_new_with_free_func(g_free);
    g_free(base);
    set_status(app, "Loading models…");
    g_thread_unref(g_thread_new("ollama-tags", models_worker, r));
}

static void on_refresh_models(GtkButton *b, gpointer data)
{
    (void)b;
    start_models_load(data);
}

/* ------------------------------------------------------------------ */
/* actions                                                             */
/* ------------------------------------------------------------------ */

static void send_request(App *app)
{
    GError *err = NULL;
    JsonNode *payload, *umsg = NULL;
    int ep;
    char *body, *url, *base, *ts, *hdr, *input, *trimmed;
    const char *model;
    Req *r;

    if (app->busy)
        return;
    model = selected_model(app);
    if (!model || !*model) {
        show_error(app, "No model selected — press the refresh button to load the model list.");
        return;
    }

    input = buffer_text(app->input_buf);
    trimmed = g_strstrip(input);
    if (!*trimmed && app->attachments->len == 0) {
        set_status(app, "Type a message or attach a file first");
        g_free(input);
        return;
    }

    payload = build_payload(app, trimmed, &ep, &umsg, &err);
    if (!payload) {
        show_error(app, err ? err->message : "Cannot build request");
        g_clear_error(&err);
        g_free(input);
        return;
    }

    body = node_to_string(payload, FALSE);
    g_free(app->last_payload);
    app->last_payload = g_strdup(body);

    /* echo the request into the transcript */
    ts = timestamp();
    hdr = g_strdup_printf("\n[You · %s]\n", ts);
    chat_append(app, hdr);
    g_free(hdr);
    g_free(ts);
    if (*trimmed) {
        chat_append(app, trimmed);
        chat_append(app, "\n");
    }
    if (app->attachments->len > 0) {
        hdr = g_strdup_printf("[%u attachment%s included]\n",
                              app->attachments->len,
                              app->attachments->len == 1 ? "" : "s");
        chat_append(app, hdr);
        g_free(hdr);
    }
    chat_append(app, "\n[Assistant]\n");

    base = base_url(app);
    url = g_strdup_printf("%s%s", base, ENDPOINTS[ep]);
    g_free(base);

    r = g_new0(Req, 1);
    r->app = app;
    r->url = url;
    r->body = body;                             /* ownership transferred */
    r->auth = auth_header(app);
    r->model = g_strdup(model);
    r->streaming = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->stream_cb));
    r->endpoint = ep;
    r->acc = g_string_new(NULL);
    r->pending = g_string_new(NULL);

    /* history: the payload was built with the old history; add ours now */
    json_array_add_element(app->messages, umsg);
    json_node_free(payload);

    app->busy = TRUE;
    g_atomic_int_set(&app->cancel, 0);
    gtk_widget_set_sensitive(app->send_btn, FALSE);
    gtk_widget_set_sensitive(app->stop_btn, TRUE);
    gtk_spinner_start(GTK_SPINNER(app->spinner));
    set_status(app, "Waiting for %s…", model);
    gtk_text_buffer_set_text(app->input_buf, "", -1);

    g_thread_unref(g_thread_new("ollama-req", req_worker, r));
}

static void on_send(GtkButton *b, gpointer data)
{
    (void)b;
    send_request(data);
}

static void on_stop(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    g_atomic_int_set(&app->cancel, 1);
    set_status(app, "Stopping…");
}

static gboolean on_input_key(GtkEventControllerKey *ctrl, guint keyval,
                             guint keycode, GdkModifierType state,
                             gpointer data)
{
    (void)ctrl;
    (void)keycode;
    if ((state & GDK_CONTROL_MASK) &&
        (keyval == GDK_KEY_Return || keyval == GDK_KEY_KP_Enter)) {
        send_request(data);
        return TRUE;
    }
    return FALSE;
}

static void on_preview_close(GtkButton *b, gpointer win)
{
    (void)b;
    gtk_window_destroy(GTK_WINDOW(win));
}

static void on_preview(GtkButton *btn, gpointer data)
{
    App *app = data;
    GError *err = NULL;
    JsonNode *payload, *umsg = NULL;
    int ep;
    char *input, *trimmed, *pretty, *title;
    GtkWidget *win, *box, *scr, *view, *close_btn;
    GtkTextBuffer *buf;

    (void)btn;
    input = buffer_text(app->input_buf);
    trimmed = g_strstrip(input);
    payload = build_payload(app, trimmed, &ep, &umsg, &err);
    g_free(input);
    if (!payload) {
        show_error(app, err ? err->message : "Cannot build request");
        g_clear_error(&err);
        return;
    }
    pretty = node_to_string(payload, TRUE);
    json_node_free(umsg);               /* preview discards the user node */
    json_node_free(payload);

    win = gtk_window_new();
    title = g_strdup_printf("Request preview — POST %s (exact bytes)",
                            ENDPOINTS[ep]);
    gtk_window_set_title(GTK_WINDOW(win), title);
    g_free(title);
    gtk_window_set_default_size(GTK_WINDOW(win), 680, 560);
    gtk_window_set_transient_for(GTK_WINDOW(win), app->win);
    gtk_window_set_modal(GTK_WINDOW(win), TRUE);

    box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_widget_set_margin_start(box, 10);
    gtk_widget_set_margin_end(box, 10);
    gtk_widget_set_margin_top(box, 10);
    gtk_widget_set_margin_bottom(box, 10);

    view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(view), FALSE);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(view), TRUE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(view), GTK_WRAP_NONE);
    buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(view));
    gtk_text_buffer_set_text(buf, pretty, -1);
    g_free(pretty);

    scr = gtk_scrolled_window_new();
    gtk_widget_set_vexpand(scr, TRUE);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scr), view);

    close_btn = gtk_button_new_with_label("Close");
    gtk_widget_set_halign(close_btn, GTK_ALIGN_END);
    g_signal_connect(close_btn, "clicked", G_CALLBACK(on_preview_close), win);

    gtk_box_append(GTK_BOX(box), scr);
    gtk_box_append(GTK_BOX(box), close_btn);
    gtk_window_set_child(GTK_WINDOW(win), box);
    gtk_window_present(GTK_WINDOW(win));
}

static void on_copy_reply(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    if (!app->last_response || !*app->last_response) {
        set_status(app, "No reply yet");
        return;
    }
    gdk_clipboard_set_text(gtk_widget_get_clipboard(app->chat_view),
                           app->last_response);
    set_status(app, "Reply copied (%" G_GSIZE_FORMAT " bytes)",
               strlen(app->last_response));
}

static void on_save_chat_done(GObject *src, GAsyncResult *res, gpointer data)
{
    App *app = data;
    GFile *f = gtk_file_dialog_save_finish(GTK_FILE_DIALOG(src), res, NULL);

    if (f) {
        char *txt = buffer_text(app->chat_buf);
        GError *err = NULL;

        if (g_file_replace_contents(f, txt, strlen(txt), NULL, FALSE,
                                    G_FILE_CREATE_NONE, NULL, NULL, &err))
            set_status(app, "Chat saved");
        else {
            show_error(app, err->message);
            g_clear_error(&err);
        }
        g_free(txt);
        g_object_unref(f);
    }
}

static void on_save_chat(GtkButton *b, gpointer data)
{
    App *app = data;
    GtkFileDialog *d = gtk_file_dialog_new();

    (void)b;
    gtk_file_dialog_set_title(d, "Save chat transcript");
    gtk_file_dialog_set_initial_name(d, "chat.md");
    gtk_file_dialog_save(d, app->win, NULL, on_save_chat_done, app);
    g_object_unref(d);
}

static void on_new_chat(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    if (app->busy) {
        set_status(app, "Stop the running request first");
        return;
    }
    json_array_unref(app->messages);
    app->messages = json_array_new();
    gtk_text_buffer_set_text(app->chat_buf, "", -1);
    g_clear_pointer(&app->last_response, g_free);
    g_clear_pointer(&app->last_payload, g_free);
    set_status(app, "New chat");
}

typedef struct {
    App     *app;
    LoadMode mode;
} AttachJob;

static void on_attach_done(GObject *src, GAsyncResult *res, gpointer data)
{
    AttachJob *j = data;
    GListModel *files =
        gtk_file_dialog_open_multiple_finish(GTK_FILE_DIALOG(src), res, NULL);

    if (files) {
        guint i, n = g_list_model_get_n_items(files);

        for (i = 0; i < n; i++) {
            GFile *f = g_list_model_get_item(files, i);
            char *path = g_file_get_path(f);

            if (path) {
                Attachment *a = g_new0(Attachment, 1);

                a->path = path;
                a->mode = j->mode;
                g_ptr_array_add(j->app->attachments, a);
            }
            g_object_unref(f);
        }
        refresh_attachment_list(j->app);
        set_status(j->app, "%u file%s attached",
                   j->app->attachments->len,
                   j->app->attachments->len == 1 ? "" : "s");
        g_object_unref(files);
    }
    g_free(j);
}

static void on_attach_files(GtkButton *b, gpointer data)
{
    App *app = data;
    AttachJob *j = g_new0(AttachJob, 1);
    GtkFileDialog *d = gtk_file_dialog_new();
    guint sel = gtk_drop_down_get_selected(GTK_DROP_DOWN(app->loadmode_dd));

    (void)b;
    j->app = app;
    j->mode = (sel == GTK_INVALID_LIST_POSITION) ? MODE_AUTO : (LoadMode)sel;
    gtk_file_dialog_set_title(d, "Attach files");
    gtk_file_dialog_open_multiple(d, app->win, NULL, on_attach_done, j);
    g_object_unref(d);
}

static void on_clear_attach(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    g_ptr_array_set_size(app->attachments, 0);
    refresh_attachment_list(app);
    set_status(app, "Attachments cleared");
}

/* ==================================================== */
/* server control and sign-in, model pulling,           */
/* token ledger with budget                             */
/* ==================================================== */

/* UI helpers defined further down, used by the page builders below */
static GtkWidget *section_label(const char *text);
static GtkWidget *prop_row(const char *label, GtkWidget *w);
static GtkWidget *make_text_area(GtkTextBuffer **buf_out, gboolean wrap,
                                 gboolean mono, int min_h);

/* ------------------------------------------------------------------ */
/* small shared helpers                                                */
/* ------------------------------------------------------------------ */

static const char *entry_str(GtkWidget *e)
{
    const char *t = gtk_editable_get_text(GTK_EDITABLE(e));

    return t ? t : "";
}

static double entry_num(GtkWidget *e)
{
    char *s = g_strstrip(g_strdup(entry_str(e)));
    double v = 0.0;

    if (*s)
        v = g_ascii_strtod(s, NULL);
    g_free(s);
    return v;
}

static char *today_string(void)
{
    GDateTime *t = g_date_time_new_now_local();
    char *s = g_date_time_format(t, "%Y-%m-%d");

    g_date_time_unref(t);
    return s;
}

static void log_append(App *app, const char *s)
{
    GtkTextIter end;
    char *safe;

    if (!s || !*s || !app->log_buf)
        return;
    safe = g_utf8_make_valid(s, -1);
    gtk_text_buffer_get_end_iter(app->log_buf, &end);
    gtk_text_buffer_insert(app->log_buf, &end, safe, -1);
    g_free(safe);
}

/* read-only variant of make_text_area */
static GtkWidget *ro_text_area(GtkTextBuffer **buf_out, gboolean wrap,
                               gboolean mono, int min_h)
{
    GtkWidget *scr = make_text_area(buf_out, wrap, mono, min_h);
    GtkWidget *v = gtk_scrolled_window_get_child(GTK_SCROLLED_WINDOW(scr));

    gtk_text_view_set_editable(GTK_TEXT_VIEW(v), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(v), FALSE);
    return scr;
}

/* A model is billed when it carries the cloud tag, or the host is ollama.com */
static gboolean model_is_cloud(App *app, const char *model)
{
    char *base;
    gboolean cloud_host;

    if (!model || !*model)
        return FALSE;
    if (g_str_has_suffix(model, "-cloud") || strstr(model, ":cloud"))
        return TRUE;
    base = base_url(app);
    cloud_host = (strstr(base, "ollama.com") != NULL);
    g_free(base);
    return cloud_host;
}

/* "Authorization: Bearer …", or NULL when no key is set */
static char *auth_header(App *app)
{
    char *key;
    char *h = NULL;

    if (!app->apikey_entry)
        return NULL;
    key = g_strstrip(g_strdup(entry_str(app->apikey_entry)));
    if (*key)
        h = g_strdup_printf("Authorization: Bearer %s", key);
    g_free(key);
    return h;
}

/* ------------------------------------------------------------------ */
/* token ledger                                                       */
/* ------------------------------------------------------------------ */

static void usage_free(gpointer p)
{
    Usage *u = p;

    g_free(u->model);
    g_free(u);
}

static Usage *usage_for(App *app, const char *model, gboolean cloud)
{
    guint i;
    Usage *u;

    for (i = 0; i < app->usage->len; i++) {
        u = g_ptr_array_index(app->usage, i);
        if (g_strcmp0(u->model, model) == 0)
            return u;
    }
    u = g_new0(Usage, 1);
    u->model = g_strdup(model);
    u->cloud = cloud;
    g_ptr_array_add(app->usage, u);
    return u;
}

static void render_usage(App *app)
{
    GString *t;
    double price_in, price_out, budget;
    const char *cur;
    double cost;
    gint64 bill_in = 0, bill_out = 0, free_in = 0, free_out = 0;
    char *summary;
    guint i;

    if (!app->usage_buf || !app->price_in_e)
        return;

    price_in = entry_num(app->price_in_e);
    price_out = entry_num(app->price_out_e);
    budget = entry_num(app->budget_e);
    cur = entry_str(app->currency_e);

    t = g_string_new(NULL);
    g_string_append_printf(t, "%-30s %5s %10s %10s %7s\n",
                           "model", "req", "in", "out", "billed");
    g_string_append(t, "---------------------------------"
                       "---------------------------\n");
    for (i = 0; i < app->usage->len; i++) {
        Usage *u = g_ptr_array_index(app->usage, i);

        g_string_append_printf(t, "%-30s %5" G_GINT64_FORMAT
                                  " %10" G_GINT64_FORMAT
                                  " %10" G_GINT64_FORMAT " %7s\n",
                               u->model, u->requests, u->prompt_tokens,
                               u->eval_tokens, u->cloud ? "yes" : "no");
        if (u->cloud) {
            bill_in += u->prompt_tokens;
            bill_out += u->eval_tokens;
        } else {
            free_in += u->prompt_tokens;
            free_out += u->eval_tokens;
        }
    }
    if (app->usage->len == 0)
        g_string_append(t, "(nothing recorded yet)\n");

    cost = ((double)bill_in * price_in + (double)bill_out * price_out) / 1.0e6;

    g_string_append_printf(t,
        "\nbilled    in %" G_GINT64_FORMAT "   out %" G_GINT64_FORMAT
        "\nlocal     in %" G_GINT64_FORMAT "   out %" G_GINT64_FORMAT
        "   (not billed)\n", bill_in, bill_out, free_in, free_out);
    gtk_text_buffer_set_text(app->usage_buf, t->str, -1);
    g_string_free(t, TRUE);

    if (budget > 0.0) {
        double within = (cost < budget) ? cost : budget;
        double extra = (cost > budget) ? cost - budget : 0.0;
        double left = (budget > cost) ? budget - cost : 0.0;

        summary = g_strdup_printf(
            "period from %s\nspend %.4f %s against a %.2f %s budget\n"
            "paid usage %.4f · extra usage %.4f · budget left %.4f %s",
            app->period_start ? app->period_start : "?",
            cost, cur, budget, cur, within, extra, left, cur);
    } else {
        summary = g_strdup_printf(
            "period from %s\nestimated spend %.4f %s "
            "(enter a budget above to track what is left and any overage)",
            app->period_start ? app->period_start : "?", cost, cur);
    }
    gtk_label_set_text(GTK_LABEL(app->usage_summary), summary);
    g_free(summary);
}

static void usage_record(App *app, const char *model, gint64 in_tok,
                         gint64 out_tok)
{
    gboolean cloud;
    Usage *u;

    if (!model || !*model || (in_tok <= 0 && out_tok <= 0))
        return;
    cloud = model_is_cloud(app, model);
    u = usage_for(app, model, cloud);
    u->cloud = u->cloud || cloud;
    u->requests += 1;
    if (in_tok > 0)
        u->prompt_tokens += in_tok;
    if (out_tok > 0)
        u->eval_tokens += out_tok;
    render_usage(app);
}

/* ------------------------------------------------------------------ */
/* configuration and ledger persistence                               */
/* ------------------------------------------------------------------ */

static char *state_path(void)
{
    return g_build_filename(g_get_user_config_dir(), "ollama-gui",
                            "state.json", NULL);
}

static void load_state(App *app)
{
    char *path = state_path();
    char *data = NULL;
    gsize len = 0;
    JsonParser *p;
    JsonObject *o = NULL;
    const char *env;
    const char *s;

    env = g_getenv("OLLAMA_API_KEY");
    app->cfg_host = g_strdup("http://localhost:11434");
    app->cfg_key = g_strdup(env ? env : "");
    app->cfg_price_in = g_strdup("0");
    app->cfg_price_out = g_strdup("0");
    app->cfg_budget = g_strdup("0");
    app->cfg_currency = g_strdup("USD");
    app->cfg_serve_cmd = g_strdup("ollama serve");
    app->cfg_signin_cmd = g_strdup("ollama signin");
    app->cfg_remember_key = FALSE;
    app->period_start = today_string();

    if (!g_file_get_contents(path, &data, &len, NULL)) {
        g_free(path);
        return;
    }
    p = json_parser_new();
    if (json_parser_load_from_data(p, data, (gssize)len, NULL)) {
        JsonNode *root = json_parser_get_root(p);

        if (root && JSON_NODE_HOLDS_OBJECT(root))
            o = json_node_get_object(root);
    }
    if (o) {
        JsonArray *arr;

#define TAKE_STR(field, key)                    \
        s = obj_str(o, key);                    \
        if (s && *s) {                          \
            g_free(app->field);                 \
            app->field = g_strdup(s);           \
        }
        TAKE_STR(cfg_host, "host")
        TAKE_STR(cfg_price_in, "price_in")
        TAKE_STR(cfg_price_out, "price_out")
        TAKE_STR(cfg_budget, "budget")
        TAKE_STR(cfg_currency, "currency")
        TAKE_STR(cfg_serve_cmd, "serve_cmd")
        TAKE_STR(cfg_signin_cmd, "signin_cmd")
        TAKE_STR(period_start, "period_start")
#undef TAKE_STR
        app->cfg_remember_key = obj_true(o, "remember_key");
        if (app->cfg_remember_key) {
            s = obj_str(o, "api_key");
            if (s && *s) {
                g_free(app->cfg_key);
                app->cfg_key = g_strdup(s);
            }
        }
        arr = json_object_has_member(o, "usage")
                  ? json_object_get_array_member(o, "usage") : NULL;
        if (arr) {
            guint i, n = json_array_get_length(arr);

            for (i = 0; i < n; i++) {
                JsonNode *el = json_array_get_element(arr, i);
                JsonObject *uo;
                Usage *u;

                if (!JSON_NODE_HOLDS_OBJECT(el))
                    continue;
                uo = json_node_get_object(el);
                if (!obj_str(uo, "model"))
                    continue;
                u = g_new0(Usage, 1);
                u->model = g_strdup(obj_str(uo, "model"));
                u->cloud = obj_true(uo, "cloud");
                u->requests = obj_int(uo, "requests", 0);
                u->prompt_tokens = obj_int(uo, "prompt_tokens", 0);
                u->eval_tokens = obj_int(uo, "eval_tokens", 0);
                g_ptr_array_add(app->usage, u);
            }
        }
    }
    g_object_unref(p);
    g_free(data);
    g_free(path);
}

static void save_state(App *app)
{
    char *path, *dir, *text;
    JsonBuilder *b;
    JsonNode *root;
    gboolean remember;
    guint i;

    if (!app->host_entry || !app->remember_key_cb)
        return;                          /* widgets not built yet */

    path = state_path();
    dir = g_path_get_dirname(path);
    g_mkdir_with_parents(dir, 0700);
    remember = gtk_check_button_get_active(GTK_CHECK_BUTTON(app->remember_key_cb));

    b = json_builder_new();
    json_builder_begin_object(b);
    json_builder_set_member_name(b, "host");
    json_builder_add_string_value(b, entry_str(app->host_entry));
    json_builder_set_member_name(b, "price_in");
    json_builder_add_string_value(b, entry_str(app->price_in_e));
    json_builder_set_member_name(b, "price_out");
    json_builder_add_string_value(b, entry_str(app->price_out_e));
    json_builder_set_member_name(b, "budget");
    json_builder_add_string_value(b, entry_str(app->budget_e));
    json_builder_set_member_name(b, "currency");
    json_builder_add_string_value(b, entry_str(app->currency_e));
    json_builder_set_member_name(b, "serve_cmd");
    json_builder_add_string_value(b, entry_str(app->serve_cmd_e));
    json_builder_set_member_name(b, "signin_cmd");
    json_builder_add_string_value(b, entry_str(app->signin_cmd_e));
    json_builder_set_member_name(b, "period_start");
    json_builder_add_string_value(b, app->period_start ? app->period_start : "");
    json_builder_set_member_name(b, "remember_key");
    json_builder_add_boolean_value(b, remember);
    json_builder_set_member_name(b, "api_key");
    json_builder_add_string_value(b, remember ? entry_str(app->apikey_entry) : "");

    json_builder_set_member_name(b, "usage");
    json_builder_begin_array(b);
    for (i = 0; i < app->usage->len; i++) {
        Usage *u = g_ptr_array_index(app->usage, i);

        json_builder_begin_object(b);
        json_builder_set_member_name(b, "model");
        json_builder_add_string_value(b, u->model);
        json_builder_set_member_name(b, "cloud");
        json_builder_add_boolean_value(b, u->cloud);
        json_builder_set_member_name(b, "requests");
        json_builder_add_int_value(b, u->requests);
        json_builder_set_member_name(b, "prompt_tokens");
        json_builder_add_int_value(b, u->prompt_tokens);
        json_builder_set_member_name(b, "eval_tokens");
        json_builder_add_int_value(b, u->eval_tokens);
        json_builder_end_object(b);
    }
    json_builder_end_array(b);
    json_builder_end_object(b);

    root = json_builder_get_root(b);
    text = node_to_string(root, TRUE);
    if (g_file_set_contents(path, text, -1, NULL))
        g_chmod(path, 0600);             /* the file can hold an API key */

    g_free(text);
    json_node_free(root);
    g_object_unref(b);
    g_free(dir);
    g_free(path);
}

static void on_usage_apply(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    render_usage(app);
    save_state(app);
    set_status(app, "Prices and budget applied");
}

static void on_usage_reset(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    g_ptr_array_set_size(app->usage, 0);
    g_free(app->period_start);
    app->period_start = today_string();
    render_usage(app);
    save_state(app);
    set_status(app, "Usage period restarted");
}

static void on_save_usage_done(GObject *src, GAsyncResult *res, gpointer data)
{
    App *app = data;
    GError *err = NULL;
    GFile *f = gtk_file_dialog_save_finish(GTK_FILE_DIALOG(src), res, &err);
    char *path, *text;

    if (!f) {
        g_clear_error(&err);
        return;
    }
    path = g_file_get_path(f);
    g_object_unref(f);
    if (!path)
        return;
    text = buffer_text(app->usage_buf);
    if (!g_file_set_contents(path, text, -1, &err)) {
        show_error(app, err->message);
        g_clear_error(&err);
    } else {
        set_status(app, "Written: %s", path);
    }
    g_free(text);
    g_free(path);
}

static void on_save_usage(GtkButton *b, gpointer data)
{
    App *app = data;
    GtkFileDialog *d = gtk_file_dialog_new();

    (void)b;
    gtk_file_dialog_set_title(d, "Save usage table");
    gtk_file_dialog_set_initial_name(d, "usage.txt");
    gtk_file_dialog_save(d, app->win, NULL, on_save_usage_done, app);
    g_object_unref(d);
}

/* ------------------------------------------------------------------ */
/* server version probe (async)                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    App  *app;
    char *url;
    char *auth;
    char *text;                          /* filled in by the worker */
} Probe;

static void probe_free(gpointer p)
{
    Probe *pr = p;

    g_free(pr->url);
    g_free(pr->auth);
    g_free(pr->text);
    g_free(pr);
}

static gboolean ui_probe_cb(gpointer p)
{
    Probe *pr = p;

    if (pr->app->version_label)
        gtk_label_set_text(GTK_LABEL(pr->app->version_label),
                           pr->text ? pr->text : "server: unknown");
    return G_SOURCE_REMOVE;
}

static gpointer probe_worker(gpointer data)
{
    Probe *pr = data;
    CURL *curl = curl_easy_init();
    GString *body = g_string_new(NULL);
    struct curl_slist *hdrs = NULL;
    CURLcode cc;
    long code = 0;

    if (pr->auth)
        hdrs = curl_slist_append(hdrs, pr->auth);
    curl_easy_setopt(curl, CURLOPT_URL, pr->url);
    if (hdrs)
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, get_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, body);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 6L);
    cc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (hdrs)
        curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);

    if (cc != CURLE_OK) {
        pr->text = g_strdup_printf("server: not reachable (%s)",
                                   curl_easy_strerror(cc));
    } else if (code != 200) {
        pr->text = g_strdup_printf("server: HTTP %ld from /api/version", code);
    } else {
        JsonParser *p = json_parser_new();
        const char *v = NULL;

        if (json_parser_load_from_data(p, body->str, (gssize)body->len, NULL)) {
            JsonNode *root = json_parser_get_root(p);

            if (root && JSON_NODE_HOLDS_OBJECT(root))
                v = obj_str(json_node_get_object(root), "version");
        }
        pr->text = g_strdup_printf("server: running, version %s",
                                   v ? v : "unknown");
        g_object_unref(p);
    }
    g_string_free(body, TRUE);
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, ui_probe_cb, pr, probe_free);
    return NULL;
}

static void start_version_probe(App *app)
{
    Probe *pr = g_new0(Probe, 1);
    char *base = base_url(app);

    pr->app = app;
    pr->url = g_strdup_printf("%s/api/version", base);
    pr->auth = auth_header(app);
    g_free(base);
    g_thread_unref(g_thread_new("ollama-version", probe_worker, pr));
}

static void on_probe_clicked(GtkButton *b, gpointer data)
{
    (void)b;
    start_version_probe(data);
}

/* ------------------------------------------------------------------ */
/* installed-model inventory: /api/tags plus /api/ps (async)          */
/* ------------------------------------------------------------------ */

typedef struct {
    App  *app;
    char *tags_url;
    char *ps_url;
    char *auth;
    char *table;                         /* filled in by the worker */
} Inventory;

static void inventory_free(gpointer p)
{
    Inventory *inv = p;

    g_free(inv->tags_url);
    g_free(inv->ps_url);
    g_free(inv->auth);
    g_free(inv->table);
    g_free(inv);
}

static gboolean ui_inventory_cb(gpointer p)
{
    Inventory *inv = p;

    if (inv->app->models_buf)
        gtk_text_buffer_set_text(inv->app->models_buf,
                                 inv->table ? inv->table : "", -1);
    return G_SOURCE_REMOVE;
}

static long simple_get(const char *url, const char *auth, GString *body)
{
    CURL *curl = curl_easy_init();
    struct curl_slist *hdrs = NULL;
    long code = 0;

    if (!curl)
        return 0;
    if (auth)
        hdrs = curl_slist_append(hdrs, auth);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    if (hdrs)
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, get_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, body);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    if (curl_easy_perform(curl) == CURLE_OK)
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (hdrs)
        curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);
    return code;
}

static JsonArray *models_array_of(JsonParser *p, const char *body, gsize len)
{
    JsonNode *root;
    JsonObject *o;

    if (!json_parser_load_from_data(p, body, (gssize)len, NULL))
        return NULL;
    root = json_parser_get_root(p);
    if (!root || !JSON_NODE_HOLDS_OBJECT(root))
        return NULL;
    o = json_node_get_object(root);
    if (!json_object_has_member(o, "models"))
        return NULL;
    if (!JSON_NODE_HOLDS_ARRAY(json_object_get_member(o, "models")))
        return NULL;
    return json_object_get_array_member(o, "models");
}

static gpointer inventory_worker(gpointer data)
{
    Inventory *inv = data;
    GString *tags = g_string_new(NULL);
    GString *ps = g_string_new(NULL);
    GString *table = g_string_new(NULL);
    GPtrArray *loaded = g_ptr_array_new_with_free_func(g_free);
    JsonParser *pt = json_parser_new();
    JsonParser *pp = json_parser_new();
    JsonArray *arr;
    guint i, j, n;

    if (simple_get(inv->tags_url, inv->auth, tags) != 200) {
        g_string_append(table, "(cannot read /api/tags — is the server "
                               "running? see the Server page)\n");
        goto out;
    }
    if (simple_get(inv->ps_url, inv->auth, ps) == 200) {
        JsonArray *pa = models_array_of(pp, ps->str, ps->len);

        n = pa ? json_array_get_length(pa) : 0;
        for (i = 0; i < n; i++) {
            JsonNode *el = json_array_get_element(pa, i);
            const char *nm;

            if (!JSON_NODE_HOLDS_OBJECT(el))
                continue;
            nm = obj_str(json_node_get_object(el), "name");
            if (!nm)
                nm = obj_str(json_node_get_object(el), "model");
            if (nm)
                g_ptr_array_add(loaded, g_strdup(nm));
        }
    }

    g_string_append_printf(table, "%-32s %8s %8s %8s %7s\n",
                           "model", "size", "params", "quant", "loaded");
    g_string_append(table, "-----------------------------------"
                           "---------------------------\n");

    arr = models_array_of(pt, tags->str, tags->len);
    n = arr ? json_array_get_length(arr) : 0;
    for (i = 0; i < n; i++) {
        JsonNode *el = json_array_get_element(arr, i);
        JsonObject *mo, *det = NULL;
        const char *name, *params = "", *quant = "";
        gint64 size;
        gboolean is_loaded = FALSE;

        if (!JSON_NODE_HOLDS_OBJECT(el))
            continue;
        mo = json_node_get_object(el);
        name = obj_str(mo, "name");
        if (!name)
            name = obj_str(mo, "model");
        if (!name)
            continue;
        if (json_object_has_member(mo, "details") &&
            JSON_NODE_HOLDS_OBJECT(json_object_get_member(mo, "details")))
            det = json_object_get_object_member(mo, "details");
        if (det) {
            const char *s = obj_str(det, "parameter_size");
            const char *q = obj_str(det, "quantization_level");

            if (s)
                params = s;
            if (q)
                quant = q;
        }
        size = obj_int(mo, "size", 0);
        for (j = 0; j < loaded->len; j++) {
            if (g_strcmp0(g_ptr_array_index(loaded, j), name) == 0) {
                is_loaded = TRUE;
                break;
            }
        }
        g_string_append_printf(table, "%-32s %7.2fG %8s %8s %7s\n",
                               name, (double)size / 1.0e9, params, quant,
                               is_loaded ? "yes" : "");
    }
    if (n == 0)
        g_string_append(table, "(no models installed — pull one above)\n");

out:
    inv->table = g_string_free(table, FALSE);
    g_object_unref(pt);
    g_object_unref(pp);
    g_ptr_array_free(loaded, TRUE);
    g_string_free(tags, TRUE);
    g_string_free(ps, TRUE);
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, ui_inventory_cb, inv,
                    inventory_free);
    return NULL;
}

static void start_inventory_load(App *app)
{
    Inventory *inv = g_new0(Inventory, 1);
    char *base = base_url(app);

    inv->app = app;
    inv->tags_url = g_strdup_printf("%s/api/tags", base);
    inv->ps_url = g_strdup_printf("%s/api/ps", base);
    inv->auth = auth_header(app);
    g_free(base);
    g_thread_unref(g_thread_new("ollama-inventory", inventory_worker, inv));
}

static void on_refresh_all(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    start_models_load(app);
    start_inventory_load(app);
    start_version_probe(app);
}

/* ------------------------------------------------------------------ */
/* pulling a model (/api/pull, streaming progress)                    */
/* ------------------------------------------------------------------ */

typedef struct {
    App     *app;
    char    *url;
    char    *body;
    char    *auth;
    GString *pending;
    long     http_code;
} PullReq;

typedef struct {
    App     *app;
    char    *text;
    double   fraction;                   /* < 0 leaves the bar alone */
    gboolean done;
} UiPull;

static void ui_pull_free(gpointer p)
{
    UiPull *u = p;

    g_free(u->text);
    g_free(u);
}

static gboolean ui_pull_cb(gpointer p)
{
    UiPull *u = p;
    App *app = u->app;

    if (u->fraction >= 0.0)
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(app->pull_bar),
                                      u->fraction);
    if (u->text)
        gtk_label_set_text(GTK_LABEL(app->pull_status), u->text);
    if (u->done) {
        app->pulling = FALSE;
        gtk_widget_set_sensitive(app->pull_btn, TRUE);
        start_models_load(app);          /* the dropdown picks it up */
        start_inventory_load(app);
    }
    return G_SOURCE_REMOVE;
}

static void post_pull(App *app, const char *text, double fraction,
                      gboolean done)
{
    UiPull *u = g_new0(UiPull, 1);

    u->app = app;
    u->text = text ? g_strdup(text) : NULL;
    u->fraction = fraction;
    u->done = done;
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, ui_pull_cb, u, ui_pull_free);
}

static void handle_pull_line(PullReq *r, const char *line)
{
    JsonParser *p;
    JsonNode *root;
    JsonObject *o = NULL;
    const char *status, *errm;
    gint64 total, completed;
    double frac = -1.0;

    while (*line == ' ' || *line == '\t' || *line == '\r')
        line++;
    if (!*line)
        return;

    p = json_parser_new();
    if (!json_parser_load_from_data(p, line, -1, NULL)) {
        g_object_unref(p);
        return;
    }
    root = json_parser_get_root(p);
    if (root && JSON_NODE_HOLDS_OBJECT(root))
        o = json_node_get_object(root);

    errm = obj_str(o, "error");
    if (errm) {
        post_pull(r->app, errm, -1.0, FALSE);
        g_object_unref(p);
        return;
    }
    status = obj_str(o, "status");
    total = obj_int(o, "total", 0);
    completed = obj_int(o, "completed", 0);
    if (total > 0)
        frac = (double)completed / (double)total;
    if (status) {
        char *label;

        if (total > 0)
            label = g_strdup_printf("%s  %.1f / %.1f MB", status,
                                    (double)completed / 1.0e6,
                                    (double)total / 1.0e6);
        else
            label = g_strdup(status);
        post_pull(r->app, label, frac, FALSE);
        g_free(label);
    }
    g_object_unref(p);
}

static size_t pull_write_cb(char *ptr, size_t size, size_t nmemb, void *ud)
{
    PullReq *r = ud;
    size_t n = size * nmemb, consumed = 0;
    char *buf, *nl;

    if (g_atomic_int_get(&r->app->pull_cancel))
        return 0;                        /* abort the transfer */

    g_string_append_len(r->pending, ptr, (gssize)n);
    buf = r->pending->str;
    while ((nl = memchr(buf + consumed, '\n', r->pending->len - consumed))) {
        size_t off = (size_t)(nl - buf);
        char *line = g_strndup(buf + consumed, off - consumed);

        handle_pull_line(r, line);
        g_free(line);
        consumed = off + 1;
    }
    if (consumed)
        g_string_erase(r->pending, 0, (gssize)consumed);
    return n;
}

static gpointer pull_worker(gpointer data)
{
    PullReq *r = data;
    CURL *curl = curl_easy_init();
    struct curl_slist *hdrs = NULL;
    CURLcode cc;

    if (!curl) {
        post_pull(r->app, "curl init failed", -1.0, TRUE);
        goto done;
    }
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    if (r->auth)
        hdrs = curl_slist_append(hdrs, r->auth);
    curl_easy_setopt(curl, CURLOPT_URL, r->url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, r->body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(r->body));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, pull_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, r);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    cc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r->http_code);

    if (cc == CURLE_WRITE_ERROR && g_atomic_int_get(&r->app->pull_cancel)) {
        post_pull(r->app, "pull cancelled", -1.0, TRUE);
    } else if (cc != CURLE_OK) {
        post_pull(r->app, curl_easy_strerror(cc), -1.0, TRUE);
    } else if (r->http_code != 200) {
        char *m = g_strdup_printf("HTTP %ld", r->http_code);

        post_pull(r->app, m, -1.0, TRUE);
        g_free(m);
    } else {
        post_pull(r->app, "pull complete", 1.0, TRUE);
    }
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);

done:
    g_free(r->url);
    g_free(r->body);
    g_free(r->auth);
    g_string_free(r->pending, TRUE);
    g_free(r);
    return NULL;
}

static void on_pull_clicked(GtkButton *b, gpointer data)
{
    App *app = data;
    char *name = g_strstrip(g_strdup(entry_str(app->pull_entry)));
    JsonBuilder *jb;
    JsonNode *root;
    PullReq *r;
    char *base;

    (void)b;
    if (app->pulling) {
        set_status(app, "A pull is already running");
        g_free(name);
        return;
    }
    if (!*name) {
        set_status(app, "Type a model name to pull, e.g. llama3.2:3b");
        g_free(name);
        return;
    }

    jb = json_builder_new();
    json_builder_begin_object(jb);
    json_builder_set_member_name(jb, "model");
    json_builder_add_string_value(jb, name);
    json_builder_set_member_name(jb, "stream");
    json_builder_add_boolean_value(jb, TRUE);
    json_builder_end_object(jb);
    root = json_builder_get_root(jb);
    g_object_unref(jb);

    base = base_url(app);
    r = g_new0(PullReq, 1);
    r->app = app;
    r->url = g_strdup_printf("%s/api/pull", base);
    r->body = node_to_string(root, FALSE);
    r->auth = auth_header(app);
    r->pending = g_string_new(NULL);
    g_free(base);
    json_node_free(root);

    app->pulling = TRUE;
    g_atomic_int_set(&app->pull_cancel, 0);
    gtk_widget_set_sensitive(app->pull_btn, FALSE);
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(app->pull_bar), 0.0);
    gtk_label_set_text(GTK_LABEL(app->pull_status), "starting pull…");
    set_status(app, "Pulling %s…", name);
    g_free(name);

    g_thread_unref(g_thread_new("ollama-pull", pull_worker, r));
}

static void on_pull_cancel(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    g_atomic_int_set(&app->pull_cancel, 1);
}

/* ------------------------------------------------------------------ */
/* running the server and signing in, as child processes              */
/* ------------------------------------------------------------------ */

static void read_next_line(App *app, GDataInputStream *in);

static void on_log_line(GObject *src, GAsyncResult *res, gpointer data)
{
    App *app = data;
    GDataInputStream *in = G_DATA_INPUT_STREAM(src);
    gsize len = 0;
    GError *err = NULL;
    char *line = g_data_input_stream_read_line_finish(in, res, &len, &err);

    if (!line) {
        g_clear_error(&err);
        g_object_unref(in);              /* EOF: release our reference */
        return;
    }
    log_append(app, line);
    log_append(app, "\n");
    g_free(line);
    read_next_line(app, in);
}

static void read_next_line(App *app, GDataInputStream *in)
{
    g_data_input_stream_read_line_async(in, G_PRIORITY_DEFAULT, NULL,
                                        on_log_line, app);
}

static void watch_output(App *app, GSubprocess *proc)
{
    GInputStream *out = g_subprocess_get_stdout_pipe(proc);

    if (out)
        read_next_line(app, g_data_input_stream_new(out));
}

static gboolean delayed_probe(gpointer data)
{
    App *app = data;

    start_version_probe(app);
    start_models_load(app);
    start_inventory_load(app);
    return G_SOURCE_REMOVE;
}

static void on_server_exit(GObject *src, GAsyncResult *res, gpointer data)
{
    App *app = data;
    GError *err = NULL;

    if (!g_subprocess_wait_check_finish(G_SUBPROCESS(src), res, &err)) {
        log_append(app, "[server exited] ");
        log_append(app, err ? err->message : "unknown reason");
        log_append(app, "\n");
        g_clear_error(&err);
    } else {
        log_append(app, "[server exited cleanly]\n");
    }
    if (app->server_proc == G_SUBPROCESS(src)) {
        g_object_unref(app->server_proc);
        app->server_proc = NULL;
        gtk_widget_set_sensitive(app->serve_btn, TRUE);
        gtk_widget_set_sensitive(app->serve_stop_btn, FALSE);
    }
    start_version_probe(app);
}

static void on_oneshot_exit(GObject *src, GAsyncResult *res, gpointer data)
{
    App *app = data;
    GError *err = NULL;

    if (!g_subprocess_wait_check_finish(G_SUBPROCESS(src), res, &err)) {
        log_append(app, "[command failed] ");
        log_append(app, err ? err->message : "unknown reason");
        log_append(app, "\n");
        g_clear_error(&err);
    } else {
        log_append(app, "[command finished]\n");
    }
    g_object_unref(src);
    start_version_probe(app);
}

static GSubprocess *spawn_command(App *app, const char *cmdline, GError **err)
{
    char **argv = NULL;
    int n = 0;
    GSubprocess *proc;

    if (!g_shell_parse_argv(cmdline, &n, &argv, err))
        return NULL;
    proc = g_subprocess_newv((const char *const *)argv,
                             G_SUBPROCESS_FLAGS_STDOUT_PIPE |
                             G_SUBPROCESS_FLAGS_STDERR_MERGE, err);
    g_strfreev(argv);
    if (!proc)
        return NULL;
    log_append(app, "$ ");
    log_append(app, cmdline);
    log_append(app, "\n");
    watch_output(app, proc);
    return proc;
}

static void on_serve_start(GtkButton *b, gpointer data)
{
    App *app = data;
    GError *err = NULL;
    GSubprocess *proc;

    (void)b;
    if (app->server_proc) {
        set_status(app, "The server was already started from this window");
        return;
    }
    proc = spawn_command(app, entry_str(app->serve_cmd_e), &err);
    if (!proc) {
        show_error(app, err ? err->message : "Could not start the server");
        g_clear_error(&err);
        return;
    }
    app->server_proc = proc;
    gtk_widget_set_sensitive(app->serve_btn, FALSE);
    gtk_widget_set_sensitive(app->serve_stop_btn, TRUE);
    g_subprocess_wait_check_async(proc, NULL, on_server_exit, app);
    g_timeout_add_seconds(2, delayed_probe, app);
    set_status(app, "Server starting…");
}

static void on_serve_stop(GtkButton *b, gpointer data)
{
    App *app = data;

    (void)b;
    if (!app->server_proc) {
        set_status(app, "No server was started from this window");
        return;
    }
    g_subprocess_send_signal(app->server_proc, SIGTERM);
    log_append(app, "[SIGTERM sent]\n");
}

static void on_signin_clicked(GtkButton *b, gpointer data)
{
    App *app = data;
    GError *err = NULL;
    GSubprocess *proc;

    (void)b;
    proc = spawn_command(app, entry_str(app->signin_cmd_e), &err);
    if (!proc) {
        show_error(app, err ? err->message
                            : "Could not run the sign-in command");
        g_clear_error(&err);
        return;
    }
    log_append(app, "[sign-in started: confirm in the browser window that "
                    "opens, or follow the URL printed above]\n");
    g_subprocess_wait_check_async(proc, NULL, on_oneshot_exit, app);
}

static void on_shutdown(GApplication *gapp, gpointer data)
{
    App *app = data;

    (void)gapp;
    save_state(app);
    if (app->server_proc) {
        g_subprocess_send_signal(app->server_proc, SIGTERM);
        g_object_unref(app->server_proc);
        app->server_proc = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* the three new pages                                                */
/* ------------------------------------------------------------------ */

static GtkWidget *note_label(const char *text)
{
    GtkWidget *l = gtk_label_new(text);

    gtk_label_set_xalign(GTK_LABEL(l), 0.0f);
    gtk_label_set_wrap(GTK_LABEL(l), TRUE);
    gtk_widget_set_margin_bottom(l, 2);
    return l;
}

static GtkWidget *page_box(void)
{
    GtkWidget *b = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);

    gtk_widget_set_margin_start(b, 10);
    gtk_widget_set_margin_end(b, 10);
    gtk_widget_set_margin_top(b, 10);
    gtk_widget_set_margin_bottom(b, 10);
    return b;
}

static GtkWidget *page_scroller(GtkWidget *child)
{
    GtkWidget *sw = gtk_scrolled_window_new();

    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_min_content_width(GTK_SCROLLED_WINDOW(sw),
                                              PANE_LEFT_MIN);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(sw), child);
    return sw;
}

static GtkWidget *build_models_page(App *app)
{
    GtkWidget *p = page_box();
    GtkWidget *row, *scr, *btn;

    gtk_box_append(GTK_BOX(p), section_label("Pull a model"));
    gtk_box_append(GTK_BOX(p), note_label(
        "Name as it appears in the library, for example llama3.2:3b, or a "
        "cloud variant such as gpt-oss:120b-cloud (sign in first)."));

    app->pull_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->pull_entry), "llama3.2:3b");
    gtk_box_append(GTK_BOX(p), app->pull_entry);

    row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    app->pull_btn = gtk_button_new_with_label("Pull");
    gtk_widget_set_hexpand(app->pull_btn, TRUE);
    g_signal_connect(app->pull_btn, "clicked",
                     G_CALLBACK(on_pull_clicked), app);
    btn = gtk_button_new_with_label("Cancel");
    g_signal_connect(btn, "clicked", G_CALLBACK(on_pull_cancel), app);
    gtk_box_append(GTK_BOX(row), app->pull_btn);
    gtk_box_append(GTK_BOX(row), btn);
    gtk_box_append(GTK_BOX(p), row);

    app->pull_bar = gtk_progress_bar_new();
    gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(app->pull_bar), TRUE);
    gtk_box_append(GTK_BOX(p), app->pull_bar);
    app->pull_status = note_label("idle");
    gtk_box_append(GTK_BOX(p), app->pull_status);

    gtk_box_append(GTK_BOX(p), section_label("Installed models"));
    btn = gtk_button_new_with_label("Refresh list");
    g_signal_connect(btn, "clicked", G_CALLBACK(on_refresh_all), app);
    gtk_box_append(GTK_BOX(p), btn);

    scr = ro_text_area(&app->models_buf, FALSE, TRUE, 240);
    gtk_text_buffer_set_text(app->models_buf, "(loading…)", -1);
    gtk_widget_set_vexpand(scr, TRUE);
    gtk_box_append(GTK_BOX(p), scr);
    gtk_box_append(GTK_BOX(p), note_label(
        "Choose the one to use from the Model list on the Request page; that "
        "list is rebuilt after every pull."));

    return page_scroller(p);
}

static GtkWidget *build_server_page(App *app)
{
    GtkWidget *p = page_box();
    GtkWidget *row, *scr, *btn;

    gtk_box_append(GTK_BOX(p), section_label("Local server"));
    app->version_label = note_label("server: unknown");
    gtk_box_append(GTK_BOX(p), app->version_label);

    app->serve_cmd_e = gtk_entry_new();
    gtk_editable_set_text(GTK_EDITABLE(app->serve_cmd_e), app->cfg_serve_cmd);
    gtk_box_append(GTK_BOX(p), prop_row("Start cmd", app->serve_cmd_e));

    row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    app->serve_btn = gtk_button_new_with_label("Start server");
    gtk_widget_set_hexpand(app->serve_btn, TRUE);
    g_signal_connect(app->serve_btn, "clicked", G_CALLBACK(on_serve_start), app);
    app->serve_stop_btn = gtk_button_new_with_label("Stop");
    gtk_widget_set_sensitive(app->serve_stop_btn, FALSE);
    g_signal_connect(app->serve_stop_btn, "clicked",
                     G_CALLBACK(on_serve_stop), app);
    btn = gtk_button_new_with_label("Check");
    g_signal_connect(btn, "clicked", G_CALLBACK(on_probe_clicked), app);
    gtk_box_append(GTK_BOX(row), app->serve_btn);
    gtk_box_append(GTK_BOX(row), app->serve_stop_btn);
    gtk_box_append(GTK_BOX(row), btn);
    gtk_box_append(GTK_BOX(p), row);
    gtk_box_append(GTK_BOX(p), note_label(
        "Stop affects only a server started from this window; a "
        "systemd-managed instance keeps running."));

    gtk_box_append(GTK_BOX(p), section_label("Sign in"));
    gtk_box_append(GTK_BOX(p), note_label(
        "Sign-in registers this machine's key with your ollama.com account, "
        "which cloud models require. The command opens a browser; the URL "
        "also appears in the log below."));
    app->signin_cmd_e = gtk_entry_new();
    gtk_editable_set_text(GTK_EDITABLE(app->signin_cmd_e), app->cfg_signin_cmd);
    gtk_box_append(GTK_BOX(p), prop_row("Command", app->signin_cmd_e));
    btn = gtk_button_new_with_label("Run sign-in");
    g_signal_connect(btn, "clicked", G_CALLBACK(on_signin_clicked), app);
    gtk_box_append(GTK_BOX(p), btn);

    gtk_box_append(GTK_BOX(p), section_label("API key"));
    gtk_box_append(GTK_BOX(p), note_label(
        "Sent as an Authorization: Bearer header. Needed only when talking "
        "straight to https://ollama.com; a local server signs its own cloud "
        "requests."));
    app->apikey_entry = gtk_entry_new();
    gtk_entry_set_visibility(GTK_ENTRY(app->apikey_entry), FALSE);
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->apikey_entry),
                                   "taken from OLLAMA_API_KEY if set");
    if (app->cfg_key && *app->cfg_key)
        gtk_editable_set_text(GTK_EDITABLE(app->apikey_entry), app->cfg_key);
    gtk_box_append(GTK_BOX(p), app->apikey_entry);
    app->remember_key_cb = gtk_check_button_new_with_label(
        "Store the key in the config file (plain text, mode 0600)");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->remember_key_cb),
                                app->cfg_remember_key);
    gtk_box_append(GTK_BOX(p), app->remember_key_cb);

    gtk_box_append(GTK_BOX(p), section_label("Log"));
    scr = ro_text_area(&app->log_buf, FALSE, TRUE, 180);
    gtk_widget_set_vexpand(scr, TRUE);
    gtk_box_append(GTK_BOX(p), scr);

    return page_scroller(p);
}

static GtkWidget *build_usage_page(App *app)
{
    GtkWidget *p = page_box();
    GtkWidget *row, *scr, *btn;

    gtk_box_append(GTK_BOX(p), section_label("Token ledger"));
    gtk_box_append(GTK_BOX(p), note_label(
        "Ollama reports prompt_eval_count and eval_count per request but "
        "exposes no account quota endpoint, so this page is a local tally "
        "priced with the rates you enter. It is an estimate, not your "
        "account balance."));

    app->price_in_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->price_in_e), "0.00");
    gtk_editable_set_text(GTK_EDITABLE(app->price_in_e), app->cfg_price_in);
    gtk_box_append(GTK_BOX(p), prop_row("in / 1M tok", app->price_in_e));

    app->price_out_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->price_out_e), "0.00");
    gtk_editable_set_text(GTK_EDITABLE(app->price_out_e), app->cfg_price_out);
    gtk_box_append(GTK_BOX(p), prop_row("out / 1M tok", app->price_out_e));

    app->budget_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->budget_e), "0.00");
    gtk_editable_set_text(GTK_EDITABLE(app->budget_e), app->cfg_budget);
    gtk_box_append(GTK_BOX(p), prop_row("budget", app->budget_e));

    app->currency_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->currency_e), "USD");
    gtk_editable_set_text(GTK_EDITABLE(app->currency_e), app->cfg_currency);
    gtk_box_append(GTK_BOX(p), prop_row("currency", app->currency_e));

    row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    btn = gtk_button_new_with_label("Apply");
    g_signal_connect(btn, "clicked", G_CALLBACK(on_usage_apply), app);
    gtk_box_append(GTK_BOX(row), btn);
    btn = gtk_button_new_with_label("Reset period");
    g_signal_connect(btn, "clicked", G_CALLBACK(on_usage_reset), app);
    gtk_box_append(GTK_BOX(row), btn);
    btn = gtk_button_new_with_label("Save table…");
    g_signal_connect(btn, "clicked", G_CALLBACK(on_save_usage), app);
    gtk_box_append(GTK_BOX(row), btn);
    gtk_box_append(GTK_BOX(p), row);

    app->usage_summary = note_label("");
    gtk_box_append(GTK_BOX(p), app->usage_summary);

    scr = ro_text_area(&app->usage_buf, FALSE, TRUE, 240);
    gtk_widget_set_vexpand(scr, TRUE);
    gtk_box_append(GTK_BOX(p), scr);

    gtk_box_append(GTK_BOX(p), note_label(
        "Models tagged -cloud or :cloud, and anything sent to ollama.com, "
        "count as billed. Everything else is local and tallied separately."));

    return page_scroller(p);
}

/* Wrap the existing request builder and the three new pages in a stack. */
static GtkWidget *build_left_pane(App *app, GtkWidget *request_page)
{
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    GtkWidget *stack = gtk_stack_new();
    GtkWidget *switcher = gtk_stack_switcher_new();

    gtk_stack_set_transition_type(GTK_STACK(stack),
                                  GTK_STACK_TRANSITION_TYPE_NONE);
    gtk_stack_add_titled(GTK_STACK(stack), request_page, "request", "Request");
    gtk_stack_add_titled(GTK_STACK(stack), build_models_page(app),
                         "models", "Models");
    gtk_stack_add_titled(GTK_STACK(stack), build_server_page(app),
                         "server", "Server");
    gtk_stack_add_titled(GTK_STACK(stack), build_usage_page(app),
                         "usage", "Usage");
    gtk_stack_switcher_set_stack(GTK_STACK_SWITCHER(switcher), GTK_STACK(stack));

    gtk_widget_set_margin_top(switcher, 6);
    gtk_widget_set_vexpand(stack, TRUE);
    gtk_box_append(GTK_BOX(box), switcher);
    gtk_box_append(GTK_BOX(box), stack);
    return box;
}

/* ------------------------------------------------------------------ */
/* UI construction                                                    */
/* ------------------------------------------------------------------ */

static GtkWidget *section_label(const char *text)
{
    GtkWidget *l = gtk_label_new(NULL);
    char *markup = g_strdup_printf("<b>%s</b>", text);

    gtk_label_set_markup(GTK_LABEL(l), markup);
    gtk_label_set_xalign(GTK_LABEL(l), 0.0f);
    g_free(markup);
    return l;
}

static GtkWidget *prop_row(const char *label, GtkWidget *w)
{
    GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *l = gtk_label_new(label);

    gtk_label_set_xalign(GTK_LABEL(l), 0.0f);
    gtk_widget_set_size_request(l, 86, -1);
    gtk_widget_set_hexpand(w, TRUE);
    gtk_box_append(GTK_BOX(row), l);
    gtk_box_append(GTK_BOX(row), w);
    return row;
}

static GtkWidget *make_text_area(GtkTextBuffer **buf_out, gboolean wrap,
                                 gboolean mono, int min_h)
{
    GtkWidget *view = gtk_text_view_new();
    GtkWidget *scr = gtk_scrolled_window_new();

    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(view),
                                wrap ? GTK_WRAP_WORD_CHAR : GTK_WRAP_NONE);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(view), mono);
    gtk_text_view_set_top_margin(GTK_TEXT_VIEW(view), 6);
    gtk_text_view_set_bottom_margin(GTK_TEXT_VIEW(view), 6);
    gtk_text_view_set_left_margin(GTK_TEXT_VIEW(view), 6);
    gtk_text_view_set_right_margin(GTK_TEXT_VIEW(view), 6);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scr), view);
    gtk_scrolled_window_set_min_content_height(GTK_SCROLLED_WINDOW(scr), min_h);
    *buf_out = gtk_text_view_get_buffer(GTK_TEXT_VIEW(view));
    return scr;
}

static gboolean set_initial_split(gpointer data)
{
    App *app = data;

    gtk_paned_set_position(GTK_PANED(app->paned), 390);
    return G_SOURCE_REMOVE;
}

static void activate(GtkApplication *gapp, gpointer user_data)
{
    App *app = g_new0(App, 1);
    GtkWidget *win, *top, *split, *left_sw, *left, *chat_sw, *bottom;
    GtkWidget *host_box, *refresh_btn, *grid, *toggles, *att_row;
    GtkWidget *attach_btn, *clear_att_btn, *scr, *btn_row, *spacer, *st_row;
    GtkWidget *preview_btn, *copy_btn, *save_btn, *new_btn;
    GtkEventController *motion, *keys;
    GtkTextIter e;

    (void)user_data;

    app->models = g_ptr_array_new_with_free_func(g_free);
    app->attachments = g_ptr_array_new_with_free_func(attachment_free);
    app->messages = json_array_new();
    app->usage = g_ptr_array_new_with_free_func(usage_free);
    load_state(app);                    /* features 4-6: config + ledger */

    win = gtk_application_window_new(gapp);
    app->win = GTK_WINDOW(win);
    gtk_window_set_title(GTK_WINDOW(win), "Ollama GUI");
    gtk_window_set_default_size(GTK_WINDOW(win), 1180, 780);

    top = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_window_set_child(GTK_WINDOW(win), top);

    /* -------- left panel (request builder) -------- */

    left = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_widget_set_margin_start(left, 10);
    gtk_widget_set_margin_end(left, 10);
    gtk_widget_set_margin_top(left, 10);
    gtk_widget_set_margin_bottom(left, 10);

    gtk_box_append(GTK_BOX(left), section_label("Connection"));

    app->host_entry = gtk_entry_new();
    gtk_editable_set_text(GTK_EDITABLE(app->host_entry), app->cfg_host);
    gtk_widget_set_hexpand(app->host_entry, TRUE);
    refresh_btn = gtk_button_new_with_label("⟳");
    gtk_widget_set_tooltip_text(refresh_btn, "Load model list from /api/tags");
    g_signal_connect(refresh_btn, "clicked", G_CALLBACK(on_refresh_all), app);
    host_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_append(GTK_BOX(host_box), app->host_entry);
    gtk_box_append(GTK_BOX(host_box), refresh_btn);
    gtk_box_append(GTK_BOX(left), prop_row("Host", host_box));

    /* placeholder until the real list arrives */
    {
        const char *const init[] = { "(press ⟳ to load)", NULL };

        app->model_list = gtk_string_list_new(init);
    }
    app->model_dd = gtk_drop_down_new(G_LIST_MODEL(app->model_list), NULL);
    gtk_box_append(GTK_BOX(left), prop_row("Model", app->model_dd));

    app->endpoint_dd = gtk_drop_down_new_from_strings(ENDPOINTS);
    gtk_box_append(GTK_BOX(left), prop_row("Endpoint", app->endpoint_dd));

    gtk_box_append(GTK_BOX(left), section_label("System prompt"));
    scr = make_text_area(&app->sys_buf, TRUE, FALSE, 90);
    gtk_box_append(GTK_BOX(left), scr);

    gtk_box_append(GTK_BOX(left), section_label("Output"));
    app->format_dd = gtk_drop_down_new_from_strings(OUT_MODES);
    gtk_box_append(GTK_BOX(left), prop_row("Format", app->format_dd));
    gtk_box_append(GTK_BOX(left),
                   section_label("JSON schema (for format = JSON schema)"));
    scr = make_text_area(&app->schema_buf, FALSE, TRUE, 100);
    gtk_box_append(GTK_BOX(left), scr);

    gtk_box_append(GTK_BOX(left), section_label("Generation options"));

    app->temp_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->temp_e), "e.g. 0.8");
    app->topp_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->topp_e), "e.g. 0.9");
    app->ctx_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->ctx_e), "e.g. 4096");
    app->npred_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->npred_e), "e.g. 512 (-1 = ∞)");
    app->seed_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->seed_e), "e.g. 42");
    app->keep_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->keep_e), "e.g. 5m");
    app->limit_e = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(app->limit_e), "20000");

    grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 6);
#define GRID_PAIR(row, col, lab, entry)                                     \
    do {                                                                    \
        GtkWidget *l = gtk_label_new(lab);                                  \
        gtk_label_set_xalign(GTK_LABEL(l), 0.0f);                           \
        gtk_widget_set_hexpand(entry, TRUE);                                \
        gtk_grid_attach(GTK_GRID(grid), l, (col), (row), 1, 1);             \
        gtk_grid_attach(GTK_GRID(grid), entry, (col) + 1, (row), 1, 1);     \
    } while (0)
    GRID_PAIR(0, 0, "temperature", app->temp_e);
    GRID_PAIR(0, 2, "top_p", app->topp_e);
    GRID_PAIR(1, 0, "num_ctx", app->ctx_e);
    GRID_PAIR(1, 2, "num_predict", app->npred_e);
    GRID_PAIR(2, 0, "seed", app->seed_e);
    GRID_PAIR(2, 2, "keep_alive", app->keep_e);
    GRID_PAIR(3, 0, "char limit", app->limit_e);
#undef GRID_PAIR
    gtk_box_append(GTK_BOX(left), grid);

    gtk_box_append(GTK_BOX(left), section_label("Toggles"));

    toggles = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    app->stream_cb = gtk_check_button_new_with_label("Stream responses");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->stream_cb), TRUE);
    app->think_cb = gtk_check_button_new_with_label("Enable thinking");
    app->hist_cb = gtk_check_button_new_with_label("Send chat history");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->hist_cb), TRUE);
    app->ground_cb = gtk_check_button_new_with_label(
        "Also ground schema in the prompt");
    app->ptr_cb = gtk_check_button_new_with_label(
        "Split follows mouse pointer");
    gtk_widget_set_tooltip_text(app->ptr_cb,
        "Move the pointer left/right across the window to steer the "
        "divider between the settings and the chat");
    app->colour_cb = gtk_check_button_new_with_label("Colour");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(app->colour_cb), TRUE);
    gtk_widget_set_tooltip_text(app->colour_cb,
        "Colourful markdown rendering of the answers");
    g_signal_connect(app->colour_cb, "toggled",
                     G_CALLBACK(on_colour_toggled), app);

    gtk_box_append(GTK_BOX(toggles), app->stream_cb);
    gtk_box_append(GTK_BOX(toggles), app->think_cb);
    gtk_box_append(GTK_BOX(toggles), app->hist_cb);
    gtk_box_append(GTK_BOX(toggles), app->ground_cb);
    gtk_box_append(GTK_BOX(toggles), app->ptr_cb);
    gtk_box_append(GTK_BOX(toggles), app->colour_cb);
    gtk_box_append(GTK_BOX(left), toggles);

    gtk_box_append(GTK_BOX(left), section_label("Attachments"));

    app->loadmode_dd = gtk_drop_down_new_from_strings(LOAD_MODES);
    gtk_widget_set_hexpand(app->loadmode_dd, TRUE);
    attach_btn = gtk_button_new_with_label("Attach…");
    g_signal_connect(attach_btn, "clicked", G_CALLBACK(on_attach_files), app);
    clear_att_btn = gtk_button_new_with_label("Clear");
    g_signal_connect(clear_att_btn, "clicked", G_CALLBACK(on_clear_attach), app);
    att_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_append(GTK_BOX(att_row), app->loadmode_dd);
    gtk_box_append(GTK_BOX(att_row), attach_btn);
    gtk_box_append(GTK_BOX(att_row), clear_att_btn);
    gtk_box_append(GTK_BOX(left), att_row);

    scr = make_text_area(&app->attach_buf, FALSE, TRUE, 64);
    {
        /* the view inside `scr` should be read-only */
        GtkWidget *v = gtk_scrolled_window_get_child(GTK_SCROLLED_WINDOW(scr));

        gtk_text_view_set_editable(GTK_TEXT_VIEW(v), FALSE);
    }
    gtk_box_append(GTK_BOX(left), scr);
    refresh_attachment_list(app);

    left_sw = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(left_sw),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_min_content_width(GTK_SCROLLED_WINDOW(left_sw),
                                              PANE_LEFT_MIN);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(left_sw), left);

    /* -------- chat transcript -------- */

    app->chat_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(app->chat_view), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(app->chat_view), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(app->chat_view),
                                GTK_WRAP_WORD_CHAR);
    gtk_text_view_set_top_margin(GTK_TEXT_VIEW(app->chat_view), 8);
    gtk_text_view_set_bottom_margin(GTK_TEXT_VIEW(app->chat_view), 8);
    gtk_text_view_set_left_margin(GTK_TEXT_VIEW(app->chat_view), 8);
    gtk_text_view_set_right_margin(GTK_TEXT_VIEW(app->chat_view), 8);
    gtk_widget_set_vexpand(app->chat_view, TRUE);
    gtk_widget_set_hexpand(app->chat_view, TRUE);
    app->chat_buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(app->chat_view));
    gtk_text_buffer_get_end_iter(app->chat_buf, &e);
    app->end_mark = gtk_text_buffer_create_mark(app->chat_buf, NULL, &e, FALSE);
    gtk_text_buffer_set_text(app->chat_buf,
        "Pick a model on the left, type below, press Send (or Ctrl+Enter).\n",
        -1);

    chat_sw = gtk_scrolled_window_new();
    gtk_widget_set_vexpand(chat_sw, TRUE);
    gtk_widget_set_hexpand(chat_sw, TRUE);
    gtk_scrolled_window_set_min_content_width(GTK_SCROLLED_WINDOW(chat_sw),
                                              PANE_RIGHT_MIN);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(chat_sw), app->chat_view);

    /* -------- paned split + pointer steering -------- */

    split = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    app->paned = split;
    gtk_paned_set_wide_handle(GTK_PANED(split), TRUE);
    gtk_paned_set_resize_start_child(GTK_PANED(split), FALSE);
    gtk_paned_set_resize_end_child(GTK_PANED(split), TRUE);
    gtk_paned_set_shrink_start_child(GTK_PANED(split), FALSE);
    gtk_paned_set_shrink_end_child(GTK_PANED(split), FALSE);
    gtk_paned_set_start_child(GTK_PANED(split), build_left_pane(app, left_sw));
    gtk_paned_set_end_child(GTK_PANED(split), chat_sw);
    gtk_widget_set_vexpand(split, TRUE);
    gtk_widget_set_hexpand(split, TRUE);

    motion = gtk_event_controller_motion_new();
    g_signal_connect(motion, "motion", G_CALLBACK(on_root_motion), app);
    gtk_widget_add_controller(split, motion);

    gtk_box_append(GTK_BOX(top), split);

    /* -------- composer -------- */

    bottom = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_margin_start(bottom, 8);
    gtk_widget_set_margin_end(bottom, 8);
    gtk_widget_set_margin_top(bottom, 6);
    gtk_widget_set_margin_bottom(bottom, 8);

    scr = make_text_area(&app->input_buf, TRUE, FALSE, 80);
    {
        GtkWidget *v = gtk_scrolled_window_get_child(GTK_SCROLLED_WINDOW(scr));

        keys = gtk_event_controller_key_new();
        g_signal_connect(keys, "key-pressed", G_CALLBACK(on_input_key), app);
        gtk_widget_add_controller(v, keys);
    }
    gtk_box_append(GTK_BOX(bottom), scr);

    btn_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);

    app->send_btn = gtk_button_new_with_label("Send");
    gtk_widget_add_css_class(app->send_btn, "suggested-action");
    g_signal_connect(app->send_btn, "clicked", G_CALLBACK(on_send), app);
    app->stop_btn = gtk_button_new_with_label("Stop");
    gtk_widget_add_css_class(app->stop_btn, "destructive-action");
    gtk_widget_set_sensitive(app->stop_btn, FALSE);
    g_signal_connect(app->stop_btn, "clicked", G_CALLBACK(on_stop), app);
    preview_btn = gtk_button_new_with_label("Preview request");
    g_signal_connect(preview_btn, "clicked", G_CALLBACK(on_preview), app);

    spacer = gtk_label_new("");
    gtk_widget_set_hexpand(spacer, TRUE);

    /* buttons */
    app->code_btn = gtk_button_new_with_label("Extract code…");
    gtk_widget_set_tooltip_text(app->code_btn,
        "Save every fenced code/script block of the last reply to files");
    g_signal_connect(app->code_btn, "clicked", G_CALLBACK(on_extract_code), app);
    app->points_btn = gtk_button_new_with_label("Extract points…");
    gtk_widget_set_tooltip_text(app->points_btn,
        "Save the headings and bullet points of the last reply to a markdown file");
    g_signal_connect(app->points_btn, "clicked",
                     G_CALLBACK(on_extract_points), app);

    copy_btn = gtk_button_new_with_label("Copy reply");
    g_signal_connect(copy_btn, "clicked", G_CALLBACK(on_copy_reply), app);
    save_btn = gtk_button_new_with_label("Save chat…");
    g_signal_connect(save_btn, "clicked", G_CALLBACK(on_save_chat), app);
    new_btn = gtk_button_new_with_label("New chat");
    g_signal_connect(new_btn, "clicked", G_CALLBACK(on_new_chat), app);

    gtk_box_append(GTK_BOX(btn_row), app->send_btn);
    gtk_box_append(GTK_BOX(btn_row), app->stop_btn);
    gtk_box_append(GTK_BOX(btn_row), preview_btn);
    gtk_box_append(GTK_BOX(btn_row), spacer);
    gtk_box_append(GTK_BOX(btn_row), app->code_btn);
    gtk_box_append(GTK_BOX(btn_row), app->points_btn);
    gtk_box_append(GTK_BOX(btn_row), copy_btn);
    gtk_box_append(GTK_BOX(btn_row), save_btn);
    gtk_box_append(GTK_BOX(btn_row), new_btn);
    gtk_box_append(GTK_BOX(bottom), btn_row);

    st_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    app->spinner = gtk_spinner_new();
    app->status = gtk_label_new("Ready");
    gtk_label_set_xalign(GTK_LABEL(app->status), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(app->status), PANGO_ELLIPSIZE_END);
    gtk_widget_set_hexpand(app->status, TRUE);
    gtk_box_append(GTK_BOX(st_row), app->spinner);
    gtk_box_append(GTK_BOX(st_row), app->status);
    gtk_box_append(GTK_BOX(bottom), st_row);

    gtk_box_append(GTK_BOX(top), bottom);

    gtk_window_set_default_widget(GTK_WINDOW(win), app->send_btn);
    gtk_window_present(GTK_WINDOW(win));

    /* initial divider position, once the window has a real width */
    g_idle_add(set_initial_split, app);

    /* fetch the model list right away */
    start_models_load(app);

    /* features 4-6: server state, inventory, ledger, and save on exit */
    start_version_probe(app);
    start_inventory_load(app);
    render_usage(app);
    g_signal_connect(gapp, "shutdown", G_CALLBACK(on_shutdown), app);
}

int main(int argc, char **argv)
{
    GtkApplication *gapp;
    int status;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    gapp = gtk_application_new("com.example.OllamaGui",
                               G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(gapp, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(gapp), argc, argv);
    g_object_unref(gapp);
    curl_global_cleanup();
    return status;
}
