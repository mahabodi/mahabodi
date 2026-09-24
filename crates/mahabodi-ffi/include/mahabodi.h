/* Bodi C ABI. All strings are UTF-8 JSON. Strings returned by Bodi must be released with
 * bodi_string_free. A handle is thread-safe: bodi_call may be used concurrently.
 * Result envelope of bodi_call: {"ok": <value>} or {"error": "<message>"}. */
#ifndef MAHABODI_H
#define MAHABODI_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct BodiHandle BodiHandle;

/* config_json may be NULL/"" for defaults. On failure returns NULL and sets *err (if err != NULL). */
BodiHandle *bodi_new(const char *config_json, char **err);
/* Never returns NULL. */
char *bodi_call(const BodiHandle *h, const char *method, const char *args_json);
void bodi_free(BodiHandle *h);
void bodi_string_free(char *s);
const char *bodi_version(void);

#ifdef __cplusplus
}
#endif
#endif
