package ai.mahabodi;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Bodi for the JVM. All arguments and results are JSON text, so the binding has no JSON
 * library dependency; use any JSON library you like on top. Thread-safe: one instance may
 * be shared by many threads. Close it (try-with-resources) to free native memory.
 *
 * <pre>
 * try (Bodi b = new Bodi()) {
 *     b.ingest("Refunds take five days.", "text", "kb");
 *     String result = b.query("refunds", 5);       // JSON: {"matched":true,"stage":"exact",...}
 * }
 * </pre>
 *
 * The native library is loaded from the system property {@code mahabodi.library.path} (a file),
 * else from {@code java.library.path} as {@code mahabodi_jni}.
 */
public final class Bodi implements AutoCloseable {
    static {
        String p = System.getProperty("mahabodi.library.path");
        if (p != null && Files.exists(Paths.get(p))) System.load(Paths.get(p).toAbsolutePath().toString());
        else System.loadLibrary("mahabodi_jni");
    }

    private static native long nativeNew(String configJson);
    private static native String nativeCall(long handle, String method, String argsJson);
    private static native void nativeFree(long handle);

    private volatile long handle;

    public Bodi() { this(""); }

    public Bodi(String configJson) { handle = nativeNew(configJson == null ? "" : configJson); }

    /** Call any Bodi method with JSON args; returns JSON. */
    public String call(String method, String argsJson) {
        long h = handle;
        if (h == 0) throw new BodiException("engine is closed");
        return nativeCall(h, method, argsJson == null ? "{}" : argsJson);
    }

    public String ingest(String text, String format, String source) {
        return call("ingest", "{\"text\":" + Json.str(text) + ",\"format\":" + Json.str(format) + ",\"source\":" + Json.str(source) + "}");
    }
    /** Bulk ingest; docsJson: [{"text": ..., "format"?: ..., "source"?: ...}, ...] (one rebuild). */
    public String ingestBatch(String docsJson) { return call("ingest_batch", "{\"docs\":" + docsJson + "}"); }
    public String query(String q, int k) { return call("query", "{\"q\":" + Json.str(q) + ",\"k\":" + k + "}"); }
    public String traverse(String start, int hops, int limit) {
        return call("traverse", "{\"start\":" + Json.str(start) + ",\"hops\":" + hops + ",\"limit\":" + limit + "}");
    }
    public String context(String q, int k, int maxChars) {
        return call("context", "{\"q\":" + Json.str(q) + ",\"k\":" + k + ",\"max_chars\":" + maxChars + "}");
    }
    public String density() { return call("density", "{}"); }
    public String ensureDensity() { return call("ensure_density", "{}"); }
    public String snapshot() { return call("snapshot", "{}"); }
    public String restore(String snapshotJson) { return call("restore", "{\"snapshot\":" + snapshotJson + "}"); }
    public String stats() { return call("stats", "{}"); }

    public void loadLaya(String dir) { call("load_laya", "{\"dir\":" + Json.str(dir) + "}"); }
    public void loadEmbedder(String dir) { call("load_embedder", "{\"dir\":" + Json.str(dir) + "}"); }
    /** Experience memory. statesJson: JSON array; labelsJson: JSON array of {questionId: gold}. */
    public String learn(String statesJson, String questionsJson, String labelsJson) {
        return call("learn", "{\"states\":" + statesJson + ",\"questions\":" + questionsJson + ",\"labels\":" + labelsJson + "}");
    }
    public void forget() { call("forget", "{}"); }
    /** stateJson: a JSON string/object/array; questionsJson: {"id": {"type": ..., "instructions": ..., ...}}. */
    public String decide(String stateJson, String questionsJson) {
        return call("decide", "{\"state\":" + stateJson + ",\"questions\":" + questionsJson + "}");
    }
    public String predict(String stateJson, String questionsJson) {
        return call("predict", "{\"state\":" + stateJson + ",\"questions\":" + questionsJson + "}");
    }
    public String decideWithMemory(String stateJson, String questionsJson, String query) {
        return call("decide_with_memory", "{\"state\":" + stateJson + ",\"questions\":" + questionsJson + ",\"query\":" + Json.str(query) + "}");
    }

    @Override
    public synchronized void close() {
        long h = handle;
        handle = 0;
        if (h != 0) nativeFree(h);
    }

    /** Minimal JSON string quoting helper. */
    public static final class Json {
        private Json() {}
        public static String str(String s) {
            if (s == null) return "null";
            StringBuilder b = new StringBuilder(s.length() + 2).append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"': b.append("\\\""); break;
                    case '\\': b.append("\\\\"); break;
                    case '\n': b.append("\\n"); break;
                    case '\r': b.append("\\r"); break;
                    case '\t': b.append("\\t"); break;
                    default:
                        if (c < 0x20) b.append(String.format("\\u%04x", (int) c)); else b.append(c);
                }
            }
            return b.append('"').toString();
        }
    }

    static String readFile(Path p) throws java.io.IOException { return new String(Files.readAllBytes(p), java.nio.charset.StandardCharsets.UTF_8); }
}
