using System;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

namespace MahaBodi;

/// <summary>Raised for any Bodi error (invalid input, missing model, unknown method, ...).</summary>
public sealed class BodiException : Exception
{
    public BodiException(string message) : base(message) { }
}

internal static class Native
{
    private const string Lib = "mahabodi";

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr bodi_new(byte[]? configJson, out IntPtr err);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr bodi_call(IntPtr handle, byte[] method, byte[] argsJson);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void bodi_free(IntPtr handle);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void bodi_string_free(IntPtr s);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr bodi_version();

    internal static byte[] Utf8Z(string s)
    {
        var n = System.Text.Encoding.UTF8.GetByteCount(s);
        var b = new byte[n + 1];
        System.Text.Encoding.UTF8.GetBytes(s, 0, s.Length, b, 0);
        return b;
    }

    internal static string TakeString(IntPtr p)
    {
        try { return Marshal.PtrToStringUTF8(p) ?? ""; }
        finally { bodi_string_free(p); }
    }
}

/// <summary>
/// Bodi engine. Thread-safe: one instance can be shared across threads. Dispose to free
/// native memory. Every method returns parsed JSON (<see cref="JsonNode"/>).
/// </summary>
public sealed class Bodi : IDisposable
{
    private IntPtr _h;
    private readonly object _gate = new();

    public Bodi(string? configJson = null)
    {
        _h = Native.bodi_new(configJson is null ? null : Native.Utf8Z(configJson), out var err);
        if (_h == IntPtr.Zero)
            throw new BodiException(err == IntPtr.Zero ? "bodi_new failed" : Native.TakeString(err));
    }

    public static string Version => Marshal.PtrToStringUTF8(Native.bodi_version()) ?? "";

    /// <summary>Call any Bodi method with JSON args; returns the parsed result.</summary>
    public JsonNode? Call(string method, JsonObject? args = null)
    {
        IntPtr h;
        lock (_gate) h = _h;
        if (h == IntPtr.Zero) throw new ObjectDisposedException(nameof(Bodi));
        var raw = Native.TakeString(Native.bodi_call(h, Native.Utf8Z(method), Native.Utf8Z((args ?? new JsonObject()).ToJsonString())));
        var env = JsonNode.Parse(raw)!.AsObject();
        if (env.TryGetPropertyValue("error", out var e)) throw new BodiException(e!.GetValue<string>());
        return env["ok"];
    }

    public Task<JsonNode?> CallAsync(string method, JsonObject? args = null) => Task.Run(() => Call(method, args));

    public JsonNode? Ingest(string text, string format = "auto", string source = "doc") =>
        Call("ingest", new JsonObject { ["text"] = text, ["format"] = format, ["source"] = source });
    public JsonNode? IngestBatch(JsonArray docs) => Call("ingest_batch", new JsonObject { ["docs"] = docs.DeepClone() });
    public JsonNode? Query(string q, int k = 5) => Call("query", new JsonObject { ["q"] = q, ["k"] = k });
    public JsonNode? Traverse(string start, int hops = 2, int limit = 50) =>
        Call("traverse", new JsonObject { ["start"] = start, ["hops"] = hops, ["limit"] = limit });
    public JsonNode? Context(string q, int k = 5, int maxChars = 2000) =>
        Call("context", new JsonObject { ["q"] = q, ["k"] = k, ["max_chars"] = maxChars });
    public JsonNode? Density() => Call("density");
    public JsonNode? EnsureDensity() => Call("ensure_density");
    public JsonNode? Snapshot() => Call("snapshot");
    public JsonNode? Restore(JsonNode snapshot) => Call("restore", new JsonObject { ["snapshot"] = snapshot.DeepClone() });
    public JsonNode? Stats() => Call("stats");

    public void LoadLaya(string dir) => Call("load_laya", new JsonObject { ["dir"] = dir });
    public void LoadEmbedder(string dir) => Call("load_embedder", new JsonObject { ["dir"] = dir });
    /// <summary>Experience memory: labels[i] maps question id to the gold label of states[i].</summary>
    /// <param name="calibrate">Run Laya on up to N labelled cases; where memory is clearly better, decisions come from memory (0 = off).</param>
    public Task<JsonNode?> LearnAsync(JsonArray states, JsonObject questions, JsonArray labels, int calibrate = 0) =>
        CallAsync("learn", new JsonObject { ["states"] = states.DeepClone(), ["questions"] = questions.DeepClone(), ["labels"] = labels.DeepClone(), ["calibrate"] = calibrate });
    /// <summary>The engine's effective decide() defaults.</summary>
    public JsonNode? DecideDefaults() => Call("decide_defaults");
    public void Forget() => Call("forget");
    public Task<JsonNode?> DecideAsync(JsonNode state, JsonObject questions, JsonObject? options = null) =>
        CallAsync("decide", new JsonObject { ["state"] = state.DeepClone(), ["questions"] = questions.DeepClone(), ["options"] = options?.DeepClone() });
    public Task<JsonNode?> PredictAsync(JsonNode state, JsonObject questions) =>
        CallAsync("predict", new JsonObject { ["state"] = state.DeepClone(), ["questions"] = questions.DeepClone() });

    public void Dispose()
    {
        IntPtr h;
        lock (_gate) { h = _h; _h = IntPtr.Zero; }
        if (h != IntPtr.Zero) Native.bodi_free(h);
    }
}
