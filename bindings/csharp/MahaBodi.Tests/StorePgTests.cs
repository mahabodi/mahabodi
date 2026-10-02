using System;
using System.Text.Json.Nodes;
using MahaBodi;
using Xunit;

// PostgreSQL store through the .NET binding (native library built with the `postgres` feature).
// Skipped unless MAHABODI_TEST_PG_DSN is set; skipped at store_open when this build has no store,
// mirroring bindings/python/tests/test_store_pg.py.
public class StorePgTests
{
    [SkippableFact]
    public void StoreRoundTrip()
    {
        var dsn = Environment.GetEnvironmentVariable("MAHABODI_TEST_PG_DSN");
        Skip.If(string.IsNullOrEmpty(dsn), "set MAHABODI_TEST_PG_DSN to run the PostgreSQL store test");
        using var b = new Bodi();
        // Unique namespace per run. The engine exposes no store_drop method, so the namespace's rows
        // are left behind (the Rust test cleans up with direct SQL, which the binding cannot do).
        var ns = $"tcs_{Environment.ProcessId}";
        try
        {
            b.Call("store_open", new JsonObject { ["dsn"] = dsn, ["namespace"] = ns, ["vector_type"] = "vector", ["create"] = true });
        }
        catch (BodiException e) // a library built without the postgres feature
        {
            Skip.If(true, $"store not available in this build: {e.Message}");
        }
        var docs = new JsonArray
        {
            new JsonObject { ["text"] = "Travel over $500 must be approved by a manager.", ["source"] = "policy1" },
            new JsonObject { ["text"] = "Expense reports are due within 30 days.", ["source"] = "policy2" },
            new JsonObject { ["text"] = "Refunds above $5,000 need the finance director.", ["source"] = "policy3" },
        };
        b.Call("store_ingest_batch", new JsonObject { ["docs"] = docs });
        b.Call("store_build_index");
        var q = b.Call("store_query", new JsonObject { ["q"] = "who approves travel", ["k"] = 10, ["mode"] = "hybrid" })!;
        Assert.False(string.IsNullOrEmpty(q["stage"]!.GetValue<string>()), q.ToJsonString());
        var hits = q["hits"]!.AsArray();
        Assert.NotEmpty(hits);
        Assert.Contains("approved by a manager", hits[0]!["text"]!.GetValue<string>());
        Assert.False(q["handoff"]!.GetValue<bool>(), q.ToJsonString());
        // No hub fallback in the store: a query nothing matches is an empty handoff.
        var miss = b.Call("store_query", new JsonObject { ["q"] = "zzqx unrelated gibberish", ["k"] = 5 })!;
        var stage = miss["stage"]!.GetValue<string>();
        Assert.True(miss["handoff"]!.GetValue<bool>() || stage == "hub" || stage == "empty_memory", miss.ToJsonString());
    }
}
