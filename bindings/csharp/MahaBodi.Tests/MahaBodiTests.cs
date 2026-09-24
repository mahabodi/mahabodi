using System;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using MahaBodi;
using Xunit;

public class BodiTests
{
    static string Fixture(string n) => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "crates", "mahabodi-core", "tests", "fixtures", n + ".md"));

    [Fact]
    public void IngestQueryDensity()
    {
        using var b = new Bodi();
        var r = b.Ingest(Fixture("robotics"), source: "robotics")!;
        Assert.True(r["density"]!["passes"]!.GetValue<bool>());
        var q = b.Query("spacecraft", 3)!;
        Assert.True(q["matched"]!.GetValue<bool>());
        Assert.Equal("exact", q["stage"]!.GetValue<string>());
        Assert.All(q["hits"]!.AsArray(), h => Assert.StartsWith("F_", h!["id"]!.GetValue<string>()));
        var miss = b.Query("zzqxv wkkpj")!;
        Assert.False(miss["matched"]!.GetValue<bool>());
        Assert.True(miss["handoff"]!.GetValue<bool>());
        Assert.Equal("", b.Context("zzqxv wkkpj")!["context"]!.GetValue<string>());
        Assert.False(string.IsNullOrEmpty(Bodi.Version));
    }

    [Fact]
    public async Task ErrorsDisposeSnapshotThreads()
    {
        Assert.Throws<BodiException>(() => new Bodi("{\"auto_density\": 5}"));
        var b = new Bodi("{\"auto_density\": true}");
        Assert.Throws<BodiException>(() => b.Call("nope"));
        await Assert.ThrowsAsync<BodiException>(() => b.DecideAsync(JsonValue.Create("x")!, new JsonObject { ["q"] = new JsonObject { ["type"] = "noul", ["instructions"] = "?" } }));
        b.Ingest(Fixture("world_events"), source: "we");
        using var b2 = new Bodi();
        b2.Restore(b.Snapshot()!);
        Assert.Equal(b.Query("elevator", 3)!.ToJsonString(), b2.Query("elevator", 3)!.ToJsonString());
        var tasks = Enumerable.Range(0, 64).Select(i => Task.Run(() => i % 8 == 0 ? b.Ingest($"Thread note {i} on space elevators.", "text") : b.Query("elevator", 3))).ToArray();
        await Task.WhenAll(tasks);
        b.Dispose();
        b.Dispose();
        Assert.Throws<ObjectDisposedException>(() => b.Query("x"));
    }

    [SkippableFact]
    public async Task LayaDecision()
    {
        var dir = Environment.GetEnvironmentVariable("BODI_LAYA_DIR");
        // reported as Skipped (not Passed) when no model is available
        Skip.If(string.IsNullOrEmpty(dir), "set BODI_LAYA_DIR to run the Laya model test");
        using var b = new Bodi();
        b.LoadLaya(dir);
        var q = JsonNode.Parse("{\"topic\":{\"type\":\"choice\",\"instructions\":\"What is the topic of `article`?\",\"criteria\":{\"world\":\"world news\",\"sports\":\"sports\",\"business\":\"business\",\"sci_tech\":\"science and technology\"}}}")!.AsObject();
        var r = await b.DecideAsync(JsonNode.Parse("{\"article\":\"The Lakers beat the Celtics 110-102 in overtime on Sunday.\"}")!, q);
        Assert.Equal("sports", r!["answers"]!["topic"]!["choice"]!.GetValue<string>());
    }
}
