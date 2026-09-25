package ai.mahabodi;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Paths;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

class BodiTest {
    static String fixture(String n) throws Exception {
        return Bodi.readFile(Paths.get("..", "..", "crates", "mahabodi-core", "tests", "fixtures", n + ".md"));
    }

    @Test
    void ingestQueryDensity() throws Exception {
        try (Bodi b = new Bodi()) {
            String r = b.ingest(fixture("robotics"), "auto", "robotics");
            assertTrue(r.contains("\"passes\":true"), r);
            String q = b.query("spacecraft", 3);
            assertTrue(q.contains("\"matched\":true") && q.contains("\"stage\":\"exact\""), q);
            String miss = b.query("zzqxv wkkpj", 5);
            assertTrue(miss.contains("\"matched\":false") && miss.contains("\"handoff\":true"), miss);
            assertTrue(b.context("zzqxv wkkpj", 5, 2000).contains("\"context\":\"\""));
        }
    }

    @Test
    void decideDefaults() {
        try (Bodi b = new Bodi()) {
            String d = b.decideDefaults();
            assertTrue(d.contains("\"experience_override_agree\":6") && d.contains("\"experience_memory_first_margin\":0.2") && d.contains("\"oos_min_similarity\":null"), d);
        }
    }

    @Test
    void errorsAndClose() {
        Bodi b = new Bodi("{\"auto_density\":true}");
        assertThrows(BodiException.class, () -> b.call("nope", "{}"));
        assertThrows(BodiException.class, () -> b.call("query", "{not json"));
        assertThrows(BodiException.class, () -> b.decide("\"x\"", "{\"q\":{\"type\":\"noul\",\"instructions\":\"?\"}}"));
        b.close();
        b.close();
        assertThrows(BodiException.class, () -> b.query("x", 1));
        assertThrows(BodiException.class, () -> new Bodi("{\"auto_density\": 5}"));
    }

    @Test
    void snapshotRestoreAndThreads() throws Exception {
        try (Bodi b = new Bodi(); Bodi b2 = new Bodi()) {
            b.ingest(fixture("world_events"), "auto", "we");
            b2.restore(b.snapshot());
            assertEquals(b.query("elevator", 3), b2.query("elevator", 3));
            ExecutorService ex = Executors.newFixedThreadPool(8);
            java.util.List<Future<String>> fs = new java.util.ArrayList<>();
            for (int i = 0; i < 64; i++) {
                final int n = i;
                fs.add(ex.submit(() -> n % 8 == 0 ? b.ingest("Thread note " + n + " on space elevators.", "text", "t") : b.query("elevator", 3)));
            }
            for (Future<String> f : fs) assertNotNull(f.get());
            ex.shutdown();
        }
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "BODI_LAYA_DIR", matches = ".+")
    void layaDecision() {
        try (Bodi b = new Bodi()) {
            b.loadLaya(System.getenv("BODI_LAYA_DIR"));
            String q = "{\"topic\":{\"type\":\"choice\",\"instructions\":\"What is the topic of `article`?\",\"criteria\":{\"world\":\"world news\",\"sports\":\"sports\",\"business\":\"business\",\"sci_tech\":\"science and technology\"}}}";
            String r = b.decide("{\"article\":\"The Lakers beat the Celtics 110-102 in overtime on Sunday.\"}", q);
            assertTrue(r.contains("\"choice\":\"sports\""), r);
        }
    }
}
