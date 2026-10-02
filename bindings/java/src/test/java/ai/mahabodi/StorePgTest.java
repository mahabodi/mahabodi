package ai.mahabodi;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * PostgreSQL store through the JNI binding (native library built with the {@code postgres} feature).
 * Enabled only when MAHABODI_TEST_PG_DSN is set; aborted (skipped) at store_open when this build has
 * no store, mirroring bindings/python/tests/test_store_pg.py.
 */
class StorePgTest {
    @Test
    @EnabledIfEnvironmentVariable(named = "MAHABODI_TEST_PG_DSN", matches = ".+")
    void storeRoundTrip() {
        String dsn = System.getenv("MAHABODI_TEST_PG_DSN");
        try (Bodi b = new Bodi()) {
            // Unique namespace per run. The engine exposes no store_drop method, so the namespace's
            // rows are left behind (the Rust test cleans up with direct SQL, which the binding cannot do).
            String ns = "tjava_" + ProcessHandle.current().pid();
            try {
                b.call("store_open", "{\"dsn\":" + Bodi.Json.str(dsn) + ",\"namespace\":" + Bodi.Json.str(ns) + ",\"vector_type\":\"vector\",\"create\":true}");
            } catch (BodiException e) { // a library built without the postgres feature
                assumeTrue(false, "store not available in this build: " + e.getMessage());
            }
            b.call("store_ingest_batch", "{\"docs\":["
                + "{\"text\":\"Travel over $500 must be approved by a manager.\",\"source\":\"policy1\"},"
                + "{\"text\":\"Expense reports are due within 30 days.\",\"source\":\"policy2\"},"
                + "{\"text\":\"Refunds above $5,000 need the finance director.\",\"source\":\"policy3\"}]}");
            b.call("store_build_index", "{}");
            String q = b.call("store_query", "{\"q\":\"who approves travel\",\"k\":10,\"mode\":\"hybrid\"}");
            assertTrue(q.contains("\"stage\":\""), q);
            assertFalse(q.contains("\"hits\":[]"), q);
            // the top hit must be the travel doc: a store Hit is a flat object, so the first '}' after
            // "hits":[ closes it
            int h = q.indexOf("\"hits\":[");
            String top = q.substring(h, q.indexOf('}', h) + 1);
            assertTrue(top.contains("approved by a manager"), q);
            assertTrue(q.contains("\"handoff\":false"), q);
            // No hub fallback in the store: a query nothing matches is an empty handoff.
            String miss = b.call("store_query", "{\"q\":\"zzqx unrelated gibberish\",\"k\":5}");
            assertTrue(miss.contains("\"handoff\":true") || miss.contains("\"stage\":\"hub\"") || miss.contains("\"stage\":\"empty_memory\""), miss);
        }
    }
}
