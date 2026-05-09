/**
 * vibe-plugin-cursor Provider Tests
 *
 * Tests for the CursorProvider class exported via the vibePlugin.
 */
import { beforeEach, describe, expect, it, mock } from "bun:test";

// Mock @cursor/sdk before importing the plugin.
// The plugin treats the SDK structurally: it expects an `Agent.create({...})`
// returning an instance with `send()` -> Run, where Run exposes
// `stream()` (async iterable of SDKMessage), `wait()`, `cancel()`, and
// `close()` on the agent itself.
mock.module("@cursor/sdk", () => {
  const buildAssistantMessage = (text: string) => ({
    type: "assistant",
    agent_id: "mock-agent",
    run_id: "mock-run",
    message: {
      role: "assistant",
      content: [{ type: "text", text }],
    },
  });

  class MockRun {
    readonly id = "mock-run";
    readonly agentId = "mock-agent";
    private text: string;
    constructor(text: string) {
      this.text = text;
    }
    async *stream(): AsyncGenerator<unknown> {
      yield buildAssistantMessage(this.text);
    }
    async wait(): Promise<{ status: string }> {
      return { status: "finished" };
    }
    async cancel(): Promise<void> {
      /* noop */
    }
  }

  class MockAgentInstance {
    readonly agentId = "mock-agent";
    private text: string;
    constructor(text: string) {
      this.text = text;
    }
    async send(_message: string): Promise<MockRun> {
      return new MockRun(this.text);
    }
    close(): void {
      /* noop */
    }
  }

  const Agent = {
    create: mock(async (_options: Record<string, unknown>) => {
      return new MockAgentInstance("Hello from Cursor");
    }),
  };

  const Cursor = {
    me: mock(async () => ({ id: "mock-user", email: "test@example.com" })),
    models: {
      list: mock(async () => [{ id: "auto", displayName: "Cursor Auto" }]),
    },
  };

  return { Agent, Cursor };
});

const { createPlugin } = await import("../index.js");
const vibePlugin = createPlugin({ name: "test", dataDir: "/tmp" });

// Extract the provider from the plugin
const provider = vibePlugin.providers!.ai!;

describe("CursorProvider", () => {
  const sessionConfig = {
    name: "test-session",
    agentType: "cursor",
    model: "auto",
    maxTokens: 4096,
  };

  beforeEach(() => {
    // Ensure SDK mode is used
    process.env["CURSOR_API_KEY"] = "test-key-123";
    provider.setMode!("sdk");
  });

  // ── Session Lifecycle ───────────────────────────────────────────

  describe("createSession", () => {
    it("creates a new session with generated ID", async () => {
      const session = await provider.createSession(sessionConfig);

      expect(session.id).toBeDefined();
      expect(session.name).toBe("test-session");
      expect(session.agentType).toBe("cursor");
      expect(session.provider).toBe("cursor");
      expect(session.status).toBe("active");
      expect(session.stats.inputTokens).toBe(0);
      expect(session.stats.outputTokens).toBe(0);
      expect(session.stats.requestCount).toBe(0);
      expect(session.createdAt).toBeDefined();
    });

    it("uses provided sessionId from providerConfig", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "custom-id-001" },
      });

      expect(session.id).toBe("custom-id-001");
    });

    it("returns existing session if ID already exists", async () => {
      const session1 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });
      const session2 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });

      expect(session1.id).toBe(session2.id);
      expect(session2.status).toBe("active");
    });
  });

  describe("configureSession", () => {
    it("updates session config", async () => {
      const session = await provider.createSession({ ...sessionConfig });
      await provider.configureSession(session.id, { model: "gpt-5" });

      // Verify by listing sessions
      const sessions = await provider.listSessions();
      const updated = sessions.find((s) => s.id === session.id);
      expect(updated?.config.model).toBe("gpt-5");
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.configureSession("does-not-exist", { model: "x" }),
      ).rejects.toThrow("not found");
    });
  });

  describe("destroySession", () => {
    it("terminates session and cleans up", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "destroy-me" },
      });

      await provider.destroySession(session.id);

      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("terminated");
    });

    it("no-ops for unknown session ID", async () => {
      // Should not throw
      await provider.destroySession("nonexistent-session");
    });
  });

  describe("listSessions", () => {
    it("returns all sessions", async () => {
      const id = `list-test-${Date.now()}`;
      await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: id },
      });

      const sessions = await provider.listSessions();
      expect(sessions.length).toBeGreaterThanOrEqual(1);
      expect(sessions.some((s) => s.id === id)).toBe(true);
    });
  });

  describe("getSessionStatus", () => {
    it("returns status for existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `status-${Date.now()}` },
      });

      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("active");
    });

    it("returns terminated for unknown session", async () => {
      const status = await provider.getSessionStatus("totally-unknown");
      expect(status).toBe("terminated");
    });
  });

  // ── sendPrompt ──────────────────────────────────────────────────

  describe("sendPrompt", () => {
    it("sends prompt via SDK adapter and returns response", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `prompt-${Date.now()}` },
      });

      const response = await provider.sendPrompt(session.id, "What is 2+2?");

      expect(response.content).toBe("Hello from Cursor");
      expect(response.model).toBe("auto");
      expect(response.inputTokens).toBeGreaterThan(0);
      expect(response.outputTokens).toBeGreaterThan(0);
      expect(response.durationMs).toBeGreaterThanOrEqual(0);
    });

    it("accumulates usage stats across multiple prompts", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `multi-prompt-${Date.now()}` },
      });

      await provider.sendPrompt(session.id, "First prompt");
      await provider.sendPrompt(session.id, "Second prompt");

      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBeGreaterThan(0);
      expect(stats.outputTokens).toBeGreaterThan(0);
      expect(stats.requestCount).toBe(2);
    });

    it("throws for non-existent session", async () => {
      await expect(provider.sendPrompt("ghost", "Hello")).rejects.toThrow(
        "not found",
      );
    });

    it("throws for terminated session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `terminated-prompt-${Date.now()}` },
      });
      await provider.destroySession(session.id);

      await expect(provider.sendPrompt(session.id, "Hello")).rejects.toThrow(
        "terminated",
      );
    });
  });

  // ── getUsageStats ───────────────────────────────────────────────

  describe("getUsageStats", () => {
    it("returns zero stats for fresh session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `fresh-stats-${Date.now()}` },
      });

      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBe(0);
      expect(stats.outputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
      expect(stats.estimatedCostUsd).toBe(0);
    });

    it("returns default stats for unknown session", async () => {
      const stats = await provider.getUsageStats("no-such-session");
      expect(stats.inputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
    });
  });

  // ── healthCheck ─────────────────────────────────────────────────

  describe("healthCheck", () => {
    it("returns ok when SDK is available", async () => {
      const result = await provider.healthCheck();
      expect(result.ok).toBe(true);
      expect(result.message).toContain("SDK");
    });
  });

  // ── getCapabilities ─────────────────────────────────────────────

  describe("getCapabilities", () => {
    it("returns correct capabilities for SDK mode", () => {
      provider.setMode!("sdk");
      const caps = provider.getCapabilities!();

      expect(caps.streaming).toBe(true);
      expect(caps.vision).toBe(true);
      expect(caps.fileAttachments).toBe(true);
      expect(caps.toolUse).toBe(true);
      expect(caps.mcpSupport).toBe(true);
      expect(caps.cancelSupport).toBe(true);
      expect(caps.modelListing).toBe(true);
    });

    it("returns correct capabilities for CLI mode", () => {
      provider.setMode!("cli");
      const caps = provider.getCapabilities!();

      expect(caps.streaming).toBe(false);
      expect(caps.vision).toBe(false);
      expect(caps.cancelSupport).toBe(false);

      // Restore SDK mode
      provider.setMode!("sdk");
    });
  });

  // ── getMode / setMode ───────────────────────────────────────────

  describe("getMode / setMode", () => {
    it("defaults to sdk when CURSOR_API_KEY is set", () => {
      process.env["CURSOR_API_KEY"] = "key";
      const mode = provider.getMode!();
      expect(mode).toBe("sdk");
    });

    it("allows explicit mode switching", () => {
      provider.setMode!("cli");
      expect(provider.getMode!()).toBe("cli");

      provider.setMode!("sdk");
      expect(provider.getMode!()).toBe("sdk");
    });
  });

  // ── listModels ──────────────────────────────────────────────────

  describe("listModels", () => {
    it("returns available Cursor models", async () => {
      const models = await provider.listModels!();

      expect(models.length).toBeGreaterThanOrEqual(2);
      expect(models.every((m) => m.provider === "cursor")).toBe(true);

      const auto = models.find((m) => m.id === "auto");
      expect(auto).toBeDefined();
      expect(auto!.supportsStreaming).toBe(true);
      expect(auto!.contextWindow).toBe(200_000);
    });

    it("returns a copy (not the internal array)", async () => {
      const models1 = await provider.listModels!();
      const models2 = await provider.listModels!();
      expect(models1).not.toBe(models2);
      expect(models1).toEqual(models2);
    });
  });

  // ── cancelRequest ───────────────────────────────────────────────

  describe("cancelRequest", () => {
    it("throws for unknown session", async () => {
      await expect(provider.cancelRequest!("missing")).rejects.toThrow(
        "not found",
      );
    });
  });

  // ── attachFiles ─────────────────────────────────────────────────

  describe("attachFiles", () => {
    it("attaches files to an existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `files-${Date.now()}` },
      });

      await provider.attachFiles!(session.id, [
        {
          filename: "test.txt",
          mimeType: "text/plain",
          content: "hello",
          size: 5,
        },
      ]);

      // No error means success
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.attachFiles!("none", [
          { filename: "f.txt", mimeType: "text/plain", content: "x", size: 1 },
        ]),
      ).rejects.toThrow("not found");
    });
  });
});
