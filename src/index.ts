/**
 * vibe-plugin-cursor
 *
 * Cursor AI agent provider for VibeControls Agent.
 * Implements the AIAgentProvider interface with dual-mode support:
 * - SDK mode: Uses @cursor/sdk (TypeScript SDK for Cursor agents) for direct API access
 * - CLI mode: Uses the `cursor-agent` CLI binary
 *
 * Mode auto-detection: SDK if CURSOR_API_KEY is set, CLI if `cursor-agent`
 * binary is found, error if neither is available.
 */

import { Elysia } from "elysia";

// ── Locally Redeclared Interfaces ────────────────────────────────────────
// (Avoid hard dependency on @vibecontrols/agent)

type ProviderMode = "sdk" | "cli";

interface AIModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsStreaming: boolean;
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

interface AIProviderCapabilities {
  streaming: boolean;
  vision: boolean;
  fileAttachments: boolean;
  toolUse: boolean;
  mcpSupport: boolean;
  voiceMode: boolean;
  cancelSupport: boolean;
  modelListing: boolean;
}

interface AIFileAttachment {
  filename: string;
  mimeType: string;
  content: Buffer | string;
  size: number;
}

interface VibePlugin {
  name: string;
  version: string;
  description?: string;
  tags?: Array<
    "backend" | "frontend" | "cli" | "provider" | "adapter" | "integration"
  >;
  cliCommand?: string;
  apiPrefix?: string;
  prerequisites?: Array<{
    name: string;
    kind: "binary" | "npm" | "pip" | "cargo" | "manual";
    requiresSudo: boolean;
    description?: string;
  }>;
  createRoutes?: () => unknown;
  providers?: { ai?: AIAgentProvider; [key: string]: unknown };
  onServerStart?: (
    app: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
  onServerStop?: () => void | Promise<void>;
  onCliSetup?: (
    program: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
}

interface HostServices {
  logger?: {
    info: (source: string, msg: string) => void;
    warn: (source: string, msg: string) => void;
    error: (source: string, msg: string) => void;
    debug: (source: string, msg: string) => void;
  };
  serviceRegistry?: {
    getService: <T>(pluginName: string, serviceName: string) => T | undefined;
  };
  getConfig: (key: string) => string | undefined | Promise<string | undefined>;
}

type AISessionStatus =
  | "active"
  | "idle"
  | "processing"
  | "error"
  | "terminated";
type AILogType =
  | "input"
  | "output"
  | "thinking"
  | "event"
  | "error"
  | "metadata";

interface AISessionConfig {
  name: string;
  agentType: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  workingDirectory?: string;
  providerConfig?: Record<string, unknown>;
}

interface AISession {
  id: string;
  name: string;
  status: AISessionStatus;
  agentType: string;
  provider: string;
  config: AISessionConfig;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

interface AIContext {
  id: string;
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface AIResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  thinkingSteps?: string[];
  durationMs: number;
  metadata?: Record<string, unknown>;
}

interface AIStreamChunk {
  type: "text" | "thinking" | "error" | "done";
  content: string;
  tokensUsed?: number;
}

interface AILog {
  id: string;
  sessionId: string;
  type: AILogType;
  content: string;
  tokenCount?: number;
  model?: string;
  durationMs?: number;
  agentMetadata?: Record<string, unknown>;
  createdAt: string;
}

interface AILogFilter {
  types?: AILogType[];
  startDate?: string;
  endDate?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

interface AIUsageStats {
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  estimatedCostUsd: number;
  modelBreakdown?: Record<
    string,
    { inputTokens: number; outputTokens: number; requestCount: number }
  >;
}

interface AIAgentProvider {
  readonly name: string;
  createSession(config: AISessionConfig): Promise<AISession>;
  sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse>;
  streamPrompt?(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse>;
  getSessionLogs(sessionId: string, filter?: AILogFilter): Promise<AILog[]>;
  getUsageStats(sessionId: string): Promise<AIUsageStats>;
  configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void>;
  destroySession(sessionId: string): Promise<void>;
  listSessions(): Promise<AISession[]>;
  getSessionStatus(sessionId: string): Promise<AISessionStatus>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancelRequest?(sessionId: string): Promise<void>;
  getCapabilities?(): AIProviderCapabilities;
  attachFiles?(sessionId: string, files: AIFileAttachment[]): Promise<void>;
  getMode?(): ProviderMode;
  setMode?(mode: ProviderMode): void;
  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null;
  sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }>;
}

// Log ingester interface (from ai plugin's service registry)
interface LogIngester {
  append(input: {
    sessionId: string;
    type: AILogType;
    content: string;
    tokenCount?: number;
    model?: string;
    durationMs?: number;
    agentMetadata?: Record<string, unknown>;
  }): unknown;
}

// ── Provider Adapter Interface ──────────────────────────────────────────

interface ProviderAdapter {
  readonly mode: ProviderMode;

  sendPrompt(
    prompt: string,
    config: AISessionConfig,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  healthCheck(): Promise<{ ok: boolean; message?: string }>;
}

// ── Constants ───────────────────────────────────────────────────────────

const PROVIDER_NAME = "cursor";
const CLI_COMMAND = "cursor-agent";
const DISPLAY_NAME = "Cursor";

/**
 * Resolve the cursor-agent binary path with the platform-correct extension.
 * On Windows `Bun.spawn` calls CreateProcess directly which doesn't honour
 * PATHEXT, so a bare `cursor-agent` won't find `cursor-agent.exe`/`.cmd`.
 * `Bun.which` searches PATH the same way the shell does.
 */
function resolveCursorAgentCmd(): string {
  const found =
    typeof Bun !== "undefined" && typeof Bun.which === "function"
      ? Bun.which(CLI_COMMAND)
      : null;
  if (found) return found;
  return process.platform === "win32" ? `${CLI_COMMAND}.exe` : CLI_COMMAND;
}
// Cursor's well-known model IDs. The SDK accepts an optional ModelSelection
// (id + variant params); when omitted the agent picks the user's default.
const DEFAULT_MODEL = "auto";
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["sdk", "cli"];
// Per https://cursor.com/docs/cli/overview the canonical install is:
//   curl https://cursor.com/install -fsS | bash
// We invoke it via `bash -lc` so the pipeline executes correctly.
const CLI_INSTALL_COMMAND = [
  "bash",
  "-lc",
  "curl https://cursor.com/install -fsS | bash",
];

// Conservative default model catalog. Cursor's actual list is account-scoped
// and reachable via Cursor.models.list() at runtime; this static list is what
// listModels() returns when the SDK is not yet authenticated.
const CURSOR_MODELS: AIModelInfo[] = [
  {
    id: "auto",
    name: "Cursor Auto",
    provider: PROVIDER_NAME,
    contextWindow: 200_000,
    maxOutputTokens: 16_384,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 0,
    outputPricePerMToken: 0,
  },
  {
    id: "claude-sonnet-4",
    name: "Claude Sonnet 4 (via Cursor)",
    provider: PROVIDER_NAME,
    contextWindow: 200_000,
    maxOutputTokens: 16_384,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 3.0,
    outputPricePerMToken: 15.0,
  },
  {
    id: "gpt-5",
    name: "GPT-5 (via Cursor)",
    provider: PROVIDER_NAME,
    contextWindow: 200_000,
    maxOutputTokens: 16_384,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 0,
    outputPricePerMToken: 0,
  },
];

// ── SDK Adapter ─────────────────────────────────────────────────────────

/**
 * Minimal structural types for @cursor/sdk so we don't need to import the
 * package at module load time (kept consistent with how the claude plugin
 * structurally types @anthropic-ai/sdk).
 */
interface CursorSdkRunResult {
  status?: string;
  errorCode?: string;
}

interface CursorSdkRun {
  readonly id: string;
  readonly agentId: string;
  stream(): AsyncIterable<unknown>;
  wait(): Promise<CursorSdkRunResult>;
  cancel(): Promise<void>;
}

interface CursorSdkAgentInstance {
  readonly agentId: string;
  send(
    message: string,
    options?: Record<string, unknown>,
  ): Promise<CursorSdkRun>;
  close(): void;
}

interface CursorSdkModule {
  Agent: {
    create(options: Record<string, unknown>): Promise<CursorSdkAgentInstance>;
  };
  Cursor: {
    me?(options?: Record<string, unknown>): Promise<unknown>;
    models?: {
      list(options?: Record<string, unknown>): Promise<unknown[]>;
    };
  };
}

type CursorAuthResolver = () => Promise<string | undefined>;

class CursorSdkAdapter implements ProviderAdapter {
  readonly mode: ProviderMode = "sdk";
  private sdk: CursorSdkModule | null = null;
  private resolveAuth: CursorAuthResolver;

  constructor(resolveAuth: CursorAuthResolver) {
    this.resolveAuth = resolveAuth;
  }

  private async getSdk(): Promise<{
    sdk: CursorSdkModule;
    apiKey: string;
  }> {
    const apiKey = await this.resolveAuth();
    if (!apiKey) {
      throw new Error(
        "Cursor SDK auth is not configured. Set CURSOR_API_KEY in the " +
          "environment, or store it in agent config (e.g. POST /api/config " +
          "{ key: 'CURSOR_API_KEY', value: '...' }).",
      );
    }

    if (this.sdk) return { sdk: this.sdk, apiKey };

    let mod: unknown;
    try {
      mod = await import("@cursor/sdk");
    } catch {
      throw new Error(
        "Failed to load @cursor/sdk. Install it with: bun add @cursor/sdk",
      );
    }

    this.sdk = mod as CursorSdkModule;
    return { sdk: this.sdk, apiKey };
  }

  private extractText(message: unknown): string {
    // SDKAssistantMessage shape:
    //   { type: "assistant", message: { role: "assistant", content: [{ type: "text", text }, ...] } }
    if (!message || typeof message !== "object") return "";
    const m = message as { type?: string; message?: unknown; text?: string };
    if (m.type === "assistant" && m.message && typeof m.message === "object") {
      const inner = m.message as {
        content?: Array<{ type?: string; text?: string }>;
      };
      const blocks = Array.isArray(inner.content) ? inner.content : [];
      return blocks
        .filter((b) => b.type === "text")
        .map((b) => b.text ?? "")
        .join("");
    }
    return "";
  }

  async sendPrompt(
    prompt: string,
    config: AISessionConfig,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const { sdk, apiKey } = await this.getSdk();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const agentOptions: Record<string, unknown> = {
      apiKey,
      ...(model && model !== "auto" ? { model: { id: model } } : {}),
      local: {
        cwd: config.workingDirectory || process.cwd(),
      },
    };

    const agent = await sdk.Agent.create(agentOptions);
    let cancelled = false;
    const onAbort = (): void => {
      cancelled = true;
    };
    signal?.addEventListener("abort", onAbort);

    let content = "";
    try {
      const fullPrompt = config.systemPrompt
        ? `${config.systemPrompt}\n\n${prompt}`
        : prompt;
      const run = await agent.send(fullPrompt);
      for await (const message of run.stream()) {
        if (cancelled) {
          await run.cancel().catch(() => {});
          break;
        }
        content += this.extractText(message);
      }
      await run.wait().catch(() => {});
    } finally {
      signal?.removeEventListener("abort", onAbort);
      agent.close();
    }

    const durationMs = Date.now() - startTime;
    // Cursor SDK does not surface raw token counts at run-result level today;
    // approximate from character lengths so usage tracking still progresses.
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);

    return {
      content,
      model,
      inputTokens,
      outputTokens,
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const { sdk, apiKey } = await this.getSdk();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const agentOptions: Record<string, unknown> = {
      apiKey,
      ...(model && model !== "auto" ? { model: { id: model } } : {}),
      local: {
        cwd: config.workingDirectory || process.cwd(),
      },
    };

    const agent = await sdk.Agent.create(agentOptions);
    let cancelled = false;
    const onAbort = (): void => {
      cancelled = true;
    };
    signal?.addEventListener("abort", onAbort);

    let content = "";
    try {
      const fullPrompt = config.systemPrompt
        ? `${config.systemPrompt}\n\n${prompt}`
        : prompt;
      const run = await agent.send(fullPrompt);
      try {
        for await (const message of run.stream()) {
          if (cancelled) {
            await run.cancel().catch(() => {});
            break;
          }
          const text = this.extractText(message);
          if (text) {
            content += text;
            onChunk({ type: "text", content: text });
          }
        }
        await run.wait().catch(() => {});
      } catch (err) {
        onChunk({
          type: "error",
          content: err instanceof Error ? err.message : String(err),
        });
        throw err;
      }
    } finally {
      signal?.removeEventListener("abort", onAbort);
      agent.close();
    }

    onChunk({ type: "done", content: "" });

    const durationMs = Date.now() - startTime;
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);

    return {
      content,
      model,
      inputTokens,
      outputTokens,
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      await this.getSdk();
      return {
        ok: true,
        message: `${DISPLAY_NAME} SDK ready (API key configured)`,
      };
    } catch (err) {
      return {
        ok: false,
        message:
          err instanceof Error ? err.message : "SDK initialization failed",
      };
    }
  }
}

// ── CLI Adapter ─────────────────────────────────────────────────────────

class CursorCliAdapter implements ProviderAdapter {
  readonly mode: ProviderMode = "cli";

  private buildCliArgs(config: AISessionConfig, prompt: string): string[] {
    const args: string[] = [];
    if (config.model) args.push("--model", config.model);
    // cursor-agent uses `-p`/`--print` for non-interactive prompts.
    args.push("--print", prompt);
    return args;
  }

  async sendPrompt(
    prompt: string,
    config: AISessionConfig,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const startTime = Date.now();
    const finalPrompt = config.systemPrompt
      ? `${config.systemPrompt}\n\n${prompt}`
      : prompt;
    const args = this.buildCliArgs(config, finalPrompt);

    const proc = Bun.spawn([resolveCursorAgentCmd(), ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      timeout: (config.providerConfig?.["timeoutMs"] as number) || 300_000,
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;
    const durationMs = Date.now() - startTime;

    if (exitCode !== 0 && !stdout) {
      throw new Error(
        `${DISPLAY_NAME} CLI exited with code ${exitCode}: ${stderr}`,
      );
    }

    const content = stdout.trim() || stderr.trim();
    // CLI does not provide real token counts; approximate from character lengths
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);
    const model = config.model || DEFAULT_MODEL;

    return {
      content,
      model,
      inputTokens,
      outputTokens,
      durationMs,
      metadata: { exitCode, provider: PROVIDER_NAME, mode: "cli" },
    };
  }

  async streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    // CLI does not support true streaming; run full prompt then emit chunks
    const result = await this.sendPrompt(prompt, config);
    onChunk({ type: "text", content: result.content });
    onChunk({ type: "done", content: "" });
    return result;
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const proc = Bun.spawnSync([resolveCursorAgentCmd(), "--version"], {
        timeout: 5000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          message: `${DISPLAY_NAME} CLI ${proc.stdout.toString().trim()}`,
        };
      }
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not available (exit code ${proc.exitCode})`,
      };
    } catch {
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not installed or not in PATH`,
      };
    }
  }
}

// ── Provider Implementation ─────────────────────────────────────────────

interface ManagedSession {
  id: string;
  config: AISessionConfig;
  status: AISessionStatus;
  stats: AIUsageStats;
  abortController: AbortController | null;
  files: AIFileAttachment[];
  createdAt: string;
  updatedAt: string;
}

class CursorProvider implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private activeMode: ProviderMode | null = null;
  private adapter: ProviderAdapter | null = null;
  private cachedApiKey: string | undefined;

  setHostServices(hs: HostServices): void {
    this.hostServices = hs;
    this.logIngester =
      hs.serviceRegistry?.getService<LogIngester>("ai", "log-ingester") ?? null;

    // Warm the cache so detectMode() can see DB-stored credentials.
    void Promise.resolve(hs.getConfig("CURSOR_API_KEY"))
      .then((apiKey) => {
        const trimmed = apiKey?.trim();
        if (trimmed) this.cachedApiKey = trimmed;
      })
      .catch(() => {});
  }

  getSupportedModes(): ProviderMode[] {
    return [...SUPPORTED_MODES];
  }

  getDisplayName(): string {
    return DISPLAY_NAME;
  }

  getPrereqApiPrefix(): string {
    return API_PREFIX;
  }

  private async resolveAuth(): Promise<string | undefined> {
    const envKey = process.env["CURSOR_API_KEY"]?.trim();
    if (envKey) return envKey;

    if (this.cachedApiKey) return this.cachedApiKey;

    if (this.hostServices) {
      try {
        const apiKey = (
          await this.hostServices.getConfig("CURSOR_API_KEY")
        )?.trim();
        if (apiKey) {
          this.cachedApiKey = apiKey;
          return apiKey;
        }
      } catch {
        return undefined;
      }
    }
    return undefined;
  }

  // ── Mode Management ──────────────────────────────────────────────────

  getMode(): ProviderMode {
    if (this.activeMode) return this.activeMode;
    return this.detectMode();
  }

  setMode(mode: ProviderMode): void {
    if (!SUPPORTED_MODES.includes(mode)) {
      throw new Error(`${DISPLAY_NAME} does not support ${mode} mode`);
    }
    this.activeMode = mode;
    this.adapter = null; // Force re-creation on next use
    this.log("info", `Mode explicitly set to: ${mode}`);
  }

  private detectMode(): ProviderMode {
    if (process.env["CURSOR_API_KEY"]?.trim() || this.cachedApiKey) {
      return "sdk";
    }

    try {
      // Cross-platform: `which` on POSIX, `where.exe` on Windows.
      const finder = process.platform === "win32" ? "where.exe" : "which";
      const proc = Bun.spawnSync([finder, CLI_COMMAND], {
        timeout: 3000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) return "cli";
    } catch {
      // CLI not found
    }

    // Default to SDK mode; healthCheck will report the actual failure
    return "sdk";
  }

  private getAdapter(): ProviderAdapter {
    if (this.adapter) return this.adapter;

    const mode = this.getMode();
    this.adapter =
      mode === "sdk"
        ? new CursorSdkAdapter(() => this.resolveAuth())
        : new CursorCliAdapter();
    this.activeMode = mode;
    this.log("info", `Adapter initialized in ${mode} mode`);
    return this.adapter;
  }

  // ── Session Management ───────────────────────────────────────────────

  async createSession(config: AISessionConfig): Promise<AISession> {
    const id =
      (config.providerConfig?.["sessionId"] as string) || crypto.randomUUID();
    const now = new Date().toISOString();

    // If session already exists in memory, return it
    const existing = this.sessions.get(id);
    if (existing) {
      existing.status = "active";
      existing.updatedAt = now;
      return this.toAISession(existing);
    }

    const session: ManagedSession = {
      id,
      config,
      status: "active",
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      },
      abortController: null,
      files: [],
      createdAt: now,
      updatedAt: now,
    };

    this.sessions.set(id, session);
    this.log("info", `Session created: ${id} (${config.name})`);

    return this.toAISession(session);
  }

  async sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const result = await adapter.sendPrompt(
        fullPrompt,
        session.config,
        abortController.signal,
      );

      this.updateSessionStats(session, result.inputTokens, result.outputTokens);

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  async streamPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const chunkHandler = onChunk ?? ((_c: AIStreamChunk) => {});

      const result = await adapter.streamPrompt(
        fullPrompt,
        session.config,
        chunkHandler,
        abortController.signal,
      );

      this.updateSessionStats(session, result.inputTokens, result.outputTokens);

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  // ── Extended Methods ─────────────────────────────────────────────────

  async listModels(): Promise<AIModelInfo[]> {
    return [...CURSOR_MODELS];
  }

  async cancelRequest(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);

    if (session.abortController) {
      session.abortController.abort();
      session.abortController = null;
      session.status = "active";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Request cancelled for session: ${sessionId}`);
    }
  }

  getCapabilities(): AIProviderCapabilities {
    const mode = this.getMode();
    return {
      streaming: mode === "sdk",
      vision: mode === "sdk",
      fileAttachments: true,
      toolUse: mode === "sdk",
      mcpSupport: mode === "sdk",
      voiceMode: false,
      cancelSupport: mode === "sdk",
      modelListing: true,
    };
  }

  async attachFiles(
    sessionId: string,
    files: AIFileAttachment[],
  ): Promise<void> {
    const session = this.getSession(sessionId);
    session.files.push(...files);
    session.updatedAt = new Date().toISOString();
    this.log(
      "debug",
      `Attached ${files.length} file(s) to session ${sessionId}`,
    );
  }

  // ── Standard Methods ─────────────────────────────────────────────────

  async getSessionLogs(
    _sessionId: string,
    _filter?: AILogFilter,
  ): Promise<AILog[]> {
    return [];
  }

  async getUsageStats(sessionId: string): Promise<AIUsageStats> {
    const session = this.sessions.get(sessionId);
    return (
      session?.stats ?? {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      }
    );
  }

  async configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void> {
    const session = this.getSession(sessionId);
    Object.assign(session.config, config);
    session.updatedAt = new Date().toISOString();
  }

  async destroySession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      if (session.abortController) {
        session.abortController.abort();
        session.abortController = null;
      }
      session.status = "terminated";
      session.files = [];
      session.updatedAt = new Date().toISOString();
      this.log("info", `Session terminated: ${sessionId}`);
    }
  }

  async listSessions(): Promise<AISession[]> {
    return Array.from(this.sessions.values()).map((s) => this.toAISession(s));
  }

  async getSessionStatus(sessionId: string): Promise<AISessionStatus> {
    return this.sessions.get(sessionId)?.status ?? "terminated";
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    const adapter = this.getAdapter();
    return adapter.healthCheck();
  }

  // ── `vibe ai run` / `vibe ai sdk` integration ────────────────────────

  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null {
    const env: Record<string, string> = {};
    const apiKey =
      process.env["CURSOR_API_KEY"]?.trim() || this.cachedApiKey;
    if (apiKey) env["CURSOR_API_KEY"] = apiKey;
    return { binary: CLI_COMMAND, env };
  }

  async sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }> {
    const adapter = new CursorSdkAdapter(() => this.resolveAuth());
    const config: AISessionConfig = {
      name: "vibe-ai-sdk",
      agentType: PROVIDER_NAME,
      model: opts.model ?? DEFAULT_MODEL,
      maxTokens: opts.maxTokens,
      providerConfig: opts.extras,
    };
    const result = await adapter.sendPrompt(opts.prompt, config);
    return {
      text: result.content,
      usage: {
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      },
    };
  }

  // ── Private Helpers ──────────────────────────────────────────────────

  private getSession(sessionId: string): ManagedSession {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");
    return session;
  }

  private buildFullPrompt(
    prompt: string,
    context?: AIContext[],
    files?: AIFileAttachment[],
  ): string {
    let fullPrompt = prompt;

    if (context && context.length > 0) {
      const contextStr = context
        .map((c) => `--- Context (${c.type}): ---\n${c.content}`)
        .join("\n\n");
      fullPrompt = `${prompt}\n\n${contextStr}`;
    }

    if (files && files.length > 0) {
      const fileStr = files
        .map((f) => {
          const textContent =
            typeof f.content === "string"
              ? f.content
              : f.content.toString("utf-8");
          return `--- File: ${f.filename} (${f.mimeType}, ${f.size} bytes) ---\n${textContent}`;
        })
        .join("\n\n");
      fullPrompt = `${fullPrompt}\n\n${fileStr}`;
    }

    return fullPrompt;
  }

  private updateSessionStats(
    session: ManagedSession,
    inputTokens: number,
    outputTokens: number,
  ): void {
    const model = session.config.model || DEFAULT_MODEL;
    const modelInfo = CURSOR_MODELS.find((m) => m.id === model);

    session.stats.inputTokens += inputTokens;
    session.stats.outputTokens += outputTokens;
    session.stats.requestCount += 1;

    if (modelInfo) {
      const cost =
        (inputTokens / 1_000_000) * modelInfo.inputPricePerMToken +
        (outputTokens / 1_000_000) * modelInfo.outputPricePerMToken;
      session.stats.estimatedCostUsd += cost;
    }

    if (!session.stats.modelBreakdown) {
      session.stats.modelBreakdown = {};
    }
    const breakdown = session.stats.modelBreakdown[model] ?? {
      inputTokens: 0,
      outputTokens: 0,
      requestCount: 0,
    };
    breakdown.inputTokens += inputTokens;
    breakdown.outputTokens += outputTokens;
    breakdown.requestCount += 1;
    session.stats.modelBreakdown[model] = breakdown;

    session.status = "active";
    session.updatedAt = new Date().toISOString();
  }

  private toAISession(s: ManagedSession): AISession {
    return {
      id: s.id,
      name: s.config.name,
      status: s.status,
      agentType: s.config.agentType,
      provider: PROVIDER_NAME,
      config: s.config,
      stats: s.stats,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    };
  }

  private log(level: "info" | "error" | "debug", msg: string): void {
    this.hostServices?.logger?.[level]?.(`${PROVIDER_NAME}-provider`, msg);
  }
}

// ── Plugin Export ────────────────────────────────────────────────────────

function getCliVersion(): string | null {
  try {
    const proc = Bun.spawnSync([resolveCursorAgentCmd(), "--version"], {
      timeout: 5000,
      stdout: "pipe",
      stderr: "ignore",
    });
    if (proc.exitCode === 0) return proc.stdout.toString().trim();
  } catch {
    // Binary not found.
  }
  return null;
}

function createPrereqsRoutes() {
  return new Elysia({ prefix: "/prereqs" })
    .get("/status", () => {
      const version = getCliVersion();
      return {
        satisfied: Boolean(version),
        missing: version
          ? []
          : [
              {
                name: CLI_COMMAND,
                kind: "binary" as const,
                requiresSudo: false,
                description: `${DISPLAY_NAME} CLI for CLI mode`,
              },
            ],
      };
    })
    .post("/install", () => {
      if (getCliVersion()) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }

      const proc = Bun.spawnSync(CLI_INSTALL_COMMAND, {
        timeout: 180_000,
        stdout: "pipe",
        stderr: "pipe",
      });
      if (proc.exitCode === 0 && getCliVersion()) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }
      return {
        ok: false,
        installed: [],
        pendingSudo: [],
        errors: [
          {
            name: CLI_COMMAND,
            message:
              proc.stderr.toString().trim() ||
              `Run manually: ${CLI_INSTALL_COMMAND.join(" ")}`,
          },
        ],
      };
    });
}

const provider = new CursorProvider();

export const vibePlugin: VibePlugin = {
  name: "cursor",
  version: "1.0.0",
  description:
    "Cursor AI agent provider for VibeControls (dual-mode: SDK + CLI)",
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [
    {
      name: CLI_COMMAND,
      kind: "binary",
      requiresSudo: false,
      description: `${DISPLAY_NAME} CLI for CLI mode`,
    },
  ],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),

  onServerStart(_app, hostServices) {
    if (hostServices) provider.setHostServices(hostServices);
  },

  onServerStop() {
    for (const [id] of (provider as CursorProvider)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
  },
};

export default vibePlugin;
