package nvidia

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"nvidia-playgroud-go/internal/models"
	"nvidia-playgroud-go/internal/reasoning"
)

// --- Client ---

// Client is the NVIDIA Build Playground predict client (multi-model).
type Client struct {
	captchaToken string // reverse-engineered captcha token

	// HTTP client (can be overridden)
	httpClient *http.Client

	// default request parameters
	model     string
	maxTokens int
	seed      int
	temp      float64
	topP      float64

	// thinking is the explicit on/off request; the actual wire knob is
	// model-specific (see WithThinking).
	thinking *bool

	// reasoningEffort is an explicit level (low/medium/high/max, or none).
	// It takes precedence over thinking and is mapped to the model's knob.
	reasoningEffort string

	// infoOverride bypasses registry lookup in buildRequest (custom
	// namespace/slug/function-id targets). Nil means use the registry.
	infoOverride *models.ModelInfo
}

// Option configures the Client.
type Option func(*Client)

// WithCaptchaToken sets the hCaptcha token for the reverse-engineered endpoint.
// Obtain this from the Playground page: widget's data-hcaptcha-response attribute.
func WithCaptchaToken(token string) Option {
	return func(c *Client) { c.captchaToken = token }
}

// WithHTTPClient overrides the default HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.httpClient = hc }
}

// WithModel overrides the model name.
func WithModel(m string) Option {
	return func(c *Client) { c.model = m }
}

// WithModelInfo pins the NVCF invocation target (namespace, slug,
// function id) directly, bypassing the model registry. Use it to call a
// predict endpoint the registry does not know yet. The request body's
// model field is the canonical model id, mapped to the wire spelling.
func WithModelInfo(info models.ModelInfo) Option {
	return func(c *Client) { c.infoOverride = &info }
}

// WithDefaults sets default request parameters.
func WithDefaults(maxTokens int, seed int, temp, topP float64) Option {
	return func(c *Client) {
		c.maxTokens = maxTokens
		c.seed = seed
		c.temp = temp
		c.topP = topP
	}
}

// WithThinking forces thinking mode (reasoning_content) on or off. The knob
// is model-specific: reasoning_effort for deepseek/nemotron, enable_thinking
// for toggle models, thinking_mode for minimax. Without it, thinking kwargs
// are injected only for models that declare the Thinking capability (most of
// the current anonymous catalog rejects thinking kwargs with 400).
func WithThinking(enable bool) Option {
	return func(c *Client) { c.thinking = &enable }
}

// WithReasoningEffort requests an explicit reasoning level (low, medium,
// high, max, or none where the model supports it). It takes precedence over
// WithThinking and is mapped to the knob the model accepts: a top-level
// reasoning_effort for kimi, chat_template_kwargs for the other models.
func WithReasoningEffort(level string) Option {
	return func(c *Client) { c.reasoningEffort = level }
}

// New creates a client configured with an hCaptcha token (default model
// moonshotai/kimi-k3_1-pro; override with WithModel).
func New(opts ...Option) *Client {
	c := &Client{
		// No overall timeout: http.Client.Timeout also covers reading the
		// response body, which would truncate a long SSE stream (a model that
		// reasons for minutes). Non-streaming calls get requestTimeout from
		// doRequest; streams are bounded by the caller's context only.
		httpClient: &http.Client{},
		model:      DefaultModel,
		maxTokens:  defaultMaxTokens,
		seed:       defaultSeed,
		temp:       defaultTemp,
		topP:       defaultTopP,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// --- internal request helpers ---

func (c *Client) buildRequest(ctx context.Context, req *ChatRequest) (*http.Request, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	info, canonical, err := c.modelInfo(model)
	if err != nil {
		return nil, fmt.Errorf("nvidia: %w", err)
	}
	// A "<function-id>@<model>" pin selects the NVCF function; the body keeps
	// the plain registry model id NVIDIA expects to see.
	if req.Model == model && model != canonical {
		req.Model = canonical
	}
	// The wire model string can differ from the registry key: kimi's backend
	// expects "moonshotai/kimi-k3.1-pro" while the registry id uses an
	// underscore. Marshal a copy so the caller's request keeps the registry id.
	bodyReq := *req
	bodyReq.Model = models.RequestModel(canonical)
	body, err := json.Marshal(&bodyReq)
	if err != nil {
		return nil, fmt.Errorf("nvidia: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", info.PredictEndpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("nvidia: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("accept", "text/event-stream")
	httpReq.Header.Set("nv-function-id", info.FunctionID)
	httpReq.Header.Set("nv-captcha-token", c.captchaToken)
	httpReq.Header.Set("Origin", "https://build.nvidia.com")
	httpReq.Header.Set("Referer", "https://build.nvidia.com/")

	return httpReq, nil
}

func (c *Client) applyDefaults(r *ChatRequest) error {
	if r.Model == "" {
		r.Model = c.model
	}
	if r.MaxTokens == 0 {
		r.MaxTokens = c.maxTokens
	}
	if r.Seed == 0 {
		r.Seed = c.seed
	}
	if r.Temperature == 0 {
		r.Temperature = c.temp
	}
	if r.TopP == 0 {
		r.TopP = c.topP
	}
	level := c.reasoningEffort
	if level == "" {
		level = r.ReasoningEffort
	}
	if level != "" {
		if err := reasoning.ValidateEffort(level); err != nil {
			return fmt.Errorf("nvidia: %w", err)
		}
	}
	if len(r.ChatTemplateKwargs) == 0 {
		enable := true
		if c.thinking != nil {
			enable = *c.thinking
		}
		if c.thinking != nil || level != "" {
			// Explicit control: use the knob this model actually accepts.
			// kimi-k3_1-pro takes reasoning_effort at the request root.
			if effort, ok := reasoning.TopLevelEffort(r.Model, level, enable); ok {
				r.ReasoningEffort = effort
				return nil
			}
			if level != "" {
				if kw, ok := reasoning.EffortKwargs(r.Model, level); ok {
					r.ChatTemplateKwargs = kw
					r.ReasoningEffort = "" // consumed into the kwargs
					return nil
				}
			}
			if c.thinking != nil {
				r.ChatTemplateKwargs = reasoning.ThinkingKwargs(r.Model, enable)
				return nil
			}
		}
		// Only inject thinking kwargs for models known to accept them
		// (Capability.Thinking). Most of the current anonymous catalog
		// (e.g. kimi-k3_1-pro) rejects chat_template_kwargs with 400.
		info, _, err := c.modelInfo(r.Model)
		if err != nil {
			return fmt.Errorf("nvidia: %w", err)
		}
		if info.Capability == nil || !info.Capability.Thinking {
			return nil
		}
		r.ChatTemplateKwargs = reasoning.ThinkingKwargs(r.Model, true)
	}
	return nil
}

// modelInfo resolves the NVCF invocation data for a request and the canonical
// model id to send upstream: the explicit WithModelInfo override when set,
// otherwise the registry. A "<function-id>@<model>" pin overrides the registry
// function id, which is how a caller reaches a specific NVCF instance behind a
// playground slug.
func (c *Client) modelInfo(model string) (models.ModelInfo, string, error) {
	pin, base, hasPin := models.SplitFunctionRef(model)
	if hasPin {
		if err := models.ValidateFunctionRef(pin, base); err != nil {
			return models.ModelInfo{}, "", err
		}
		model = base
	}
	if c.infoOverride != nil {
		return *c.infoOverride, model, nil
	}
	info, err := models.Lookup(model)
	if err != nil {
		return models.ModelInfo{}, "", err
	}
	if hasPin {
		info.FunctionID = pin
	}
	return info, model, nil
}

// --- Public API ---

// Chat sends a synchronous chat completion request.
func (c *Client) Chat(ctx context.Context, messages []Message) (*ChatResponse, error) {
	req := &ChatRequest{
		Messages: messages,
		Stream:   false,
	}
	if err := c.applyDefaults(req); err != nil {
		return nil, err
	}

	var resp ChatResponse
	if err := c.doRequest(ctx, req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// StreamChunk is yielded by StreamChat.
type StreamChunk struct {
	Content   string
	Reasoning string // delta.reasoning_content when Thinking is on
	Usage     *Usage
	Done      bool
	Error     error
}

// StreamChat sends a streaming chat completion. Each chunk is delivered to the callback.
// The callback's return value controls whether streaming continues (true = continue).
//
// The client has no overall timeout, so long reasoning phases are not cut
// off; the stream ends when ctx is cancelled, the server closes it, or
// [DONE] arrives. A caller-supplied http.Client with a Timeout still applies
// to the whole stream (including body reads), so leave it at zero for
// streaming.
func (c *Client) StreamChat(ctx context.Context, messages []Message, cb func(StreamChunk)) error {
	req := &ChatRequest{
		Messages: messages,
		Stream:   true,
		StreamOptions: &StreamOptions{
			IncludeUsage:         true,
			ContinuousUsageStats: false, // usage once at end — less payload / cleaner clients
		},
	}
	if err := c.applyDefaults(req); err != nil {
		return err
	}

	return c.doStream(ctx, req, cb)
}

// --- Internal: doRequest (non-streaming) ---

func (c *Client) doRequest(ctx context.Context, req *ChatRequest, target interface{}) error {
	// Non-streaming requests keep the historical 120s budget. Streaming does
	// not: a whole-request deadline would cut long reasoning phases even
	// while tokens are still flowing (see doStream).
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, requestTimeout)
		defer cancel()
	}

	httpReq, err := c.buildRequest(ctx, req)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("nvidia: http do: %w", err)
	}
	defer resp.Body.Close()

	// Read full body
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("nvidia: read body: %w", err)
	}

	if resp.StatusCode >= 400 {
		return parseError(raw)
	}

	if err := json.Unmarshal(raw, target); err != nil {
		return fmt.Errorf("nvidia: decode response: %w", err)
	}
	return nil
}

// --- Internal: doStream (SSE streaming) ---

func (c *Client) doStream(ctx context.Context, req *ChatRequest, cb func(StreamChunk)) error {
	httpReq, err := c.buildRequest(ctx, req)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("nvidia: http do: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return parseError(raw)
	}

	// Read SSE stream. ReadString can return data together with io.EOF, so
	// always process the line before handling the read error.
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		line = strings.TrimSpace(line)

		// Skip empty lines / comments
		if line != "" && !strings.HasPrefix(line, ":") {
			// data: [DONE]
			if line == "data: [DONE]" {
				cb(StreamChunk{Done: true})
				return nil
			}

			// data: {...}
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")
				var chunk ChatChunk
				if err := json.Unmarshal([]byte(data), &chunk); err != nil {
					cb(StreamChunk{Error: fmt.Errorf("nvidia: decode chunk: %w", err)})
				} else {
					sc := StreamChunk{}
					if len(chunk.Choices) > 0 {
						sc.Content = chunk.Choices[0].Delta.Content
						sc.Reasoning = chunk.Choices[0].Delta.ReasoningContent
					}
					if chunk.Usage != nil {
						sc.Usage = chunk.Usage
					}
					cb(sc)
				}
			}
		}

		if err != nil {
			if err == io.EOF {
				return fmt.Errorf("nvidia: stream ended before [DONE]: %w", io.ErrUnexpectedEOF)
			}
			return fmt.Errorf("nvidia: read sse: %w", err)
		}
	}
}

// --- helpers ---

func parseError(raw []byte) error {
	var apiErr ErrorResponse
	if err := json.Unmarshal(raw, &apiErr); err == nil && apiErr.RequestStatus.StatusCode != "" {
		return &apiErr
	}
	// Try OpenAI error format
	var oaiErr struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &oaiErr); err == nil && oaiErr.Error.Message != "" {
		return fmt.Errorf("nvidia: %s: %s", oaiErr.Error.Type, oaiErr.Error.Message)
	}
	return fmt.Errorf("nvidia: http error: %s", string(raw[:min(len(raw), 300)]))
}
