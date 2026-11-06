package nvidia

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"nvidia-playgroud-go/internal/models"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestDoStreamProcessesFinalDeltaAtEOF(t *testing.T) {
	client := New()
	client.httpClient.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(
				"data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n" +
					"data: {\"choices\":[{\"delta\":{\"content\":\"last\"}}]}\n\n" +
					"data: [DONE]",
			)),
			Header: make(http.Header),
		}, nil
	})

	var content strings.Builder
	var doneCount int
	err := client.doStream(t.Context(), &ChatRequest{Model: DefaultModel}, func(chunk StreamChunk) {
		if chunk.Error != nil {
			t.Errorf("doStream() callback error = %v", chunk.Error)
		}
		content.WriteString(chunk.Content)
		if chunk.Done {
			doneCount++
		}
	})
	if err != nil {
		t.Fatalf("doStream() error = %v", err)
	}
	if got := content.String(); got != "firstlast" {
		t.Fatalf("stream content = %q, want %q", got, "firstlast")
	}
	if doneCount != 1 {
		t.Fatalf("done callbacks = %d, want 1", doneCount)
	}
}

func TestDoStreamRejectsPrematureEOF(t *testing.T) {
	client := New()
	client.httpClient.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(
				"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
			)),
			Header: make(http.Header),
		}, nil
	})

	var chunks []StreamChunk
	err := client.doStream(t.Context(), &ChatRequest{Model: DefaultModel}, func(chunk StreamChunk) {
		chunks = append(chunks, chunk)
	})
	if err == nil || !strings.Contains(err.Error(), "[DONE]") {
		t.Fatalf("doStream() error = %v, want premature EOF error", err)
	}
	if len(chunks) != 1 || chunks[0].Content != "partial" {
		t.Fatalf("chunks = %#v, want partial content before the error", chunks)
	}
}

func TestBuildRequestUsesCaptchaAuthentication(t *testing.T) {
	client := New(WithCaptchaToken("test-captcha-token"))
	chatRequest := &ChatRequest{
		Model: DefaultModel,
		Messages: []Message{
			{Role: RoleUser, Content: "Hello"},
		},
	}

	req, err := client.buildRequest(context.Background(), chatRequest)
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}

	if req.Method != http.MethodPost {
		t.Errorf("method = %q, want %q", req.Method, http.MethodPost)
	}
	info, err := models.Lookup(DefaultModel)
	if err != nil {
		t.Fatalf("models.Lookup(%q) error = %v", DefaultModel, err)
	}
	if req.URL.String() != info.PredictEndpoint() {
		t.Errorf("URL = %q, want %q", req.URL.String(), info.PredictEndpoint())
	}

	wantHeaders := map[string]string{
		"Content-Type":     "application/json",
		"Accept":           "text/event-stream",
		"nv-function-id":   info.FunctionID,
		"nv-captcha-token": "test-captcha-token",
		"Origin":           "https://build.nvidia.com",
		"Referer":          "https://build.nvidia.com/",
	}
	for name, want := range wantHeaders {
		if got := req.Header.Get(name); got != want {
			t.Errorf("header %q = %q, want %q", name, got, want)
		}
	}
	if got := req.Header.Get("Authorization"); got != "" {
		t.Errorf("Authorization header = %q, want empty", got)
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	var got ChatRequest
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if got.Model != models.RequestModel(chatRequest.Model) {
		t.Errorf("body model = %q, want %q", got.Model, models.RequestModel(chatRequest.Model))
	}
	if len(got.Messages) != 1 || got.Messages[0] != chatRequest.Messages[0] {
		t.Errorf("body messages = %#v, want %#v", got.Messages, chatRequest.Messages)
	}
}

func TestApplyDefaultsSkipsThinkingForUnverifiedModel(t *testing.T) {
	// The default model (kimi-k3_1-pro) has no Capability.Thinking hint, so the
	// client must NOT inject chat_template_kwargs (upstream 400s on it).
	client := New(WithCaptchaToken("t"))
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "Hi"}}}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	if len(req.ChatTemplateKwargs) != 0 {
		t.Fatalf("chat_template_kwargs = %#v, want none for unverified model", req.ChatTemplateKwargs)
	}
}

func TestApplyDefaultsEnablesThinkingWhenExplicit(t *testing.T) {
	// Use a toggle model: kimi takes reasoning_effort at the request root and
	// is covered by TestApplyDefaultsKimiUsesTopLevelEffort.
	client := New(WithCaptchaToken("t"), WithModel("google/diffusiongemma-26b-a4b-it"), WithThinking(true))
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "Hi"}}}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	if req.ChatTemplateKwargs["enable_thinking"] != true {
		t.Fatalf("enable_thinking = %#v, want true", req.ChatTemplateKwargs["enable_thinking"])
	}
	if req.ChatTemplateKwargs["clear_thinking"] != false {
		t.Fatalf("clear_thinking = %#v, want false", req.ChatTemplateKwargs["clear_thinking"])
	}
}

func TestWithThinkingFalse(t *testing.T) {
	client := New(WithCaptchaToken("t"), WithThinking(false))
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "Hi"}}}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	if req.ChatTemplateKwargs != nil {
		t.Fatalf("ChatTemplateKwargs = %#v, want nil when thinking disabled", req.ChatTemplateKwargs)
	}
}

func TestApplyDefaultsPreservesCallerKwargs(t *testing.T) {
	client := New(WithCaptchaToken("t"))
	req := &ChatRequest{
		Messages:           []Message{{Role: RoleUser, Content: "Hi"}},
		ChatTemplateKwargs: map[string]any{"enable_thinking": false},
	}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	if req.ChatTemplateKwargs["enable_thinking"] != false {
		t.Fatalf("got %#v", req.ChatTemplateKwargs)
	}
}

func TestApplyDefaultsFillsEmptyKwargsWhenExplicit(t *testing.T) {
	client := New(WithCaptchaToken("t"), WithModel("google/diffusiongemma-26b-a4b-it"), WithThinking(true))
	req := &ChatRequest{
		Messages:           []Message{{Role: RoleUser, Content: "Hi"}},
		ChatTemplateKwargs: map[string]any{},
	}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	if req.ChatTemplateKwargs["enable_thinking"] != true {
		t.Fatalf("empty kwargs should get defaults, got %#v", req.ChatTemplateKwargs)
	}
}

func TestApplyDefaultsMapsThinkingPerModel(t *testing.T) {
	// WithThinking must use the knob each model accepts. Sending
	// enable_thinking to deepseek/nemotron is rejected or ignored upstream.
	cases := []struct {
		model  string
		enable bool
		key    string
		want   any
	}{
		{"deepseek-ai/deepseek-v4-pro-0813", true, "reasoning_effort", "high"},
		{"deepseek-ai/deepseek-v4-pro-0813", false, "reasoning_effort", "none"},
		{"nvidia/nemotron-3-ultra-550b-a55b", true, "reasoning_effort", "medium"},
		{"minimaxai/minimax-m3", true, "thinking_mode", "enabled"},
		{"minimaxai/minimax-m3", false, "thinking_mode", "disabled"},
		{"google/diffusiongemma-26b-a4b-it", true, "enable_thinking", true},
		{"google/diffusiongemma-26b-a4b-it", false, "enable_thinking", false},
	}
	for _, tc := range cases {
		client := New(WithCaptchaToken("t"), WithModel(tc.model), WithThinking(tc.enable))
		req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "Hi"}}}
		if err := client.applyDefaults(req); err != nil {
			t.Fatal(err)
		}
		if got := req.ChatTemplateKwargs[tc.key]; got != tc.want {
			t.Errorf("%s enable=%v: %s = %#v, want %#v (kwargs %#v)",
				tc.model, tc.enable, tc.key, got, tc.want, req.ChatTemplateKwargs)
		}
		for k := range req.ChatTemplateKwargs {
			if k != tc.key && k != "clear_thinking" {
				t.Errorf("%s enable=%v: unexpected kwarg %q in %#v", tc.model, tc.enable, k, req.ChatTemplateKwargs)
			}
		}
	}
}

func TestApplyDefaultsKimiUsesTopLevelEffort(t *testing.T) {
	cases := []struct {
		name string
		opts []Option
		want string
	}{
		{"thinking on", []Option{WithThinking(true)}, "low"},
		{"thinking off", []Option{WithThinking(false)}, "low"}, // no "none" level
		{"explicit max", []Option{WithReasoningEffort("max")}, "max"},
		{"explicit beats thinking", []Option{WithThinking(true), WithReasoningEffort("medium")}, "medium"},
	}
	for _, tc := range cases {
		opts := append([]Option{WithCaptchaToken("t"), WithModel("moonshotai/kimi-k3_1-pro")}, tc.opts...)
		client := New(opts...)
		req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "Hi"}}}
		if err := client.applyDefaults(req); err != nil {
			t.Fatal(err)
		}
		if req.ReasoningEffort != tc.want {
			t.Errorf("%s: ReasoningEffort = %q, want %q", tc.name, req.ReasoningEffort, tc.want)
		}
		if len(req.ChatTemplateKwargs) != 0 {
			t.Errorf("%s: kimi must not receive kwargs, got %#v", tc.name, req.ChatTemplateKwargs)
		}
	}
}

func TestClientPinnedAndCallerKwargsPreserveRequestedEffort(t *testing.T) {
	const pin = "4f1a2b3c-0000-4000-8000-000000000001"
	for _, effort := range []string{"low", "high", "max"} {
		t.Run(effort, func(t *testing.T) {
			client := New(
				WithModel(pin+"@moonshotai/kimi-k3_1-pro"),
				WithReasoningEffort(effort),
				WithThinking(true),
				WithHTTPClient(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
					var raw map[string]any
					payload, err := io.ReadAll(req.Body)
					if err != nil {
						t.Fatal(err)
					}
					if err := json.Unmarshal(payload, &raw); err != nil {
						t.Fatal(err)
					}
					if raw["reasoning_effort"] != effort {
						t.Errorf("reasoning_effort = %#v, want %q; body=%s", raw["reasoning_effort"], effort, payload)
					}
					if raw["model"] != models.RequestModel(DefaultModel) {
						t.Errorf("model = %#v, want canonical Kimi wire model", raw["model"])
					}
					if raw["chat_template_kwargs"] != nil {
						t.Errorf("unexpected reasoning kwargs: %s", raw)
					}
					if req.Header.Get("nv-function-id") != pin {
						t.Errorf("function id = %q, want %q", req.Header.Get("nv-function-id"), pin)
					}
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"choices":[]}`)), Header: http.Header{}}, nil
				})}),
			)
			if _, err := client.Chat(t.Context(), []Message{{Role: RoleUser, Content: "hi"}}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestClientEffortDoesNotGetSuppressedByCallerKwargs(t *testing.T) {
	client := New(WithReasoningEffort("max"))
	req := &ChatRequest{
		Model:              "moonshotai/kimi-k3_1-pro",
		ChatTemplateKwargs: map[string]any{"seed": 42},
	}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	if req.ReasoningEffort != "max" || req.ChatTemplateKwargs["seed"] != 42 {
		t.Fatalf("effort = %q, kwargs = %#v; want max and preserved kwargs", req.ReasoningEffort, req.ChatTemplateKwargs)
	}
}

func TestClientReasoningDefaultsToLowAndPreservesExplicitLevels(t *testing.T) {
	for _, effort := range []string{"", "low", "high", "max"} {
		name := effort
		if name == "" {
			name = "default"
		}
		t.Run(name, func(t *testing.T) {
			client := New(WithModel("moonshotai/kimi-k3_1-pro"), WithThinking(true))
			if effort != "" {
				client = New(WithModel("moonshotai/kimi-k3_1-pro"), WithReasoningEffort(effort))
			}
			req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}
			if err := client.applyDefaults(req); err != nil {
				t.Fatal(err)
			}
			want := effort
			if want == "" {
				want = "low"
			}
			if req.ReasoningEffort != want {
				t.Fatalf("effort = %q, want %q", req.ReasoningEffort, want)
			}
		})
	}
}

func TestClientRejectsInvalidReasoningEffortBeforeHTTP(t *testing.T) {
	client := New(
		WithCaptchaToken("t"),
		WithModel("moonshotai/kimi-k3_1-pro"),
		WithReasoningEffort("hgih"),
		WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			t.Fatal("invalid effort must fail before an HTTP request")
			return nil, nil
		})}),
	)
	_, err := client.Chat(t.Context(), []Message{{Role: RoleUser, Content: "Hi"}})
	if err == nil || !strings.Contains(err.Error(), "invalid reasoning_effort") {
		t.Fatalf("Chat() error = %v, want invalid reasoning_effort", err)
	}
}

func TestBuildRequestKimiWireModel(t *testing.T) {
	client := New(WithCaptchaToken("t"), WithModel("moonshotai/kimi-k3_1-pro"))
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "Hi"}}}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	httpReq, err := client.buildRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var got ChatRequest
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if got.Model != "moonshotai/kimi-k3.1-pro" {
		t.Errorf("body model = %q, want the dot form", got.Model)
	}
	if req.Model != "moonshotai/kimi-k3_1-pro" {
		t.Errorf("caller request model = %q, want the registry id", req.Model)
	}
}

func TestNewHasNoOverallTimeout(t *testing.T) {
	// http.Client.Timeout also bounds response-body reads, so a default
	// timeout would truncate long SSE streams (long reasoning phases).
	if got := New().httpClient.Timeout; got != 0 {
		t.Fatalf("default client timeout = %v, want 0 (streams must not be capped)", got)
	}
}

// buildRequest must route each model via its own nv-function-id on the shared
// stateless endpoint (since 2026-11 the URL is model-independent). Pinned to
// two concrete registry entries so a future scrape that changes ids is caught.
func TestBuildRequestRoutesPerModel(t *testing.T) {
	client := New(WithCaptchaToken("tok"))
	cases := map[string]struct {
		endpoint string
		fnID     string
	}{
		"moonshotai/kimi-k3_1-pro": {
			"https://buildapi.ngc.nvidia.com/v3/predict/stateless/v1/chat/completions",
			"b79234a1-64e1-479e-87e7-1c8e3af74262",
		},
		"deepseek-ai/deepseek-v4-pro-0813": {
			"https://buildapi.ngc.nvidia.com/v3/predict/stateless/v1/chat/completions",
			"6e70713f-4eeb-4ef7-b4f8-2d984f4141f6",
		},
	}
	for model, want := range cases {
		req, err := client.buildRequest(context.Background(), &ChatRequest{
			Model: model, Messages: []Message{{Role: RoleUser, Content: "Hi"}},
		})
		if err != nil {
			t.Errorf("%s: buildRequest: %v", model, err)
			continue
		}
		if got := req.URL.String(); got != want.endpoint {
			t.Errorf("%s: endpoint = %q want %q", model, got, want.endpoint)
		}
		if got := req.Header.Get("nv-function-id"); got != want.fnID {
			t.Errorf("%s: nv-function-id = %q want %q", model, got, want.fnID)
		}
	}
}

func TestBuildRequestUnknownModel(t *testing.T) {
	client := New(WithCaptchaToken("tok"))
	_, err := client.buildRequest(context.Background(), &ChatRequest{
		Model: "no-such-org/never", Messages: []Message{{Role: RoleUser, Content: "Hi"}},
	})
	if err == nil {
		t.Fatal("expected error for unknown model")
	}
	if !strings.Contains(err.Error(), "no-such-org/never") {
		t.Fatalf("error %q should name the model", err.Error())
	}
}

// TestBuildRequestAppliesFunctionPin checks the "function-id@model" form: the
// pinned id becomes the nv-function-id header, the endpoint still comes from
// the registry, and the body carries the plain model id.
func TestBuildRequestAppliesFunctionPin(t *testing.T) {
	const pinned = "4f1a2b3c-0000-4000-8000-000000000001"
	client := New(WithCaptchaToken("t"), WithModel(pinned+"@"+DefaultModel))

	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "Hi"}}}
	if err := client.applyDefaults(req); err != nil {
		t.Fatal(err)
	}
	httpReq, err := client.buildRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}
	info, err := models.Lookup(DefaultModel)
	if err != nil {
		t.Fatalf("models.Lookup(%q) error = %v", DefaultModel, err)
	}
	if httpReq.URL.String() != info.PredictEndpoint() {
		t.Errorf("URL = %q, want the pinned model's endpoint %q", httpReq.URL.String(), info.PredictEndpoint())
	}
	if got := httpReq.Header.Get("nv-function-id"); got != pinned {
		t.Errorf("nv-function-id = %q, want %q", got, pinned)
	}
	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var got ChatRequest
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if got.Model != models.RequestModel(DefaultModel) {
		t.Errorf("body model = %q, want %q without the pin", got.Model, models.RequestModel(DefaultModel))
	}
	if req.Model != DefaultModel {
		t.Errorf("caller request model = %q, want %q", req.Model, DefaultModel)
	}
}

// TestBuildRequestPinRejectsMalformedID keeps a typo out of the wire: an empty
// or unsafe function id must fail locally instead of 400-ing upstream.
func TestBuildRequestPinRejectsMalformedID(t *testing.T) {
	for _, model := range []string{"@moonshotai/kimi-k3_1-pro", pinnedBadID + "@moonshotai/kimi-k3_1-pro", "4f1a2b3c-0000-4000-8000-000000000001@"} {
		client := New(WithCaptchaToken("t"), WithModel(model))
		if _, err := client.buildRequest(context.Background(), &ChatRequest{Model: model, Messages: []Message{{Role: RoleUser, Content: "Hi"}}}); err == nil {
			t.Errorf("buildRequest(%q) error = nil, want a client error", model)
		}
	}
}

const pinnedBadID = "not a uuid"

// TestBuildRequestModelInfoOverrideWins documents the precedence with an
// explicit target: WithModelInfo pins namespace/slug/function id, and a model
// pin only strips itself out of the body.
func TestBuildRequestModelInfoOverrideWins(t *testing.T) {
	const pinned = "4f1a2b3c-0000-4000-8000-000000000001"
	client := New(
		WithCaptchaToken("t"),
		WithModelInfo(models.ModelInfo{Namespace: "ns", Slug: "sl", FunctionID: "override-id"}),
	)
	model := pinned + "@" + DefaultModel
	httpReq, err := client.buildRequest(context.Background(), &ChatRequest{Model: model, Messages: []Message{{Role: RoleUser, Content: "Hi"}}})
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}
	if got := httpReq.Header.Get("nv-function-id"); got != "override-id" {
		t.Errorf("nv-function-id = %q, want the explicit override", got)
	}
	if want := "https://buildapi.ngc.nvidia.com/v3/predict/stateless/v1/chat/completions"; httpReq.URL.String() != want {
		t.Errorf("URL = %q, want %q", httpReq.URL.String(), want)
	}
	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatalf("io.ReadAll(httpReq.Body) error = %v", err)
	}
	if strings.Contains(string(body), pinned) {
		t.Errorf("body still carries the pin: %s", body)
	}
}
