package nvidia

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	clipexec "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"nvidia-playgroud-go/internal/models"
)

func TestPreparePayloadReasoningAcrossClientAPIs(t *testing.T) {
	tests := []struct {
		name   string
		format sdktranslator.Format
		input  string
		want   string
	}{
		{
			name:   "OpenAI Chat max",
			format: sdktranslator.FormatOpenAI,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`,
			want:   "max",
		},
		{
			name:   "Responses maps effort",
			format: sdktranslator.FormatOpenAIResponse,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","input":"hello","reasoning":{"effort":"medium"}}`,
			want:   "medium",
		},
		{
			name:   "Responses none rounds to Kimi minimum",
			format: sdktranslator.FormatOpenAIResponse,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","input":"hello","reasoning":{"effort":"none"}}`,
			want:   "low",
		},
		{
			name:   "Claude enabled without budget uses default",
			format: sdktranslator.FormatClaude,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","max_tokens":32,"thinking":{"type":"enabled"},"messages":[{"role":"user","content":"hi"}]}`,
			want:   "low",
		},
		{
			name:   "Claude enabled with budget preserves mapped effort",
			format: sdktranslator.FormatClaude,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","max_tokens":32000,"thinking":{"type":"enabled","budget_tokens":16384},"messages":[{"role":"user","content":"hi"}]}`,
			want:   "high",
		},
		{
			name:   "Claude disabled selects minimum available Kimi effort",
			format: sdktranslator.FormatClaude,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","max_tokens":32000,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`,
			want:   "low",
		},
		{
			name:   "Claude adaptive uses explicit effort",
			format: sdktranslator.FormatClaude,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","max_tokens":32000,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`,
			want:   "high",
		},
		{
			name:   "Claude adaptive without effort uses default",
			format: sdktranslator.FormatClaude,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","max_tokens":32000,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hi"}]}`,
			want:   "low",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body, _, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
				Model: "moonshotai/kimi-k3_1-pro", Payload: []byte(tc.input),
			}, clipexec.Options{SourceFormat: tc.format}, false)
			if err != nil {
				t.Fatal(err)
			}
			var raw map[string]any
			if err := json.Unmarshal(body, &raw); err != nil {
				t.Fatal(err)
			}
			if raw["reasoning_effort"] != tc.want {
				t.Errorf("reasoning_effort = %#v, want %q: %s", raw["reasoning_effort"], tc.want, body)
			}
			if raw["model"] != "moonshotai/kimi-k3.1-pro" {
				t.Errorf("model = %#v, want Kimi wire model", raw["model"])
			}
			for _, key := range []string{"thinking", "reasoning", "chat_template_kwargs"} {
				if _, ok := raw[key]; ok {
					t.Errorf("%s leaked into Kimi upstream request: %s", key, body)
				}
			}
		})
	}
}

func TestPreparePayloadKimiForwardsClientEffortUnchanged(t *testing.T) {
	for _, effort := range []string{"low", "medium", "high", "max"} {
		t.Run(effort, func(t *testing.T) {
			var upstreamBody map[string]json.RawMessage
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read upstream request: %v", err)
					return
				}
				if err := json.Unmarshal(body, &upstreamBody); err != nil {
					t.Errorf("decode upstream request: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"1","choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
			}))
			defer upstream.Close()

			executor := NewExecutor(Options{
				FlagCaptcha: "test-token",
				HTTPClient:  upstream.Client(),
				PredictURL:  func(models.ModelInfo) string { return upstream.URL },
			})
			payload := []byte(`{"model":"moonshotai/kimi-k3_1-pro","reasoning_effort":"` + effort + `","messages":[{"role":"user","content":"hi"}]}`)
			if _, err := executor.Execute(t.Context(), nil, clipexec.Request{
				Model: "moonshotai/kimi-k3_1-pro", Payload: payload,
			}, clipexec.Options{SourceFormat: sdktranslator.FormatOpenAI}); err != nil {
				t.Fatal(err)
			}
			if got := string(upstreamBody["reasoning_effort"]); got != `"`+effort+`"` {
				t.Fatalf("upstream reasoning_effort = %s, want %q; body=%v", got, effort, upstreamBody)
			}
			if _, exists := upstreamBody["chat_template_kwargs"]; exists {
				t.Fatalf("Kimi effort must remain top-level: %v", upstreamBody)
			}
		})
	}
}

func TestPreparePayloadClaudeDisabledThinking(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  string
	}{
		{model: "moonshotai/kimi-k3_1-pro", want: "low"},
		{model: "deepseek-ai/deepseek-v4-pro-0813", want: "none"},
	} {
		input := []byte(`{"model":"` + tc.model + `","max_tokens":32000,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`)
		body, _, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
			Model: tc.model, Payload: input,
		}, clipexec.Options{SourceFormat: sdktranslator.FormatClaude}, false)
		if err != nil {
			t.Fatal(err)
		}
		var raw map[string]any
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatal(err)
		}
		if tc.model == "moonshotai/kimi-k3_1-pro" {
			if raw["reasoning_effort"] != "low" {
				t.Fatalf("Kimi disabled reasoning = %#v, want minimum low", raw["reasoning_effort"])
			}
		} else {
			kw, ok := raw["chat_template_kwargs"].(map[string]any)
			if !ok {
				t.Fatalf("expected kwargs: %s", body)
			}
			if kw["reasoning_effort"] != "none" {
				t.Fatalf("DeepSeek disabled reasoning = %#v, want none: %s", kw, body)
			}
		}
	}
}

func TestPreparePayloadRejectsEffectiveConflictingReasoningControls(t *testing.T) {
	_, _, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
		Model:   "moonshotai/kimi-k3_1-pro",
		Payload: []byte(`{"model":"moonshotai/kimi-k3_1-pro","chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"high"},"messages":[{"role":"user","content":"hi"}]}`),
	}, clipexec.Options{SourceFormat: sdktranslator.FormatOpenAI}, false)
	if err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("error = %v, want conflicting disabled thinking and high effort", err)
	}
}

func TestPreparePayloadEffortOverridesSameSourceDisabledThinking(t *testing.T) {
	for _, effort := range []string{"low", "high", "max"} {
		t.Run(effort, func(t *testing.T) {
			body, _, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
				Model:   "moonshotai/kimi-k3_1-pro",
				Payload: []byte(`{"model":"moonshotai/kimi-k3_1-pro","enable_thinking":false,"reasoning_effort":"` + effort + `","messages":[{"role":"user","content":"hi"}]}`),
			}, clipexec.Options{SourceFormat: sdktranslator.FormatOpenAI}, false)
			if err != nil {
				t.Fatalf("same-source switch must not reject explicit effort: %v", err)
			}
			var raw map[string]any
			if err := json.Unmarshal(body, &raw); err != nil {
				t.Fatal(err)
			}
			if raw["reasoning_effort"] != effort {
				t.Fatalf("reasoning_effort = %#v, want %s: %s", raw["reasoning_effort"], effort, body)
			}
		})
	}
}

func TestPreparePayloadRejectsUnsupportedReasoningForUnlistedTarget(t *testing.T) {
	headers := http.Header{}
	headers.Set(headerFunctionID, pinnedFunction)
	headers.Set(headerFunctionSlug, "acme/unlisted")
	_, _, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
		Model:   "moonshotai/kimi-k3_1-pro",
		Payload: []byte(`{"model":"moonshotai/kimi-k3_1-pro","reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`),
	}, clipexec.Options{SourceFormat: sdktranslator.FormatOpenAI, Headers: headers}, false)
	var apiErr *coreauth.Error
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v (%T), want *coreauth.Error", err, err)
	}
	if apiErr.HTTPStatus != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; error=%v", apiErr.HTTPStatus, err)
	}
	if !strings.Contains(apiErr.Message, "acme/unlisted") || !strings.Contains(apiErr.Message, "reasoning_effort") {
		t.Fatalf("error = %q, want unsupported reasoning control for the unlisted target", apiErr.Message)
	}
}

func TestPreparePayloadRejectsNestedUnsupportedReasoningForUnlistedTarget(t *testing.T) {
	headers := http.Header{}
	headers.Set(headerFunctionID, pinnedFunction)
	headers.Set(headerFunctionSlug, "acme/unlisted")
	_, _, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
		Model:   "moonshotai/kimi-k3_1-pro",
		Payload: []byte(`{"model":"moonshotai/kimi-k3_1-pro","chat_template_kwargs":{"reasoning_effort":"high"},"messages":[{"role":"user","content":"hi"}]}`),
	}, clipexec.Options{SourceFormat: sdktranslator.FormatOpenAI, Headers: headers}, false)
	var apiErr *coreauth.Error
	if !errors.As(err, &apiErr) || apiErr.HTTPStatus != http.StatusBadRequest {
		t.Fatalf("error = %v (%T), want HTTP 400", err, err)
	}
	if !strings.Contains(apiErr.Message, "chat_template_kwargs.reasoning_effort") {
		t.Fatalf("error = %q, want nested unsupported control", apiErr.Message)
	}
}

func TestPreparePayloadAllowsUnlistedTargetWithoutReasoning(t *testing.T) {
	headers := http.Header{}
	headers.Set(headerFunctionID, pinnedFunction)
	headers.Set(headerFunctionSlug, "acme/unlisted")
	body, info, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
		Model:   "moonshotai/kimi-k3_1-pro",
		Payload: []byte(`{"model":"moonshotai/kimi-k3_1-pro","messages":[{"role":"user","content":"hi"}]}`),
	}, clipexec.Options{SourceFormat: sdktranslator.FormatOpenAI, Headers: headers}, false)
	if err != nil {
		t.Fatalf("unlisted target without reasoning controls must still be callable: %v", err)
	}
	if info.FunctionID != pinnedFunction {
		t.Fatalf("function id = %q, want %q", info.FunctionID, pinnedFunction)
	}
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatal(err)
	}
	if raw["model"] != "acme/unlisted" {
		t.Fatalf("model = %#v, want unlisted target; body=%s", raw["model"], body)
	}
	if _, exists := raw["reasoning_effort"]; exists {
		t.Fatalf("reasoning control was injected into an unprofiled target: %s", body)
	}
}

func TestPreparePayloadRejectsInvalidReasoningEffort(t *testing.T) {
	for _, tc := range []struct {
		name   string
		format sdktranslator.Format
		input  string
	}{
		{
			name:   "OpenAI Chat",
			format: sdktranslator.FormatOpenAI,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","reasoning_effort":"hgih","messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name:   "Responses",
			format: sdktranslator.FormatOpenAIResponse,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","input":"hi","reasoning":{"effort":"extreme"}}`,
		},
		{
			name:   "Claude",
			format: sdktranslator.FormatClaude,
			input:  `{"model":"moonshotai/kimi-k3_1-pro","max_tokens":32000,"thinking":{"type":"adaptive"},"output_config":{"effort":"extreme"},"messages":[{"role":"user","content":"hi"}]}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := NewExecutor(Options{}).preparePayload(clipexec.Request{
				Model: "moonshotai/kimi-k3_1-pro", Payload: []byte(tc.input),
			}, clipexec.Options{SourceFormat: tc.format}, false)
			if err == nil || !strings.Contains(err.Error(), "reasoning_effort") {
				t.Fatalf("error = %v, want explicit invalid reasoning effort", err)
			}
		})
	}
}
