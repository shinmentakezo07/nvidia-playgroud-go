package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	nvidia "nvidia-playgroud-go"
	"nvidia-playgroud-go/internal/models"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestResolveTarget(t *testing.T) {
	tests := []struct {
		name    string
		cfg     config
		wantURL string
		wantFID string
		wantMod string
		wantErr string
	}{
		{
			name:    "no flags falls back to the default registry model",
			cfg:     config{},
			wantURL: models.PredictBase + "/v3/predict/stateless/v1/chat/completions",
			wantFID: "b79234a1-64e1-479e-87e7-1c8e3af74262",
			wantMod: models.DefaultModel,
		},
		{
			name:    "registry model fills the triple",
			cfg:     config{model: "minimaxai/minimax-m3"},
			wantURL: models.PredictBase + "/v3/predict/stateless/v1/chat/completions",
			wantFID: "87ea0ddc-cff1-4bca-bf8b-3bd98a35ddd0",
			wantMod: "minimaxai/minimax-m3",
		},
		{
			name: "explicit flags override the registry",
			cfg: config{
				model:      "minimaxai/minimax-m3",
				namespace:  "otherns",
				slug:       "some-slug",
				functionID: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
			},
			wantURL: models.PredictBase + "/v3/predict/stateless/v1/chat/completions",
			wantFID: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
			wantMod: "minimaxai/minimax-m3",
		},
		{
			name:    "slug override keeps the pinned function id",
			cfg:     config{model: "moonshotai/kimi-k3_1-pro", slug: "kimi-k2"},
			wantURL: models.PredictBase + "/v3/predict/stateless/v1/chat/completions",
			wantFID: "b79234a1-64e1-479e-87e7-1c8e3af74262",
			wantMod: "moonshotai/kimi-k3_1-pro",
		},
		{
			name:    "unknown model with a pinned triple",
			cfg:     config{model: "acme/new-llm", slug: "new-llm", functionID: "fid-1"},
			wantURL: models.PredictBase + "/v3/predict/stateless/v1/chat/completions",
			wantFID: "fid-1",
			wantMod: "acme/new-llm",
		},
		{
			name:    "unknown model without a triple fails with guidance",
			cfg:     config{model: "acme/new-llm"},
			wantErr: "pass -slug and -function-id",
		},
		{
			name:    "model-field pin overrides only the function id",
			cfg:     config{model: "4f1a2b3c-0000-4000-8000-000000000001@moonshotai/kimi-k3_1-pro"},
			wantURL: models.PredictBase + "/v3/predict/stateless/v1/chat/completions",
			wantFID: "4f1a2b3c-0000-4000-8000-000000000001",
			wantMod: "moonshotai/kimi-k3_1-pro",
		},
		{
			name: "explicit -function-id wins over the model-field pin",
			cfg: config{
				model:      "4f1a2b3c-0000-4000-8000-000000000001@moonshotai/kimi-k3_1-pro",
				functionID: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
			},
			wantURL: models.PredictBase + "/v3/predict/stateless/v1/chat/completions",
			wantFID: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
			wantMod: "moonshotai/kimi-k3_1-pro",
		},
		{
			name:    "pin on an unknown model still needs the triple",
			cfg:     config{model: "4f1a2b3c-0000-4000-8000-000000000001@acme/new-llm"},
			wantErr: "unknown model: acme/new-llm",
		},
		{
			name:    "malformed pin is rejected",
			cfg:     config{model: "not a uuid@moonshotai/kimi-k3_1-pro"},
			wantErr: "invalid nv-function-id",
		},
		{
			name:    "pin without a model is rejected",
			cfg:     config{model: "4f1a2b3c-0000-4000-8000-000000000001@"},
			wantErr: "missing model id",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, model, err := resolveTarget(&tt.cfg)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want it to contain %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveTarget: %v", err)
			}
			if got := info.PredictEndpoint(); got != tt.wantURL {
				t.Errorf("endpoint = %s, want %s", got, tt.wantURL)
			}
			if info.FunctionID != tt.wantFID {
				t.Errorf("function-id = %s, want %s", info.FunctionID, tt.wantFID)
			}
			if model != tt.wantMod {
				t.Errorf("body model = %s, want %s", model, tt.wantMod)
			}
		})
	}
}

// TestResolveTargetCapability checks the thinking-capability propagation that
// drives -thinking=auto: carried over when the slug matches the registry
// entry, dropped when -slug points at a different endpoint.
func TestResolveTargetCapability(t *testing.T) {
	orig := models.All()
	defer models.Replace(orig)
	models.Replace(map[string]models.ModelInfo{
		"acme/thinker": {
			Slug: "thinker", Namespace: "ns9", FunctionID: "fid9",
			Capability: &models.ModelCapability{Thinking: true},
		},
	})

	matched, _, err := resolveTarget(&config{model: "acme/thinker"})
	if err != nil {
		t.Fatal(err)
	}
	if matched.Capability == nil || !matched.Capability.Thinking {
		t.Errorf("matching slug should carry the capability, got %+v", matched.Capability)
	}

	overridden, _, err := resolveTarget(&config{model: "acme/thinker", slug: "other"})
	if err != nil {
		t.Fatal(err)
	}
	if overridden.Capability != nil {
		t.Errorf("overridden slug should drop the capability, got %+v", overridden.Capability)
	}
}

func TestBuildMessages(t *testing.T) {
	t.Run("system plus prompt", func(t *testing.T) {
		msgs, err := buildMessages(&config{system: "s", prompt: "p"})
		if err != nil {
			t.Fatal(err)
		}
		if len(msgs) != 2 || msgs[0].Role != "system" || msgs[1].Content != "p" {
			t.Fatalf("got %+v", msgs)
		}
	})

	t.Run("messages json", func(t *testing.T) {
		msgs, err := buildMessages(&config{messages: `[{"role":"user","content":"hi"}]`, prompt: "ignored"})
		if err != nil {
			t.Fatal(err)
		}
		if len(msgs) != 1 || msgs[0].Content != "hi" {
			t.Fatalf("got %+v", msgs)
		}
	})

	t.Run("messages json wins over prompt", func(t *testing.T) {
		if _, err := buildMessages(&config{messages: `[]`}); err == nil {
			t.Fatal("empty array should error")
		}
		if _, err := buildMessages(&config{messages: `not json`}); err == nil {
			t.Fatal("invalid json should error")
		}
	})
}

func TestRequestBodyRejectsInvalidReasoningEffort(t *testing.T) {
	_, err := (&config{reasoningEffort: "hgih"}).requestBody("moonshotai/kimi-k3_1-pro", []nvidia.Message{{Role: nvidia.RoleUser, Content: "test"}})
	if err == nil || !strings.Contains(err.Error(), "invalid reasoning_effort") {
		t.Fatalf("requestBody() error = %v, want invalid effort", err)
	}
}

func TestCurlCommand(t *testing.T) {
	info := models.ModelInfo{Namespace: "ns1", Slug: "sl1", FunctionID: "fid1"}
	cmd := curlCommand(info, "P1_tok", []byte(`{"model":"a/b","messages":[{"role":"user","content":"it's"}]}`))

	for _, want := range []string{
		"/v3/predict/stateless/v1/chat/completions",
		"-H 'nv-function-id: fid1'",
		"-H 'nv-captcha-token: P1_tok'",
		`'\''`, // embedded single quote escaped for the shell
	} {
		if !strings.Contains(cmd, want) {
			t.Errorf("curl output missing %q:\n%s", want, cmd)
		}
	}
}

// TestSendOnceSharesHTTPClient proves that recreating the API client for
// one-shot captchas does not discard warm HTTP connections. Queue status is
// irrelevant: every request must go directly to the predict endpoint.
func TestSendOnceReusesHTTPConnections(t *testing.T) {
	var connCount atomic.Int32
	var pathsMu sync.Mutex
	var paths []string
	upstream := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/queues/") {
			t.Errorf("prediction must not wait for queue-status polling: %s", r.URL)
		}
		pathsMu.Lock()
		paths = append(paths, r.URL.Path)
		pathsMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	upstream.Config.ConnContext = func(ctx context.Context, _ net.Conn) context.Context {
		connCount.Add(1)
		return ctx
	}
	upstream.Start()
	defer upstream.Close()

	upstreamURL, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		r.URL.Scheme = upstreamURL.Scheme
		r.URL.Host = upstreamURL.Host
		return upstream.Client().Transport.RoundTrip(r)
	})}
	defer closeIdleConnections(client)
	info, model, err := resolveTarget(&config{})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config{stream: false, thinking: "auto", timeout: time.Second}
	for i := 0; i < 2; i++ {
		if err := sendOnce(t.Context(), cfg, info, model, "one-shot-token", []nvidia.Message{{Role: "user", Content: "hi"}}, client); err != nil {
			t.Fatal(err)
		}
	}
	if got := connCount.Load(); got != 1 {
		t.Fatalf("upstream connections = %d, want one reused keep-alive connection", got)
	}
}

func TestSendOncePinnedKimiUsesLowDefaultAndExplicitLevels(t *testing.T) {
	const pin = "4f1a2b3c-0000-4000-8000-000000000001"
	for _, effort := range []string{"", "low", "high", "max"} {
		name := effort
		if name == "" {
			name = "default"
		}
		t.Run(name, func(t *testing.T) {
			client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				var raw map[string]any
				payload, err := io.ReadAll(req.Body)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(payload, &raw); err != nil {
					t.Fatal(err)
				}
				want := effort
				if want == "" {
					want = "low"
				}
				if raw["reasoning_effort"] != want {
					t.Errorf("reasoning_effort = %#v, want %s; body=%s", raw["reasoning_effort"], want, payload)
				}
				if req.Header.Get("nv-function-id") != pin {
					t.Errorf("function id = %q, want pin", req.Header.Get("nv-function-id"))
				}
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"choices":[]}`)), Header: http.Header{}}, nil
			})}
			defer closeIdleConnections(client)
			cfg := &config{stream: false, thinking: "on", reasoningEffort: effort}
			info, model, err := resolveTarget(&config{model: pin + "@" + models.DefaultModel})
			if err != nil {
				t.Fatal(err)
			}
			if err := sendOnce(t.Context(), cfg, info, model, "test-token", []nvidia.Message{{Role: "user", Content: "hi"}}, client); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestTokenForRejectsReplay(t *testing.T) {
	c := &config{captcha: "P1_one_shot"}
	if tok, err := c.tokenFor(context.TODO(), 1); err != nil || tok != "P1_one_shot" {
		t.Fatalf("first use: tok=%q err=%v", tok, err)
	}
	if _, err := c.tokenFor(context.TODO(), 2); err == nil {
		t.Fatal("replaying a one-shot token should error")
	}
}
