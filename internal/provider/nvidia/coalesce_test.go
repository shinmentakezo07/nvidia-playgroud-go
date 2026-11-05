package nvidia

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

func TestMergeableContent(t *testing.T) {
	okChunk := map[string]any{
		"choices": []any{
			map[string]any{
				"delta":         map[string]any{"content": "hi"},
				"finish_reason": nil,
			},
		},
	}
	got, ok := mergeableContent(okChunk)
	if !ok || got != "hi" {
		t.Fatalf("mergeable content: got %q ok=%v", got, ok)
	}

	withUsage := map[string]any{
		"usage": map[string]any{"total_tokens": 1},
		"choices": []any{
			map[string]any{"delta": map[string]any{"content": "x"}},
		},
	}
	if _, ok := mergeableContent(withUsage); ok {
		t.Fatal("usage chunk should not be mergeable")
	}

	withReason := map[string]any{
		"choices": []any{
			map[string]any{
				"delta": map[string]any{
					"content":           "x",
					"reasoning_content": "think",
				},
			},
		},
	}
	if _, ok := mergeableContent(withReason); ok {
		t.Fatal("reasoning delta should not be mergeable")
	}
}

func TestTryMergeContent(t *testing.T) {
	pending := map[string]any{
		"choices": []any{
			map[string]any{"delta": map[string]any{"content": "Hel"}},
		},
	}
	if !tryMergeContent(pending, "lo") {
		t.Fatal("merge failed")
	}
	ch := pending["choices"].([]any)[0].(map[string]any)
	delta := ch["delta"].(map[string]any)
	if delta["content"] != "Hello" {
		t.Fatalf("got %v", delta["content"])
	}
}

func TestCoalesceSSEMergesWindow(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":null}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"C"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	var events []string
	if err := coalesceSSEEvents(strings.NewReader(upstream), 50*time.Millisecond, func(line string) error {
		events = append(events, strings.TrimPrefix(line, "data: "))
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	if len(events) < 2 {
		t.Fatalf("expected coalesced content + DONE, got %d events: %#v", len(events), events)
	}
	if events[len(events)-1] != "[DONE]" {
		t.Fatalf("last event want [DONE], got %q", events[len(events)-1])
	}

	var content strings.Builder
	for _, ev := range events[:len(events)-1] {
		var chunk map[string]any
		if err := json.Unmarshal([]byte(ev), &chunk); err != nil {
			t.Fatal(err)
		}
		choices := chunk["choices"].([]any)
		delta := choices[0].(map[string]any)["delta"].(map[string]any)
		content.WriteString(delta["content"].(string))
	}
	if content.String() != "ABC" {
		t.Fatalf("content=%q events=%d", content.String(), len(events)-1)
	}
	if len(events)-1 < 1 || len(events)-1 > 3 {
		t.Fatalf("want 1-3 content events after eager-first flush, got %d", len(events)-1)
	}
	var first map[string]any
	if err := json.Unmarshal([]byte(events[0]), &first); err != nil {
		t.Fatal(err)
	}
	firstContent := first["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["content"].(string)
	if firstContent != "A" {
		t.Fatalf("first event content=%q want A (eager TTFT flush)", firstContent)
	}
}

func TestCoalesceSSEPassthroughWhenOff(t *testing.T) {
	upstream := "data: {\"choices\":[{\"delta\":{\"content\":\"Z\"}}]}\n\ndata: [DONE]\n\n"
	var body strings.Builder
	if err := coalesceSSEEvents(strings.NewReader(upstream), 0, func(line string) error {
		body.WriteString(line)
		body.WriteString("\n")
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body.String(), "Z") {
		t.Fatalf("passthrough body=%q", body.String())
	}
}

func TestCoalesceSSEFlushesPendingDeltaAtEOF(t *testing.T) {
	for _, window := range []time.Duration{0, time.Hour} {
		t.Run(window.String(), func(t *testing.T) {
			upstream := "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"content\":\"last\"}}]}"
			var content strings.Builder
			err := coalesceSSEEvents(strings.NewReader(upstream), window, func(line string) error {
				var chunk struct {
					Choices []struct {
						Delta struct {
							Content string `json:"content"`
						} `json:"delta"`
					} `json:"choices"`
				}
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
					t.Errorf("decode delta: %v", err)
					return err
				}
				content.WriteString(chunk.Choices[0].Delta.Content)
				return nil
			})
			if !errors.Is(err, errStreamIncomplete) {
				t.Fatalf("error = %v, want errStreamIncomplete", err)
			}
			if got := content.String(); got != "firstlast" {
				t.Fatalf("content = %q, want firstlast", got)
			}
		})
	}
}

func TestCoalesceSSEStopsAtDoneWithoutWaitingForEOF(t *testing.T) {
	for _, window := range []time.Duration{0, time.Hour} {
		t.Run(window.String(), func(t *testing.T) {
			reader, writer := io.Pipe()
			go func() {
				_, _ = io.WriteString(writer, "data: [DONE]\n\n")
			}()
			err := coalesceSSEEvents(reader, window, func(line string) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCoalesceSSEStopsAfterDoneWithoutNewline(t *testing.T) {
	const upstream = "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n\n" +
		"data: [DONE]"
	for _, window := range []time.Duration{0, time.Hour} {
		t.Run(window.String(), func(t *testing.T) {
			var events []string
			if err := coalesceSSEEvents(strings.NewReader(upstream), window, func(line string) error {
				events = append(events, line)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if len(events) == 0 || events[len(events)-1] != "data: [DONE]" {
				t.Fatalf("events = %#v, want final [DONE]", events)
			}
		})
	}
}

func TestCoalesceSSEEofBeforeDoneReturnsError(t *testing.T) {
	for _, window := range []time.Duration{0, time.Hour} {
		t.Run(window.String(), func(t *testing.T) {
			var events []string
			err := coalesceSSEEvents(strings.NewReader(
				"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
			), window, func(line string) error {
				events = append(events, line)
				return nil
			})
			if !errors.Is(err, errStreamIncomplete) {
				t.Fatalf("error = %v, want errStreamIncomplete", err)
			}
			if len(events) != 1 || !strings.Contains(events[0], "partial") {
				t.Fatalf("events = %#v, want partial delta", events)
			}
		})
	}
}

func TestCoalesceSSEEarlyEmitErrorDoesNotHang(t *testing.T) {
	var lines []string
	for i := 0; i < 64; i++ {
		lines = append(lines,
			`data: {"id":"1","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}`,
			``,
		)
	}
	upstream := strings.Join(lines, "\n")
	errWriteFailed := errors.New("write failed")

	n := 0
	done := make(chan error, 1)
	go func() {
		done <- coalesceSSEEvents(strings.NewReader(upstream), 50*time.Millisecond, func(line string) error {
			n++
			if n > 1 {
				return errWriteFailed
			}
			return nil
		})
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected write error")
		}
		if !errors.Is(err, errWriteFailed) {
			t.Fatalf("err=%v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("coalesceSSEEvents hung after emit error")
	}
}
