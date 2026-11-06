package nvidia

import (
	"encoding/json"
	"log"
	"strings"
	"sync"

	"nvidia-playgroud-go/internal/reasoning"
)

// droppedReasoningOnce keeps the "controls dropped" warning to one line per
// model/control set instead of one per request.
var droppedReasoningOnce sync.Map

// NormalizeRequestBody applies Playground-compatible defaults:
//   - stream: force continuous_usage_stats=false (usage once at end)
//   - thinking: fold common client aliases into the model's reasoning knob
//     (chat_template_kwargs, or a top-level reasoning_effort for kimi)
func NormalizeRequestBody(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}
	model, _ := raw["model"].(string)
	if dropped := reasoning.NormalizeThinking(raw); len(dropped) > 0 {
		reportDroppedReasoning(model, dropped)
	}
	stream, _ := raw["stream"].(bool)
	if !stream {
		return json.Marshal(raw)
	}
	opts, _ := raw["stream_options"].(map[string]any)
	if opts == nil {
		opts = map[string]any{}
		raw["stream_options"] = opts
	}
	if _, ok := opts["include_usage"]; !ok {
		opts["include_usage"] = true
	}
	opts["continuous_usage_stats"] = false
	return json.Marshal(raw)
}

// reportDroppedReasoning surfaces reasoning controls that were removed
// because the model has no known reasoning profile. Without this the request
// looks accepted while the setting is silently ignored.
func reportDroppedReasoning(model string, dropped []string) {
	key := model + "|" + strings.Join(dropped, ",")
	if _, seen := droppedReasoningOnce.LoadOrStore(key, struct{}{}); seen {
		return
	}
	log.Printf("nvidia: model %q has no reasoning profile; dropped unsupported controls %v (upstream rejects them)", model, dropped)
}
