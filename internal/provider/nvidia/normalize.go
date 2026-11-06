package nvidia

import (
	"encoding/json"
	"fmt"

	"nvidia-playgroud-go/internal/reasoning"
)

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
	dropped, err := reasoning.NormalizeThinking(raw)
	if err != nil {
		return nil, err
	}
	if len(dropped) > 0 {
		return nil, fmt.Errorf("model %q has no reasoning profile; unsupported reasoning controls: %v", model, dropped)
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
