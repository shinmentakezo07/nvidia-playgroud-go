// Package reasoning maps client reasoning controls onto the per-model
// chat_template_kwargs accepted by NVIDIA playground predict backends.
//
// Models differ in how thinking is switched: some take a reasoning_effort
// level, some a boolean enable_thinking, and minimax takes thinking_mode.
// Sending the wrong knob either 400s or is silently ignored upstream, so the
// mapping lives here and is shared by the library client and the gateway.
package reasoning

import (
	"fmt"
	"strings"

	"nvidia-playgroud-go/internal/models"
)

type kind uint8

const (
	kindEffort kind = iota
	kindToggle
	kindEffortAndToggle
	kindMiniMax
	// kindTopLevelEffort models take reasoning_effort at the request root,
	// not inside chat_template_kwargs (captured from the playground UI).
	kindTopLevelEffort
)

type profile struct {
	kind         kind
	levels       []string
	defaultLevel string
}

// profiles is keyed by registry model ids. Models removed from the anonymous
// catalog in 2026-08 (glm-5.2, gpt-oss-*, qwen3.5-397b, ...) were dropped
// together with their profiles; refresh this map when the catalog changes
// (TestProfilesReferenceRegisteredModels enforces it).
var profiles = map[string]profile{
	"deepseek-ai/deepseek-v4-flash-0731": {kind: kindEffort, levels: []string{"none", "high", "max"}, defaultLevel: "high"},
	"deepseek-ai/deepseek-v4-pro-0813":   {kind: kindEffort, levels: []string{"none", "high", "max"}, defaultLevel: "high"},
	"nvidia/nemotron-3-ultra-550b-a55b":  {kind: kindEffort, levels: []string{"none", "medium", "high"}, defaultLevel: "high"},
	"google/diffusiongemma-26b-a4b-it":   {kind: kindToggle},
	"minimaxai/minimax-m3":               {kind: kindMiniMax},
	// kimi-k3_1-pro takes reasoning_effort at the request root; its public
	// spec exposes low/medium/high/max with default low, and the playground
	// sends it top-level (verified from a browser capture, 2026-11-05).
	"moonshotai/kimi-k3_1-pro": {kind: kindTopLevelEffort, levels: []string{"low", "medium", "high", "max"}, defaultLevel: "low"},
}

var effortRanks = map[string]int{
	"none":    0,
	"minimal": 1,
	"low":     2,
	"medium":  3,
	"high":    4,
	"xhigh":   5,
	"max":     6,
}

// profileFor returns the reasoning profile for model. Models without a
// compiled-in profile fall back to a toggle profile when the registry says
// they accept chat_template_kwargs (Capability.Thinking), so a live-scraped
// model keeps working without a code change.
func profileFor(model string) (profile, bool) {
	if p, ok := profiles[model]; ok {
		return p, true
	}
	if info, err := models.Lookup(model); err == nil && info.Capability != nil && info.Capability.Thinking {
		return profile{kind: kindToggle}, true
	}
	return profile{}, false
}

// ThinkingKwargs returns the chat_template_kwargs that enable or disable
// thinking for model. Models with no profile fall back to enable_thinking,
// the only knob that can be guessed; disabling on such a model returns nil
// because there is nothing to turn off.
func ThinkingKwargs(model string, enable bool) map[string]any {
	p, ok := profileFor(model)
	if !ok {
		if !enable {
			return nil
		}
		return map[string]any{"enable_thinking": true, "clear_thinking": false}
	}

	switch p.kind {
	case kindEffort:
		level, ok := mapEnabled(enable, p)
		if !ok {
			return nil
		}
		return map[string]any{"reasoning_effort": level}
	case kindToggle:
		if enable {
			return map[string]any{"enable_thinking": true, "clear_thinking": false}
		}
		return map[string]any{"enable_thinking": false}
	case kindEffortAndToggle:
		out := map[string]any{"enable_thinking": enable}
		if enable {
			if level, ok := mapEnabled(true, p); ok {
				out["reasoning_effort"] = level
			}
		}
		return out
	case kindMiniMax:
		if enable {
			return map[string]any{"thinking_mode": "enabled"}
		}
		return map[string]any{"thinking_mode": "disabled"}
	case kindTopLevelEffort:
		// The knob lives at the request root; use TopLevelEffort instead.
		return nil
	}
	return nil
}

// ValidateEffort reports whether level is a recognized client effort.
// Use it before calling TopLevelEffort or EffortKwargs, whose false result
// also covers models that do not support the requested native knob.
func ValidateEffort(level string) error {
	return validateEffort(level, "reasoning_effort")
}

// TopLevelEffort reports whether model takes reasoning_effort at the request
// root and returns the value to send. level is an explicit requested effort
// (may be empty); otherwise enable picks the model default or the lowest
// supported level when reasoning cannot be turned off.
func TopLevelEffort(model, level string, enable bool) (string, bool) {
	p, ok := profileFor(model)
	if !ok || p.kind != kindTopLevelEffort {
		return "", false
	}
	if level != "" {
		normalized, ok := normalizeEffort(level)
		if !ok {
			return "", false
		}
		return mapEffort(normalized, p)
	}
	return mapEnabled(enable, p)
}

// EffortKwargs returns chat_template_kwargs for an explicit effort level on
// models that take the knob inside chat_template_kwargs. It reports false for
// models that use a top-level field or have no profile.
func EffortKwargs(model, level string) (map[string]any, bool) {
	p, ok := profileFor(model)
	if !ok || p.kind == kindTopLevelEffort {
		return nil, false
	}
	normalized, ok := normalizeEffort(level)
	if !ok {
		return nil, false
	}
	switch p.kind {
	case kindEffort:
		mapped, ok := mapEffort(normalized, p)
		if !ok {
			return nil, false
		}
		return map[string]any{"reasoning_effort": mapped}, true
	case kindToggle:
		return map[string]any{"enable_thinking": effortEnablesThinking(normalized)}, true
	case kindEffortAndToggle:
		out := map[string]any{"enable_thinking": effortEnablesThinking(normalized)}
		if effortEnablesThinking(normalized) {
			if mapped, ok := mapEffort(normalized, p); ok {
				out["reasoning_effort"] = mapped
			}
		}
		return out, true
	case kindMiniMax:
		return map[string]any{"thinking_mode": miniMaxThinkingMode(normalized)}, true
	}
	return nil, false
}

// NormalizeThinking folds client aliases into the model's native reasoning
// controls. Invalid explicit values and conflicting controls are rejected
// rather than silently ignored or rounded to the model default.
// Aliases for unsupported models are stripped and returned for logging.
func NormalizeThinking(raw map[string]any) ([]string, error) {
	model, _ := raw["model"].(string)
	p, supported := profileFor(model)
	kw, hadKwargs := raw["chat_template_kwargs"].(map[string]any)
	if kw == nil {
		kw = map[string]any{}
	}

	effort, hasEffort := requestedEffort(raw, kw)
	enabled, hasEnabled := requestedThinkingEnabled(raw, kw)
	clearThinking, hasClearThinking := requestedClearThinking(raw, kw)
	if supported {
		if err := validateRequested(raw, kw); err != nil {
			return nil, err
		}
	}
	var dropped []string
	for _, key := range []string{"thinking", "enable_thinking", "clear_thinking", "reasoning_effort"} {
		if _, ok := raw[key]; ok && !supported {
			dropped = append(dropped, key)
		}
		delete(raw, key)
	}

	if supported {
		switch p.kind {
		case kindEffort:
			delete(kw, "enable_thinking")
			if hasEffort {
				if level, ok := mapEffort(effort, p); ok {
					kw["reasoning_effort"] = level
				}
			} else if hasEnabled {
				if level, ok := mapEnabled(enabled, p); ok {
					kw["reasoning_effort"] = level
				}
			}
		case kindToggle:
			delete(kw, "reasoning_effort")
			if hasEnabled {
				kw["enable_thinking"] = enabled
			} else if hasEffort {
				kw["enable_thinking"] = effortEnablesThinking(effort)
			}
		case kindEffortAndToggle:
			if hasEnabled {
				kw["enable_thinking"] = enabled
			} else if hasEffort {
				kw["enable_thinking"] = effortEnablesThinking(effort)
			}
			if hasEffort && (!hasEnabled || enabled) && effortEnablesThinking(effort) {
				if level, ok := mapEffort(effort, p); ok {
					kw["reasoning_effort"] = level
				} else {
					delete(kw, "reasoning_effort")
				}
			} else {
				delete(kw, "reasoning_effort")
			}
		case kindMiniMax:
			delete(kw, "enable_thinking")
			delete(kw, "reasoning_effort")
			if hasEffort {
				kw["thinking_mode"] = miniMaxThinkingMode(effort)
			} else if hasEnabled {
				kw["thinking_mode"] = "disabled"
				if enabled {
					kw["thinking_mode"] = "enabled"
				}
			}
		case kindTopLevelEffort:
			// Keep the knob at the request root. The alias loop above removed
			// it, so re-add the mapped value here.
			delete(kw, "enable_thinking")
			delete(kw, "reasoning_effort")
			var level string
			var ok bool
			switch {
			case hasEffort:
				level, ok = mapEffort(effort, p)
			case hasEnabled:
				level, ok = mapEnabled(enabled, p)
			}
			if ok {
				raw["reasoning_effort"] = level
			}
		}
		if hasClearThinking && (p.kind == kindToggle || p.kind == kindEffortAndToggle) {
			kw["clear_thinking"] = clearThinking
		}
	}

	if len(kw) > 0 || hadKwargs {
		raw["chat_template_kwargs"] = kw
	} else {
		delete(raw, "chat_template_kwargs")
	}
	return dropped, nil
}

// validateRequested rejects unknown effort and conflicting controls.
// It checks all supplied aliases, including those losing precedence.
func validateRequested(raw, kw map[string]any) error {
	for _, field := range []struct {
		value any
		name  string
	}{
		{kw["reasoning_effort"], "chat_template_kwargs.reasoning_effort"},
		{raw["reasoning_effort"], "reasoning_effort"},
	} {
		if err := validateEffort(field.value, field.name); err != nil {
			return err
		}
	}
	thinking, _ := raw["thinking"].(map[string]any)
	if err := validateEffort(thinking["reasoning_effort"], "thinking.reasoning_effort"); err != nil {
		return err
	}
	if value, ok := thinking["type"]; ok {
		typ, ok := value.(string)
		if !ok {
			return fmt.Errorf("thinking.type must be a string")
		}
		switch strings.ToLower(strings.TrimSpace(typ)) {
		case "enabled", "enable", "on", "disabled", "disable", "off", "auto", "adaptive":
		default:
			return fmt.Errorf("invalid thinking.type %q", typ)
		}
	}
	for _, field := range []struct {
		value any
		name  string
	}{
		{kw["enable_thinking"], "chat_template_kwargs.enable_thinking"},
		{raw["enable_thinking"], "enable_thinking"},
	} {
		if field.value == nil {
			continue
		}
		enabled, ok := field.value.(bool)
		if !ok {
			return fmt.Errorf("%s must be a boolean", field.name)
		}
		if !enabled && hasEnabledEffort(raw, kw) {
			return fmt.Errorf("disabled thinking conflicts with explicit reasoning_effort")
		}
	}
	if typ, ok := thinking["type"].(string); ok && isDisabledThinkingType(typ) && hasEnabledEffort(raw, kw) {
		return fmt.Errorf("disabled thinking conflicts with explicit reasoning_effort")
	}
	return nil
}

func hasEnabledEffort(raw, kw map[string]any) bool {
	thinking, _ := raw["thinking"].(map[string]any)
	for _, value := range []any{raw["reasoning_effort"], kw["reasoning_effort"], thinking["reasoning_effort"]} {
		if effort, ok := value.(string); ok {
			normalized, valid := normalizeEffort(effort)
			return valid && effortEnablesThinking(normalized)
		}
	}
	return false
}

func isDisabledThinkingType(typ string) bool {
	switch strings.ToLower(strings.TrimSpace(typ)) {
	case "disabled", "disable", "off":
		return true
	}
	return false
}

func validateEffort(value any, name string) error {
	if value == nil {
		return nil
	}
	effort, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", name)
	}
	if _, ok := normalizeEffort(effort); !ok {
		return fmt.Errorf("invalid reasoning_effort %q", effort)
	}
	return nil
}

func requestedEffort(raw, kw map[string]any) (string, bool) {
	for _, value := range []any{kw["reasoning_effort"], raw["reasoning_effort"]} {
		if effort, ok := value.(string); ok {
			if normalized, ok := normalizeEffort(effort); ok {
				return normalized, true
			}
		}
	}
	if thinking, ok := raw["thinking"].(map[string]any); ok {
		if effort, ok := thinking["reasoning_effort"].(string); ok {
			if normalized, ok := normalizeEffort(effort); ok {
				return normalized, true
			}
		}
	}
	return "", false
}

func requestedThinkingEnabled(raw, kw map[string]any) (bool, bool) {
	if enabled, ok := kw["enable_thinking"].(bool); ok {
		return enabled, true
	}
	if thinking, ok := raw["thinking"].(map[string]any); ok {
		if typ, ok := thinking["type"].(string); ok {
			switch strings.ToLower(strings.TrimSpace(typ)) {
			case "enabled", "enable", "on", "auto", "adaptive":
				return true, true
			case "disabled", "disable", "off":
				return false, true
			}
		}
	}
	enabled, ok := raw["enable_thinking"].(bool)
	return enabled, ok
}

func requestedClearThinking(raw, kw map[string]any) (any, bool) {
	if value, ok := kw["clear_thinking"]; ok {
		return value, true
	}
	if thinking, ok := raw["thinking"].(map[string]any); ok {
		if value, ok := thinking["clear_thinking"]; ok {
			return value, true
		}
	}
	value, ok := raw["clear_thinking"]
	return value, ok
}

// normalizeEffort lowercases and validates a requested effort. Unknown
// values are ignored rather than rounded to the model default: a typo must
// not silently turn thinking on (or up) behind the caller's back.
func normalizeEffort(effort string) (string, bool) {
	normalized := strings.ToLower(strings.TrimSpace(effort))
	_, ok := effortRanks[normalized]
	return normalized, ok
}

// mapEffort rounds effort up to the nearest level the model supports.
func mapEffort(effort string, p profile) (string, bool) {
	if len(p.levels) == 0 {
		return "", false
	}
	requestedRank := effortRanks[effort]
	for _, level := range p.levels {
		if effortRanks[level] >= requestedRank {
			return level, true
		}
	}
	return p.levels[len(p.levels)-1], true
}

func mapEnabled(enabled bool, p profile) (string, bool) {
	if !enabled {
		return mapEffort("none", p)
	}
	if len(p.levels) == 0 {
		return "", false
	}
	if p.defaultLevel != "" {
		return p.defaultLevel, true
	}
	return p.levels[len(p.levels)-1], true
}

func effortEnablesThinking(effort string) bool {
	return effort != "none"
}

func miniMaxThinkingMode(effort string) string {
	switch effort {
	case "none":
		return "disabled"
	case "minimal", "low", "medium":
		return "adaptive"
	default:
		return "enabled"
	}
}
