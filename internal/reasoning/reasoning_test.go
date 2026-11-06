package reasoning

import (
	"strings"
	"testing"

	"nvidia-playgroud-go/internal/models"
)

func TestProfilesReferenceRegisteredModels(t *testing.T) {
	for model := range profiles {
		if _, err := models.Lookup(model); err != nil {
			t.Errorf("reasoning profile references unsupported model %q: %v", model, err)
		}
	}
}

func TestThinkingKwargsPerModel(t *testing.T) {
	cases := []struct {
		model  string
		enable bool
		want   map[string]any
	}{
		{
			model:  "deepseek-ai/deepseek-v4-pro-0813",
			enable: true,
			want:   map[string]any{"reasoning_effort": "high"},
		},
		{
			model:  "deepseek-ai/deepseek-v4-pro-0813",
			enable: false,
			want:   map[string]any{"reasoning_effort": "none"},
		},
		{
			model:  "google/diffusiongemma-26b-a4b-it",
			enable: true,
			want:   map[string]any{"enable_thinking": true, "clear_thinking": false},
		},
		{
			model:  "google/diffusiongemma-26b-a4b-it",
			enable: false,
			want:   map[string]any{"enable_thinking": false},
		},
		{
			model:  "minimaxai/minimax-m3",
			enable: true,
			want:   map[string]any{"thinking_mode": "enabled"},
		},
		{
			model:  "minimaxai/minimax-m3",
			enable: false,
			want:   map[string]any{"thinking_mode": "disabled"},
		},
	}
	for _, tc := range cases {
		got := ThinkingKwargs(tc.model, tc.enable)
		if len(got) != len(tc.want) {
			t.Fatalf("%s enable=%v: got %#v want %#v", tc.model, tc.enable, got, tc.want)
		}
		for k, v := range tc.want {
			if got[k] != v {
				t.Fatalf("%s enable=%v: got %#v want %#v", tc.model, tc.enable, got, tc.want)
			}
		}
	}
}

func TestThinkingKwargsUnknownModelFallsBackToToggle(t *testing.T) {
	got := ThinkingKwargs("acme/not-in-registry", true)
	if got["enable_thinking"] != true {
		t.Fatalf("unknown model should fall back to enable_thinking, got %#v", got)
	}
	if got := ThinkingKwargs("acme/not-in-registry", false); got != nil {
		t.Fatalf("disabling on an unknown model should produce no kwargs, got %#v", got)
	}
}

func TestThinkingKwargsTopLevelModelIsNil(t *testing.T) {
	if got := ThinkingKwargs("moonshotai/kimi-k3_1-pro", true); got != nil {
		t.Fatalf("kimi takes reasoning_effort at the request root, got kwargs %#v", got)
	}
}

func TestTopLevelEffort(t *testing.T) {
	cases := []struct {
		level  string
		enable bool
		want   string
	}{
		{"", true, "low"},
		{"", false, "low"}, // kimi has no "none": lowest supported level
		{"low", true, "low"},
		{"medium", true, "medium"},
		{"high", true, "high"},
		{"max", true, "max"},
		{"hgih", true, ""}, // invalid level cannot select a profile
	}
	for _, tc := range cases {
		got, ok := TopLevelEffort("moonshotai/kimi-k3_1-pro", tc.level, tc.enable)
		if tc.want == "" {
			if ok {
				t.Fatalf("invalid level %q must not select a profile", tc.level)
			}
			continue
		}
		if !ok || got != tc.want {
			t.Fatalf("level=%q enable=%v: got %q ok=%v, want %q", tc.level, tc.enable, got, ok, tc.want)
		}
	}
	if _, ok := TopLevelEffort("deepseek-ai/deepseek-v4-pro-0813", "high", true); ok {
		t.Fatal("deepseek does not use a top-level effort field")
	}
}

func TestEffortKwargs(t *testing.T) {
	kw, ok := EffortKwargs("deepseek-ai/deepseek-v4-pro-0813", "max")
	if !ok || kw["reasoning_effort"] != "max" {
		t.Fatalf("deepseek effort kwargs = %#v ok=%v", kw, ok)
	}
	kw, ok = EffortKwargs("google/diffusiongemma-26b-a4b-it", "none")
	if !ok || kw["enable_thinking"] != false {
		t.Fatalf("diffusiongemma effort kwargs = %#v ok=%v", kw, ok)
	}
	if _, ok := EffortKwargs("moonshotai/kimi-k3_1-pro", "high"); ok {
		t.Fatal("kimi should use TopLevelEffort, not kwargs")
	}
}

func TestNormalizeThinkingKimiKeepsTopLevelEffort(t *testing.T) {
	raw := map[string]any{
		"model":            "moonshotai/kimi-k3_1-pro",
		"reasoning_effort": "high",
	}
	if dropped, err := NormalizeThinking(raw); err != nil || len(dropped) != 0 {
		t.Fatalf("kimi effort should be supported, dropped=%v", dropped)
	}
	if raw["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %#v, want top-level high", raw["reasoning_effort"])
	}
	if _, ok := raw["chat_template_kwargs"]; ok {
		t.Fatalf("kimi must not receive chat_template_kwargs: %#v", raw)
	}
}

func TestNormalizeThinkingKimiAliasMapsToEffort(t *testing.T) {
	raw := map[string]any{
		"model":    "moonshotai/kimi-k3_1-pro",
		"thinking": map[string]any{"type": "enabled"},
	}
	if dropped, err := NormalizeThinking(raw); err != nil || len(dropped) != 0 {
		t.Fatalf("dropped = %v", dropped)
	}
	if raw["reasoning_effort"] != "low" {
		t.Fatalf("reasoning_effort = %#v, want the model default low", raw["reasoning_effort"])
	}
	if _, ok := raw["thinking"]; ok {
		t.Fatal("thinking alias leaked")
	}
}

func TestNormalizeThinkingKimiKwargMovesToTopLevel(t *testing.T) {
	raw := map[string]any{
		"model": "moonshotai/kimi-k3_1-pro",
		"chat_template_kwargs": map[string]any{
			"reasoning_effort": "max",
			"enable_thinking":  true,
		},
	}
	if dropped, err := NormalizeThinking(raw); err != nil || len(dropped) != 0 {
		t.Fatalf("dropped = %v", dropped)
	}
	if raw["reasoning_effort"] != "max" {
		t.Fatalf("reasoning_effort = %#v, want top-level max", raw["reasoning_effort"])
	}
	if kw, ok := raw["chat_template_kwargs"].(map[string]any); ok && len(kw) != 0 {
		t.Fatalf("consumed knobs must not stay in kwargs: %#v", kw)
	}
}

func TestNormalizeThinkingReportsDroppedControls(t *testing.T) {
	raw := map[string]any{
		"model":            "poolside/laguna-xs-2.1",
		"reasoning_effort": "high",
		"thinking":         map[string]any{"type": "enabled"},
	}
	dropped, err := NormalizeThinking(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(dropped) != 2 || dropped[0] != "thinking" || dropped[1] != "reasoning_effort" {
		t.Fatalf("dropped = %v", dropped)
	}
	if _, ok := raw["reasoning_effort"]; ok {
		t.Fatal("reasoning_effort alias leaked")
	}
	if _, ok := raw["thinking"]; ok {
		t.Fatal("thinking alias leaked")
	}
}

func TestNormalizeThinkingSupportedModelDoesNotReport(t *testing.T) {
	raw := map[string]any{
		"model":            "deepseek-ai/deepseek-v4-pro-0813",
		"reasoning_effort": "max",
	}
	if dropped, err := NormalizeThinking(raw); err != nil || len(dropped) != 0 {
		t.Fatalf("dropped = %v", dropped)
	}
	kw := raw["chat_template_kwargs"].(map[string]any)
	if kw["reasoning_effort"] != "max" {
		t.Fatalf("kwargs = %#v", kw)
	}
}

func TestNormalizeThinkingRejectsEffectiveConflictsAcrossAliases(t *testing.T) {
	for _, raw := range []map[string]any{
		{
			"model": "moonshotai/kimi-k3_1-pro",
			"chat_template_kwargs": map[string]any{
				"enable_thinking":  false,
				"reasoning_effort": "high",
			},
		},
		{
			"model": "moonshotai/kimi-k3_1-pro",
			"thinking": map[string]any{
				"type":             "disabled",
				"reasoning_effort": "max",
			},
		},
	} {
		if _, err := NormalizeThinking(raw); err == nil || !strings.Contains(err.Error(), "conflicts") {
			t.Errorf("raw = %#v: error = %v, want conflict", raw, err)
		}
	}
}

func TestNormalizeThinkingHonorsAliasPrecedence(t *testing.T) {
	cases := []struct {
		name string
		raw  map[string]any
		want string
	}{
		{
			name: "top-level effort wins over nested disabled switch",
			raw:  map[string]any{"model": "moonshotai/kimi-k3_1-pro", "reasoning_effort": "high", "thinking": map[string]any{"type": "disabled"}},
			want: "high",
		},
		{
			name: "top-level high beats same-source disable switch",
			raw:  map[string]any{"model": "moonshotai/kimi-k3_1-pro", "reasoning_effort": "high", "enable_thinking": false},
			want: "high",
		},
		{
			name: "top-level none beats nested high",
			raw:  map[string]any{"model": "moonshotai/kimi-k3_1-pro", "reasoning_effort": "none", "thinking": map[string]any{"type": "enabled", "reasoning_effort": "high"}},
			want: "low",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NormalizeThinking(tc.raw); err != nil {
				t.Fatal(err)
			}
			if got := tc.raw["reasoning_effort"]; got != tc.want {
				t.Fatalf("reasoning_effort = %#v, want %s: %#v", got, tc.want, tc.raw)
			}
		})
	}
}

func TestMapEffortEmptyLevelsDoesNotPanic(t *testing.T) {
	if _, ok := mapEffort("high", profile{kind: kindEffort}); ok {
		t.Fatal("mapEffort with no levels should report no mapping")
	}
	if _, ok := mapEnabled(true, profile{kind: kindEffort}); ok {
		t.Fatal("mapEnabled with no levels should report no mapping")
	}
}
