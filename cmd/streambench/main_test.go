package main

import (
	"strings"
	"testing"
	"time"
)

func TestCollectSSEProcessesFinalDeltaWithoutNewline(t *testing.T) {
	now := time.Now()
	stats := collectSSE(strings.NewReader(`data: {"choices":[{"delta":{"content":"last"}}]}`), now, now)

	if len(stats.Events) != 1 {
		t.Fatalf("events = %d, want 1", len(stats.Events))
	}
	if got := stats.Content.String(); got != "last" {
		t.Fatalf("content = %q, want last delta", got)
	}
	if stats.DoneSeen {
		t.Fatal("done = true without a [DONE] event")
	}
}
