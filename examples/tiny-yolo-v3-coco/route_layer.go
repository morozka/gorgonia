package main

import "fmt"

type routeLayer struct {
	firstLayerIdx  int
	secondLayerIdx int
}

func (l *routeLayer) String() string {
	if l.secondLayerIdx != -1 {
		return fmt.Sprintf("Route layer: Start->%[1]d End->%[2]d", l.firstLayerIdx, l.secondLayerIdx)
	}
	return fmt.Sprintf("Route layer: Start->%[1]d", l.firstLayerIdx)
}
