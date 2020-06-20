package main

import "fmt"

type upsampleLayer struct {
	scale int
}

func (l *upsampleLayer) String() string {
	return fmt.Sprintf("Upsample layer: Scale->%[1]d", l.scale)
}

func (l *upsampleLayer) Type() string {
	return "upsample"
}
