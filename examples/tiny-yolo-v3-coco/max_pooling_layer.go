package main

import "fmt"

type maxPoolingLayer struct {
	size   int
	stride int
}

func (l *maxPoolingLayer) String() string {
	return fmt.Sprintf("Maxpooling layer: Size->%[1]d Stride->%[2]d", l.size, l.stride)
}

func (l *maxPoolingLayer) Type() string {
	return "maxpool"
}
