package main

import "gorgonia.org/gorgonia"

type layer interface {
	Output() *gorgonia.Node
	// LoadableParams() []*gorgonia.Node
	// GammaBeta() (*gorgonia.Node, *gorgonia.Node)
	Type() layertype
}

type layertype int

const (
	CONV = iota
	MAXPOOL
	UPSAMPLE
	ROUTE
	YOLO
)

func (d layertype) String() string {
	return [...]string{"convolutional", "maxpool", "upsample", "route", "yolo"}[d]
}

type convlayer struct {
	bias, scale, kernel *gorgonia.Node

	output, gamma, beta *gorgonia.Node
}

func (c convlayer) Output() *gorgonia.Node {
	return c.output
}

func (c convlayer) LoadableParams() []*gorgonia.Node {
	return []*gorgonia.Node{c.kernel, c.bias, c.scale}
}

func (c convlayer) GammaBeta() (*gorgonia.Node, *gorgonia.Node) {
	return c.gamma, c.beta
}

func (c convlayer) Type() layertype {
	return 0
}

type maxpoollayer struct {
	output *gorgonia.Node
}

func (m maxpoollayer) Output() *gorgonia.Node {
	return m.output
}

func (m maxpoollayer) Type() layertype {
	return 1
}

type upsamplelayer struct {
	output *gorgonia.Node
}

func (l upsamplelayer) Output() *gorgonia.Node {
	return l.output
}

func (l upsamplelayer) Type() layertype {
	return 2
}

type routelayer struct {
	output *gorgonia.Node
}

func (l routelayer) Output() *gorgonia.Node {
	return l.output
}

func (l routelayer) Type() layertype {
	return 3
}

type yololayer struct {
	output *gorgonia.Node
}

func (l yololayer) Output() *gorgonia.Node {
	return l.output
}

func (l yololayer) Type() layertype {
	return 4
}
