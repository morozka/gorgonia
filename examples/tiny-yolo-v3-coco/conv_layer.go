package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

type convLayer struct {
	filters        int
	padding        int
	kernelSize     int
	stride         int
	activation     string
	batchNormalize int
	bias           bool

	convNode       *gorgonia.Node
	batchNormNode  *gorgonia.Node
	activationNode *gorgonia.Node
}

func (l *convLayer) String() string {
	return fmt.Sprintf(
		"Convolution layer: Filters->%[1]d Padding->%[2]d Kernel->%[3]dx%[3]d Stride->%[4]d Activation->%[5]s Batch->%[6]d Bias->%[7]t",
		l.filters, l.padding, l.kernelSize, l.stride, l.activation, l.batchNormalize, l.bias,
	)
}

func (l *convLayer) Type() string {
	return "convolutional"
}
