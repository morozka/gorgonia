package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type convLayer struct {
	filters        int
	padding        int
	kernelSize     int
	stride         int
	activation     string
	batchNormalize int
	bias           bool

	layerIndex int

	convNode       *gorgonia.Node
	batchNormNode  *gorgonia.Node
	activationNode *gorgonia.Node

	batchOut, gamma, beta *gorgonia.Node
	batchNormOperation    *gorgonia.BatchNormOp
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

func (l *convLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {

	convOut, err := gorgonia.Conv2d(input[0], l.convNode, tensor.Shape{l.kernelSize, l.kernelSize}, []int{l.padding, l.padding}, []int{l.stride, l.stride}, []int{1, 1})
	if err != nil {
		return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare convolution operation")
	}

	if l.batchNormalize > 0 {
		scale := gorgonia.NewTensor(g, gorgonia.Float32, 4, gorgonia.WithShape(convOut.Shape()...), gorgonia.WithName(fmt.Sprintf("scale_%d", l.layerIndex)))
		bias := gorgonia.NewTensor(g, gorgonia.Float32, 4, gorgonia.WithShape(convOut.Shape()...), gorgonia.WithName(fmt.Sprintf("bias_%d", l.layerIndex)))
		l.batchOut, l.gamma, l.beta, l.batchNormOperation, err = gorgonia.BatchNorm(convOut, scale, bias, 0.1, 10e-5)
		if err != nil {
			return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare batch normalization operation")
		}
	} else {
		l.batchOut = convOut
	}

	if l.activation == "leaky" {
		activationOut, err := gorgonia.LeakyRelu(l.batchOut, 0.1)
		if err != nil {
			return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare activation operation")
		}
		return activationOut, nil
	}
	return convOut, nil
}
