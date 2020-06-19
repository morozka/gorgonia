package main

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type YoloV3Tiny struct {
	g *gorgonia.ExprGraph

	out *gorgonia.Node

	biases  map[string][]float32
	gammas  map[string][]float32
	means   map[string][]float32
	vars    map[string][]float32
	kernels map[string][]float32
}

type layer struct {
	name    string
	shape   tensor.Shape
	biases  []float32
	gammas  []float32
	means   []float32
	vars    []float32
	kernels []float32
}

func NewYoloV3Tiny(g *gorgonia.ExprGraph, classesNumber, boxesPerCell int, cfgFile, weightsFile string) (*YoloV3Tiny, error) {

	buildingBlocks, err := ParseConfiguration(cfgFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet configuration")
	}
	_ = buildingBlocks

	weightsData, err := ParseWeights(weightsFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet weights")
	}
	_ = weightsData

	lastIdx := 5 // skip first 5 values
	epsilon := float32(0.000001)

	_, _ = lastIdx, epsilon
	return nil, nil
}
