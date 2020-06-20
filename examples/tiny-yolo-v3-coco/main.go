package main

import (
	"log"

	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	width    = 416
	height   = 416
	channels = 3
	boxes    = 5
	classes  = 80
	weights  = "./data/yolov3-tiny.weights"
	cfg      = "./data/yolov3-tiny.cfg"
)

func main() {
	g := G.NewGraph()

	input := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(1, channels, width, height), gorgonia.WithName("input"))

	model, err := NewYoloV3Tiny(g, input, classes, boxes, cfg, weights)
	if err != nil {
		log.Fatalln(err)
	}

	_ = model
}
