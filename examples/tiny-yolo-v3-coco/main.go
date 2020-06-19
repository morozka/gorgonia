package main

import (
	"log"

	G "gorgonia.org/gorgonia"
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

	model, err := NewYoloV3Tiny(g, classes, boxes, cfg, weights)
	if err != nil {
		log.Fatalln(err)
	}

	_ = model
}
