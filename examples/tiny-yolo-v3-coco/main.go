package main

import (
	"fmt"
	"log"
	"time"

	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	width     = 416
	height    = 416
	channels  = 3
	boxes     = 5
	classes   = 80
	leakyCoef = 0.1
	weights   = "./data/yolov3-tiny.weights"
	cfg       = "./data/yolov3-tiny.cfg"
)

func main() {
	g := G.NewGraph()

	input := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(1, channels, width, height), gorgonia.WithName("input"))

	model, err := NewYoloV3Tiny(g, input, classes, boxes, leakyCoef, cfg, weights)
	if err != nil {
		log.Fatalln(err)
	}
	_ = model

	imgf32, err := GetFloat32Image("data/dog_416x416.jpg")
	if err != nil {
		log.Fatalln(err)
	}

	image := tensor.New(tensor.WithShape(1, channels, height, width), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
	err = gorgonia.Let(input, image)
	if err != nil {
	}

	tm := G.NewTapeMachine(g)
	defer tm.Close()
	st := time.Now()
	if err := tm.RunAll(); err != nil {
		log.Fatalf("%+v", err)
	}
	fmt.Println("Feedforwarded in:", time.Since(st))

	tm.Reset()
}
