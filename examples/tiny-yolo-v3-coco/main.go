package main

import (
	"fmt"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
)

var (
	width     = 416
	height    = 416
	channels  = 3
	boxes     = 3
	classes   = 80
	leakyCoef = 0.1
	weights   = "./data/yolov3-tiny.weights"
	cfg       = "./data/yolov3-tiny.cfg"
)

func main() {
	g := gorgonia.NewGraph()

	input := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(1, channels, width, height), gorgonia.WithName("input"))

	model, err := NewYoloV3Tiny(g, input, classes, boxes, leakyCoef, cfg, weights)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println("Net Loaded!\nLoading input...")
	img, err := GetFloat32Image("./data/dog_416x416.jpg")
	if err != nil {
		panic(err)
	}
	gorgonia.Let(input, tensor.New(tensor.WithShape(1, 3, 416, 416), tensor.WithBacking(img)))

	fmt.Println("Input loaded!\nForwarding net...")
	m := gorgonia.NewTapeMachine(g)
	err = m.RunAll()
	if err != nil {
		panic(err)
	}
	fmt.Println("Net forwarded!")
	fmt.Println(model.out[0].Shape())
	t := model.out[0].Value().(tensor.Tensor)
	att := t.Data().([]float32)

	for i := 0; i < len(att); i += 85 {
		if att[i+4] > 0.5 {
			class := 0
			var buf float32 = 0.0
			for j := 5; j < 85; j++ {
				if att[i+j] > buf {
					buf = att[i+j]
					class = (j - 5) % 80
				}
			}
			fmt.Println(att[i], att[i+1], att[i+2], att[i+3], att[i+4], class, buf)
		}
	}
	t = model.out[1].Value().(tensor.Tensor)
	att = t.Data().([]float32)
	fmt.Println(len(att))
	for i := 0; i < len(att); i += 85 {
		if att[i+4] > 0.5 {
			class := 0
			var buf float32 = 0.0
			for j := 5; j < 85; j++ {
				if att[i+j] > buf {
					buf = att[i+j]
					class = (j - 5) % 80
				}
			}
			fmt.Println(att[i], att[i+1], att[i+2], att[i+3], att[i+4], class, buf)
		}
	}
}
