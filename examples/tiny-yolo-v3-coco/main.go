package main

import (
	"fmt"
	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
	"strings"
	"time"
)

var (
	classesCoco = "person bicycle car motorbike aeroplane bus train truck boat trafficlight firehydrant stopsign parkingmeter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sportsball kite baseballbat baseballglove skateboard surfboard tennisracket bottle wineglass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hotdog pizza donut cake chair sofa pottedplant bed diningtable toilet tvmonitor laptop mouse remote keyboard cellphone microwave oven toaster sink refrigerator book clock vase scissors teddybear	hairdrier toothbrush"
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
	//weights = "./data/model/yolov2-tiny.weights"
	//cfg     = "./data/model/yolov2-tiny.cfg"
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
		fmt.Printf("Can't read []float32 from image due the error: %s\n", err.Error())
		return
	}

	image := tensor.New(tensor.WithShape(1, channels, height, width), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
	err = gorgonia.Let(input, image)
	if err != nil {
		fmt.Printf("Can't let input = []float32 due the error: %s\n", err.Error())
		return
	}

	tm := G.NewTapeMachine(g)
	defer tm.Close()
	st := time.Now()
	if err := tm.RunAll(); err != nil {
		fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
		return
	}
	fmt.Println("Feedforwarded in:", time.Since(st))

	//fmt.Println(model.out[0].Value())
	if cfg == "./data/yolov3-tiny.cfg" {
		classesCocoArr := strings.Split(classesCoco, " ")
		t := model.out[0].Value().(tensor.Tensor)
		att := t.Data().([]float32)

		for i := 0; i < len(att); i += 85 {
			if att[i+4] > 0.6 {
				class := 0
				var buf float32
				for j := 5; j < 85; j++ {
					if att[i+j] > buf {
						buf = att[i+j]
						class = (j - 5) % 80
					}
				}
				if buf*att[i+4] > 0.5 {
					fmt.Println(att[i], att[i+1], att[i+2], att[i+3], att[i+4], classesCocoArr[class], buf)
				}
			}
		}
		t = model.out[1].Value().(tensor.Tensor)
		att = t.Data().([]float32)

		for i := 0; i < len(att); i += 85 {
			if att[i+4] > 0.6 {
				class := 0
				var buf float32
				for j := 5; j < 85; j++ {
					if att[i+j] > buf {
						buf = att[i+j]
						class = (j - 5) % 80
					}
				}
				if buf*att[i+4] > 0.5 {
					fmt.Println(att[i], att[i+1], att[i+2], att[i+3], att[i+4], classesCocoArr[class], buf)
				}
			}
		}
	}
	if cfg == "./data/model/yolov2-tiny.cfg" {
		dets, err := ProcessOutput(model.out[0])
		fmt.Println(err)
		for i, j := range dets {
			fmt.Println(i, j)
		}
	}

	tm.Reset()
}
