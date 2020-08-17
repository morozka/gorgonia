package main

import (
	"flag"
	"fmt"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	imgWidth       = 416
	imgHeight      = 416
	channels       = 3
	boxes          = 3
	leakyCoef      = 0.1
	cocoClasses    = []string{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}
	scoreThreshold = float32(0.8)
	iouThreshold   = float32(0.3)
)

func main() {
	modeStr := flag.String("mode", "detector", "Choose the mode: detector/training")
	imagePathStr := flag.String("image", "./data/dog_416x416.jpg", "Path to image file for 'detector' mode")
	weightsPath := flag.String("weights", "./data/yolov3-tiny.weights", "Path to weights file")
	cfgPathStr := flag.String("cfg", "./data/yolov3-tiny.cfg", "Path to net configuration file")
	trainingFolder := flag.String("train", "./data", "Path to folder with labeled data")
	flag.Parse()

	switch *modeStr {
	case "detector":
		g := G.NewGraph()

		input := G.NewTensor(g, tensor.Float32, 4, G.WithShape(1, channels, imgWidth, imgHeight), G.WithName("input"))
		model, err := NewYoloV3Tiny(g, input, len(cocoClasses), boxes, leakyCoef, *cfgPathStr, *weightsPath)
		if err != nil {
			fmt.Printf("Can't prepare YOLOv3 network due the error: %s\n", err.Error())
			return
		}
		model.Print()

		imgf32, err := GetFloat32Image(*imagePathStr, imgHeight, imgWidth)
		if err != nil {
			fmt.Printf("Can't read []float32 from image due the error: %s\n", err.Error())
			return
		}
		image := tensor.New(tensor.WithShape(1, channels, imgHeight, imgWidth), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
		err = G.Let(input, image)
		if err != nil {
			fmt.Printf("Can't let input = []float32 due the error: %s\n", err.Error())
			return
		}

		tm := G.NewTapeMachine(g)
		defer tm.Close()

		// Feedforward
		st := time.Now()
		if err := tm.RunAll(); err != nil {
			fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
			return
		}
		fmt.Println("Feedforwarded in:", time.Since(st))

		// Postprocessing
		st = time.Now()
		dets, err := model.ProcessOutput(cocoClasses, scoreThreshold, iouThreshold)
		if err != nil {
			fmt.Printf("Can't do postprocessing due error: %s", err.Error())
			return
		}
		fmt.Println("Postprocessed in:", time.Since(st))

		fmt.Println("Detections:")
		for i := range dets {
			fmt.Println(dets[i])
		}

		tm.Reset()
		return
	case "trainig":
		// @todo
		_ = trainingFolder
		return
	default:
		// Can't reach this code because of default value for modeStr.
		return
	}
}
