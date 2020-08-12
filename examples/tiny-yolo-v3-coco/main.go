package main

import (
	"errors"
	"fmt"
	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var (
	width       = 416
	height      = 416
	channels    = 3
	boxes       = 5
	classes     = 80
	leakyCoef   = 0.1
	weights     = "./data/yolov3-tiny.weights"
	cfg         = "./data/yolov3-tiny.cfg"
	classesCoco = "person bicycle car motorbike aeroplane bus train truck boat trafficlight firehydrant stopsign parkingmeter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sportsball kite baseballbat baseballglove skateboard surfboard tennisracket bottle wineglass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hotdog pizza donut cake chair sofa pottedplant bed diningtable toilet tvmonitor laptop mouse remote keyboard cellphone microwave oven toaster sink refrigerator book clock vase scissors teddybear	hairdrier toothbrush"
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

	cost := gorgonia.Must(gorgonia.Sum(model.out[0], 0, 1, 2))
	_, err = gorgonia.Grad(cost, model.learningNodes...)
	if err != nil {
		panic(err)
	}
	prog, locMap, _ := gorgonia.Compile(g)
	tm := G.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(model.learningNodes...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(0.00001))
	_ = solver
	defer tm.Close()
	st := time.Now()
	ff, err := os.Create("log.txt")
	defer ff.Close()
	if err != nil {
		panic(err)
	}
	targets, err := prepareTrain32("./data")
	if err != nil {
		panic(err)
	}
	model.SetTarget(targets[0])
	model.ActivateTrainingMode()
	for i := 0; i < 4000; i++ {
		gorgonia.Let(input, image)
		if err := tm.RunAll(); err != nil {
			fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
			return
		}
		if i == 15 {
			solver = gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(0.000001))
		}
		if i == 150 {
			solver = gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(0.0000001))
		}
		fmt.Println(cost.Value())
		fmt.Println("===========================================================")
		err = solver.Step(gorgonia.NodesToValueGrads(model.learningNodes))
		if err != nil {

			fmt.Println(err)
		}

		ff.Write([]byte(fmt.Sprint(cost.Value(), "\n")))
		tm.Reset()
	}
	model.DeactivateTrainingMode()
	gorgonia.Let(input, image)
	if err := tm.RunAll(); err != nil {
		fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
		return
	}
	fmt.Println("Feedforwarded in:", time.Since(st))
	return
	if cfg == "./data/yolov3-tiny.cfg" {
		classesCocoArr := strings.Split(classesCoco, " ")
		t := model.out[0].Value().(tensor.Tensor)
		fmt.Println(t)
		att := t.Data().([]float32)

		fmt.Println("16th layer:")
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
				if buf*att[i+4] > 0.6 {
					fmt.Println(att[i], att[i+1], att[i+2], att[i+3], att[i+4], classesCocoArr[class], buf)
				}
			}
		}
		t = model.out[1].Value().(tensor.Tensor)
		att = t.Data().([]float32)

		fmt.Println("last layer:")
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
				if buf*att[i+4] > 0.6 {
					fmt.Println(att[i], att[i+1], att[i+2], att[i+3], att[i+4], classesCocoArr[class], buf)
				}
			}
		}
	}

	// fmt.Println(model.out.Value())
	tm.Reset()
}
func prepareTrain32(pathToDir string) ([][]float32, error) {
	files, err := ioutil.ReadDir(pathToDir)
	if err != nil {
		return [][]float32{}, err
	}
	farr := [][]float32{}
	numTrainFiles := 0
	for _, file := range files {
		cfarr := []float32{}
		if file.IsDir() || filepath.Ext(file.Name()) != ".txt" {
			continue
		}
		numTrainFiles++
		f, err := ioutil.ReadFile(pathToDir + "/" + file.Name())
		if err != nil {
			return [][]float32{}, err
		}
		str := string(f)
		fmt.Println(str)
		str = strings.ReplaceAll(str, "\n", " ")
		arr := strings.Split(str, " ")
		for i := 0; i < len(arr); i++ {
			if arr[i] == "" {
				continue
			}
			if s, err := strconv.ParseFloat(arr[i], 32); err == nil {
				if float32(s) < 0 {
					return [][]float32{}, errors.New("incorrect training data")
				}
				cfarr = append(cfarr, float32(s))
			} else {
				return [][]float32{}, err
			}
		}
		farr = append(farr, cfarr)
	}
	return farr, nil
}
