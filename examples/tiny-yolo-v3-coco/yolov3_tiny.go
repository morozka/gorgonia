package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// YoloV3Tiny YoloV3 tiny architecture
/*type YoloV3Tiny struct {
	g *gorgonia.ExprGraph

	out *gorgonia.Node

	biases  map[string][]float32
	gammas  map[string][]float32
	means   map[string][]float32
	vars    map[string][]float32
	kernels map[string][]float32
}*/

var layeri int //layer counter

func convolutional(input *gorgonia.Node, block map[string]string) (l layer, err error) {
	g := input.Graph()
	filters, padding, size, stride, batchNormalize := 0, 0, 0, 0, 0
	var batchout, gamma, beta, scale, bias, output *gorgonia.Node
	activation, ok := block["activation"]
	if !ok {
		return nil, errors.New("No field 'activation' for convolution layer")
	}
	batchNormalizeStr, ok := block["batch_normalize"]
	batchNormalize, err = strconv.Atoi(batchNormalizeStr)
	if !ok || err != nil {
		batchNormalize = 0
	}
	filtersStr, ok := block["filters"]
	filters, err = strconv.Atoi(filtersStr)
	if !ok || err != nil {
		return nil, errors.Wrap(err, "Wrong or empty 'filters' parameter for convolution layer")
	}
	paddingStr, ok := block["pad"]
	padding, err = strconv.Atoi(paddingStr)
	if !ok || err != nil {
		return nil, errors.Wrap(err, "Wrong or empty 'pad' parameter for convolution layer")
	}
	kernelSizeStr, ok := block["size"]
	size, err = strconv.Atoi(kernelSizeStr)
	if !ok || err != nil {
		return nil, errors.Wrap(err, "Wrong or empty 'size' parameter for convolution layer")
	}
	pad := 0
	if padding != 0 {
		pad = (size - 1) / 2
	}
	strideStr, ok := block["stride"]
	stride, err = strconv.Atoi(strideStr)
	if !ok || err != nil {
		return nil, errors.Wrap(err, "Wrong or empty 'stride' parameter for convolution layer")
	}
	kernel := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(1, filters, size, size), gorgonia.WithName(fmt.Sprint("kernel_%v", i)))
	convout := gorgonia.Must(gorgonia.Conv2d(input, kernel, kernel.Shape(), []int{pad, pad}, []int{stride, stride}, []int{1, 1}))
	if batchNormalize != 0 {
		scale = gorgonia.NewTensor(g, gorgonia.Float64, 1, gorgonia.WithShape(filters), gorgonia.WithName(fmt.Sprintf("scale_%v", i)))
		bias = gorgonia.NewTensor(g, gorgonia.Float64, 1, gorgonia.WithShape(filters), gorgonia.WithName(fmt.Sprintf("bias_%v", i)))
		if batchout, gamma, beta, _, err = gorgonia.BatchNorm(convout, scale, bias, 0.1, 10e-5); err != nil {
			return nil, err
		}
	}
	if activation == "leaky" {
		// leakyNode := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(convNode.Shape()...), gorgonia.WithName(fmt.Sprintf("leaky_%d", i)))
		actout, err := gorgonia.LeakyRelu(batchout, 0.1)
		if err != nil {
			return nil, err
		}
		output = actout
	}
	return &convlayer{
		bias:   bias,
		scale:  scale,
		kernel: kernel,
		gamma:  gamma,
		beta:   beta,
		output: output,
	}, nil
}

func maxpool(input *gorgonia.Node, block map[string]string) (l layer, err error) {
	sizeStr, ok := block["size"]
	if !ok {
		return nil, errors.Errorf("No field 'size' for maxpooling layer")
	}
	size, err := strconv.Atoi(sizeStr)
	if err != nil {
		return nil, errors.Wrap(err, "'size' parameter for maxpooling layer should be an integer")
	}
	strideStr, ok := block["stride"]
	if !ok {
		return nil, errors.Errorf("No field 'stride' for maxpooling layer")
	}
	stride, err := strconv.Atoi(strideStr)
	if err != nil {
		return nil, errors.Wrap(err, "'size' parameter for maxpooling layer should be an integer")
	}
	retVal := gorgonia.Must(gorgonia.MaxPool2D(input, tensor.Shape{size, size}, []int{1, 1}, []int{stride, stride}))
	return &maxpoollayer{
		output: retVal,
	}, nil
}

func upsample(input *gorgonia.Node, block map[string]string) (l layer, err error) {
	scaleStr, ok := block["stride"]
	scale, err := strconv.Atoi(scaleStr)
	if !ok || err != nil {
		return nil, errors.Wrap(err, "Wrong or empty 'stride' parameter for upsampling layer")
	}
	retVal := gorgonia.Must(gorgonia.Upsample2D(input, scale))
	return &upsamplelayer{
		output: retVal,
	}, nil
}

func route(input *gorgonia.Node, block map[string]string, nn []layer) (l layer, err error) {
	routeLayersStr, ok := block["layers"]
	if !ok {
		return nil, errors.Errorf("No field 'layers' for route layer")
	}
	layersSplit := strings.Split(routeLayersStr, ",")
	if len(layersSplit) < 1 {
		return nil, errors.Errorf("Something wrong with route layer. Check if it has one array item atleast")
	}
	for l := range layersSplit {
		layersSplit[l] = strings.TrimSpace(layersSplit[l])
	}

	nodes := make([]*gorgonia.Node, len(layersSplit)+1)
	nodes[0] = input
	for i := range layersSplit {
		ind, err := strconv.Atoi(layersSplit[i])
		if err != nil {
			return nil, errors.Wrap(err, "Each first element of 'layers' parameter for route layer should be an integer")
		}
		if ind == 0 {
			return nil, errors.Errorf("Route cannot concat it's output")
		}

		if ind < 0 {
			nodes[i+1] = nn[layeri-ind].Output()
		} else {
			nodes[i+1] = nn[ind].Output()
		}

	}
	retVal := gorgonia.Must(gorgonia.Concat(0, nodes...))

	return &routelayer{output: retVal}, nil
}

// NewYoloV3Tiny Create new tiny YOLO v3
func NewYoloV3Tiny(g *gorgonia.ExprGraph, input *gorgonia.Node, classesNumber, boxesPerCell int, leakyCoef float64, cfgFile, weightsFile string) (*YoloV3Tiny, error) {
	buildingBlocks, err := ParseConfiguration(cfgFile)
	fmt.Println(buildingBlocks)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet configuration")
	}

	weightsData, err := ParseWeights(weightsFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet weights")
	}
	var nn []layer

	current := input
	fmt.Println("Loading network...")
	layers := []*layerN{}
	outputFilters := []int{}
	prevFilters := 3
	blocks := buildingBlocks[1:]
	for layeri = range blocks {
		filtersIdx := 0
		layerType, ok := blocks[i]["type"]
		if ok {
			switch layerType {
			case "convolutional":
				conv, ok := convolutional(current, blocks[layeri])
				if ok != nil {
					panic(errors.Wrap(ok, "Error in covolution layer!"))
				}
				nn = append(nn, conv)
				current = conv.Output()
				break
			case "maxpool":
				mxpl, err := maxpool(current, blocks[layeri])
				if err != nil {
					panic(err)
				}
				nn = append(nn, mxpl)
				current = mxpl.Output()
				break
			case "upsample":
				ups, err := upsample(current, blocks[layeri])
				if err != nil {
					panic(err)
				}
				nn = append(nn, ups)
				current = ups.Output()
				break
			case "route":
				rout, err := upsample(current, blocks[layeri])
				if err != nil {
					panic(err)
				}
				nn = append(nn, rout)
				current = rout.Output()
				break
			case "yolo":
				/*maskStr, ok := blocks[layeri]["mask"]
				if !ok {
					fmt.Printf("No field 'mask' for YOLO layer")
					continue
				}
				maskSplit := strings.Split(maskStr, ",")
				if len(maskSplit) < 1 {
					fmt.Printf("Something wrong with yolo layer. Check if it has one item in 'mask' array atleast")
					continue
				}
				masks := make([]int, len(maskSplit))
				for l := range maskSplit {
					maskSplit[l] = strings.TrimSpace(maskSplit[l])
					masks[l], err = strconv.Atoi(maskSplit[l])
					if err != nil {
						fmt.Printf("Each element of 'mask' parameter for yolo layer should be an integer: %s\n", err.Error())
					}
				}
				anchorsStr, ok := blocks[дфнкi]["anchors"]
				if !ok {
					fmt.Printf("No field 'anchors' for YOLO layer")
					continue
				}
				anchorsSplit := strings.Split(anchorsStr, ",")
				if len(anchorsSplit) < 1 {
					fmt.Printf("Something wrong with yolo layer. Check if it has one item in 'anchors' array atleast")
					continue
				}
				if len(anchorsSplit)%2 != 0 {
					fmt.Printf("Number of elemnts in 'anchors' parameter for yolo layer should be divided exactly by 2 (even number)")
					continue
				}
				anchors := make([]int, len(anchorsSplit))
				for l := range anchorsSplit {
					anchorsSplit[l] = strings.TrimSpace(anchorsSplit[l])
					anchors[l], err = strconv.Atoi(anchorsSplit[l])
					if err != nil {
						fmt.Printf("Each element of 'anchors' parameter for yolo layer should be an integer: %s\n", err.Error())
					}
				}
				anchorsPairs := [][2]int{}
				for a := 0; a < len(anchors); a += 2 {
					anchorsPairs = append(anchorsPairs, [2]int{anchors[a], anchors[a+1]})
				}
				selectedAnchors := [][2]int{}
				for m := range masks {
					selectedAnchors = append(selectedAnchors, anchorsPairs[masks[m]])
				}

				var l layerN = &yoloLayer{
					masks:   masks,
					anchors: selectedAnchors,
				}
				layers = append(layers, &l)
				fmt.Println(l)
				filtersIdx = prevFilters*/
				break

			default:
				fmt.Println("Impossible")
				break
			}
		}
		prevFilters = filtersIdx
		outputFilters = append(outputFilters, filtersIdx)
	}

	fmt.Println("Loading weights...")
	//lastIdx := 5 // skip first 5 values
	//epsilon := float32(0.000001)

	/*ptr := 0
	for i := range layers {
		l := *layers[i]
		layerType := l.Type()
		// Ignore everything except convolutional layers
		if layerType == "convolutional" {
			layer := l.(*convLayer)
			if layer.batchNormalize > 0 && layer.batchNormNode != nil {
				biasesNum := layer.batchNormNode.Shape()[0]

				biases := weightsData[ptr : ptr+biasesNum]
				_ = biases
				ptr += biasesNum

				weights := weightsData[ptr : ptr+biasesNum]
				_ = weights
				ptr += biasesNum

				means := weightsData[ptr : ptr+biasesNum]
				_ = means
				ptr += biasesNum

				vars := weightsData[ptr : ptr+biasesNum]
				_ = vars
				ptr += biasesNum

				//@todo load weights/biases and etc.
			} else {
				biasesNum := layer.convNode.Shape()[0]
				convBiases := weightsData[ptr : ptr+biasesNum]
				_ = convBiases
				ptr += biasesNum
				//@todo load weights/biases and etc.
			}

			weightsNumel := layer.convNode.Shape().TotalSize()

			ptr += weightsNumel
		}
	}

	_, _ = lastIdx, epsilon*/
	return nil, nil
}
