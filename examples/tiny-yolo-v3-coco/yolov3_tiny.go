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
type YoloV3Tiny struct {
	g *gorgonia.ExprGraph

	out *gorgonia.Node

	biases  map[string][]float32
	gammas  map[string][]float32
	means   map[string][]float32
	vars    map[string][]float32
	kernels map[string][]float32
}

// type layer struct {
// 	name    string
// 	shape   tensor.Shape
// 	biases  []float32
// 	gammas  []float32
// 	means   []float32
// 	vars    []float32
// 	kernels []float32
// }

// NewYoloV3Tiny Create new tiny YOLO v3
func NewYoloV3Tiny(g *gorgonia.ExprGraph, input *gorgonia.Node, classesNumber, boxesPerCell int, cfgFile, weightsFile string) (*YoloV3Tiny, error) {

	buildingBlocks, err := ParseConfiguration(cfgFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet configuration")
	}

	weightsData, err := ParseWeights(weightsFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet weights")
	}

	_ = weightsData

	fmt.Println("Loading network...")
	layers := []*layerN{}
	prevFilters := 3
	blocks := buildingBlocks[1:]
	for i := range blocks {
		layerType, ok := blocks[i]["type"]
		if ok {
			switch layerType {
			case "convolutional":
				filters := 0
				padding := 0
				kernelSize := 0
				stride := 0
				batchNormalize := 0
				bias := false
				activation := "activation"
				activation, ok := blocks[i]["activation"]
				if !ok {
					fmt.Printf("No field 'activation' for convolution layer")
					continue
				}
				batchNormalizeStr, ok := blocks[i]["batch_normalize"]
				batchNormalize, err := strconv.Atoi(batchNormalizeStr)
				if !ok || err != nil {
					batchNormalize = 0
					bias = true
				}
				filtersStr, ok := blocks[i]["filters"]
				filters, err = strconv.Atoi(filtersStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'filters' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				paddingStr, ok := blocks[i]["pad"]
				padding, err = strconv.Atoi(paddingStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'pad' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				kernelSizeStr, ok := blocks[i]["size"]
				kernelSize, err = strconv.Atoi(kernelSizeStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'size' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				pad := 0
				if padding != 0 {
					pad = (kernelSize - 1) / 2
				}
				strideStr, ok := blocks[i]["stride"]
				stride, err = strconv.Atoi(strideStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for convolution layer: %s\n", err.Error())
					continue
				}

				var l layerN = &convLayer{
					filters:        filters,
					padding:        pad,
					kernelSize:     kernelSize,
					stride:         stride,
					activation:     activation,
					batchNormalize: batchNormalize,
					bias:           bias,
				}
				layers = append(layers, &l)
				fmt.Println(l)

				// conv node
				convNode := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(filters, prevFilters, kernelSize, kernelSize), gorgonia.WithName(fmt.Sprintf("conv_%d", i)))
				_ = convNode
				if batchNormalize != 0 {
					// @todo batch norm node
				}

				if activation == "leaky" {
					// @todo leaky node
				}

				prevFilters = filters
				break
			case "upsample":
				scale := 0
				scaleStr, ok := blocks[i]["stride"]
				scale, err = strconv.Atoi(scaleStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for upsampling layer: %s\n", err.Error())
					continue
				}

				var l layerN = &upsampleLayer{
					scale: scale,
				}
				layers = append(layers, &l)
				fmt.Println(l)

				// @todo upsample node

				break
			case "route":
				routeLayersStr, ok := blocks[i]["layers"]
				if !ok {
					fmt.Printf("No field 'layers' for route layer")
					continue
				}
				layersSplit := strings.Split(routeLayersStr, ",")
				if len(layersSplit) < 1 {
					fmt.Printf("Something wrong with route layer. Check if it has one array item atleast")
					continue
				}
				for l := range layersSplit {
					layersSplit[l] = strings.TrimSpace(layersSplit[l])
				}
				start := 0
				end := 0
				start, err := strconv.Atoi(layersSplit[0])
				if err != nil {
					fmt.Printf("Each first element of 'layers' parameter for route layer should be an integer: %s\n", err.Error())
					continue
				}
				if len(layersSplit) > 1 {
					end, err = strconv.Atoi(layersSplit[1])
					if err != nil {
						fmt.Printf("Each second element of 'layers' parameter for route layer should be an integer: %s\n", err.Error())
						continue
					}
				}

				if start > 0 {
					start = start - i
				}
				if end > 0 {
					end = end - i
				}

				l := routeLayer{
					firstLayerIdx:  i + start,
					secondLayerIdx: -1,
				}
				if end < 0 {
					l.secondLayerIdx = i + end
				}

				var ll layerN = &l
				layers = append(layers, &ll)
				fmt.Println(ll)

				// @todo upsample node
				// @todo evaluate 'prevFilters'

				break
			case "yolo":
				maskStr, ok := blocks[i]["mask"]
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
				anchorsStr, ok := blocks[i]["anchors"]
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

				// @todo detection node? or just flow?

				break
			case "maxpool":
				sizeStr, ok := blocks[i]["size"]
				if !ok {
					fmt.Printf("No field 'size' for maxpooling layer")
					continue
				}
				size, err := strconv.Atoi(sizeStr)
				if err != nil {
					fmt.Printf("'size' parameter for maxpooling layer should be an integer: %s\n", err.Error())
					continue
				}
				strideStr, ok := blocks[i]["stride"]
				if !ok {
					fmt.Printf("No field 'stride' for maxpooling layer")
					continue
				}
				stride, err := strconv.Atoi(strideStr)
				if err != nil {
					fmt.Printf("'size' parameter for maxpooling layer should be an integer: %s\n", err.Error())
					continue
				}
				var l layerN = &maxPoolingLayer{
					size:   size,
					stride: stride,
				}
				layers = append(layers, &l)
				fmt.Println(l)
				break
			default:
				break
			}
		}
	}

	fmt.Println("Loading weights...")
	lastIdx := 5 // skip first 5 values
	epsilon := float32(0.000001)
	for i := range layers {
		l := *layers[i]
		layerType := l.Type()
		// Ignore everything except convolutional layers
		if layerType == "convolutional" {
			layer := l.(*convLayer)
			if layer.batchNormalize > 0 {
				// fmt.Printf("module # %d %s\n", i, layerType)
				//@todo load weights/biases and etc.
			} else {
				//@todo load weights/biases and etc.
			}
		}
	}

	_, _ = lastIdx, epsilon
	return nil, nil
}
