package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// YoloV3Tiny YoloV3 tiny architecture
type YoloV3Tiny struct {
	g *gorgonia.ExprGraph

	out []*gorgonia.Node
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
func NewYoloV3Tiny(g *gorgonia.ExprGraph, input *gorgonia.Node, classesNumber, boxesPerCell int, leakyCoef float64, cfgFile, weightsFile string) (*YoloV3Tiny, error) {
	inputS := input.Shape()
	if len(inputS) < 4 {
		return nil, fmt.Errorf("Input for YOLOv3 should contain infromation about 4 dimensions")
	}

	buildingBlocks, err := ParseConfiguration(cfgFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet configuration")
	}

	weightsData, err := ParseWeights(weightsFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet weights")
	}

	fmt.Println("Loading network...")
	layers := []*layerN{}
	outputFilters := []int{}
	prevFilters := 3

	networkNodes := []*gorgonia.Node{}

	model := &YoloV3Tiny{
		g: g,
	}

	blocks := buildingBlocks[1:]
	for i := range blocks {
		block := blocks[i]
		filtersIdx := 0
		layerType, ok := block["type"]
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
				activation, ok := block["activation"]
				if !ok {
					fmt.Printf("No field 'activation' for convolution layer")
					continue
				}
				batchNormalizeStr, ok := block["batch_normalize"]
				batchNormalize, err := strconv.Atoi(batchNormalizeStr)
				if !ok || err != nil {
					batchNormalize = 0
					bias = true
				}
				filtersStr, ok := block["filters"]
				filters, err = strconv.Atoi(filtersStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'filters' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				paddingStr, ok := block["pad"]
				padding, err = strconv.Atoi(paddingStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'pad' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				kernelSizeStr, ok := block["size"]
				kernelSize, err = strconv.Atoi(kernelSizeStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'size' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				pad := 0
				if padding != 0 {
					pad = (kernelSize - 1) / 2
				}
				strideStr, ok := block["stride"]
				stride, err = strconv.Atoi(strideStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for convolution layer: %s\n", err.Error())
					continue
				}

				ll := &convLayer{
					filters:        filters,
					padding:        pad,
					kernelSize:     kernelSize,
					stride:         stride,
					activation:     activation,
					batchNormalize: batchNormalize,
					bias:           bias,
				}

				var l layerN = ll
				convBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Convolutional block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, convBlock)
				input = convBlock

				layers = append(layers, &l)
				fmt.Println(i, l)

				filtersIdx = filters
				break
			case "upsample":
				scale := 0
				scaleStr, ok := block["stride"]
				scale, err = strconv.Atoi(scaleStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for upsampling layer: %s\n", err.Error())
					continue
				}

				var l layerN = &upsampleLayer{
					scale: scale,
				}

				upsampleBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Upsample block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, upsampleBlock)
				input = upsampleBlock

				layers = append(layers, &l)
				fmt.Println(i, l)

				// @todo upsample node

				filtersIdx = prevFilters
				break
			case "route":
				routeLayersStr, ok := block["layers"]
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
					filtersIdx = outputFilters[i+start] + outputFilters[i+end]
				} else {
					filtersIdx = outputFilters[i+start]
				}

				var ll layerN = &l

				routeBlock, err := l.ToNode(g, networkNodes...)
				if err != nil {
					fmt.Printf("\tError preparing Route block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, routeBlock)
				input = routeBlock

				layers = append(layers, &ll)
				fmt.Println(i, ll)

				// @todo upsample node
				// @todo evaluate 'prevFilters'

				break
			case "yolo":
				maskStr, ok := block["mask"]
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
				anchorsStr, ok := block["anchors"]
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
				tresh := block["ignore_thresh"]
				ignoretr, err := strconv.ParseFloat(tresh, 32)
				if err != nil {
					fmt.Printf("Something wrong with yolo layer. Check treshold param")
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

				var l layerN = &yoloLayer{
					mask:       masks,
					anchors:    anchors,
					inputSize:  inputS[2],
					classesNum: classesNumber,
					treshold:   float32(ignoretr),
				}
				yoloBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing YOLO block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, yoloBlock)
				input = yoloBlock

				layers = append(layers, &l)
				fmt.Println(i, l)

				filtersIdx = prevFilters
				model.out = append(model.out, input)
				fmt.Println("YOLO:", yoloBlock)
				break
			case "maxpool":
				sizeStr, ok := block["size"]
				if !ok {
					fmt.Printf("No field 'size' for maxpooling layer")
					continue
				}
				size, err := strconv.Atoi(sizeStr)
				if err != nil {
					fmt.Printf("'size' parameter for maxpooling layer should be an integer: %s\n", err.Error())
					continue
				}
				strideStr, ok := block["stride"]
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

				maxpoolingBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Max-Pooling block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, maxpoolingBlock)
				input = maxpoolingBlock

				layers = append(layers, &l)
				fmt.Println(i, l)

				filtersIdx = prevFilters
				break
			default:
				fmt.Println("Impossible")
				break
			}
		}
		prevFilters = filtersIdx
		outputFilters = append(outputFilters, filtersIdx)
	}

	fmt.Println("\nLoading weights...\n")
	// lastIdx := 5 // skip first 5 values
	epsilon := float32(0.0000001)
	ptr := 5
	for i := range layers {
		l := *layers[i]
		layerType := l.Type()
		// Ignore everything except convolutional layers
		if layerType == "convolutional" {
			layer := l.(*convLayer)
			var beta, gamma, means, vars, biases []float32
			fmt.Println("Loading weights: ", layer)
			if layer.batchNormalize > 0 && layer.outShape.TotalSize() > 0 {
				biasesNum := layer.bnOut.Shape()[1]
				isize := layer.bnOut.Shape()[2:4].TotalSize()

				beta = weightsData[ptr : ptr+biasesNum]

				ptr += biasesNum
				if err != nil {
					panic(err)
				}

				gamma = weightsData[ptr : ptr+biasesNum]
				gammat, err := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum, 1), tensor.WithBacking(gamma)).Repeat(0, isize)
				if err != nil {
					panic(err)
				}
				err = gammat.Reshape(layer.bnOut.Shape()...)
				if err != nil {
					panic(err)
				}
				err = gorgonia.Let(layer.gamma, gammat)
				ptr += biasesNum
				if err != nil {
					panic(err)
				}

				means = weightsData[ptr : ptr+biasesNum]
				err = gorgonia.Let(layer.means, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum), tensor.WithBacking(means)))
				ptr += biasesNum
				if err != nil {
					panic(err)
				}

				vars = weightsData[ptr : ptr+biasesNum]

				err = gorgonia.Let(layer.vars, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum), tensor.WithBacking(vars)))
				ptr += biasesNum
				if err != nil {
					panic(err)
				}
				for i := 0; i < layer.kernels.Shape()[0]; i++ {
					scale := gamma[i] / float32(math.Sqrt(float64(vars[i]+epsilon)))

					beta[i] = (beta[i] - means[i]*scale)
				}
				betat, err := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum, 1), tensor.WithBacking(beta)).Repeat(0, isize)
				if err != nil {
					panic(err)
				}
				err = betat.Reshape(layer.bnOut.Shape()...)
				if err != nil {
					panic(err)
				}
				err = gorgonia.Let(layer.beta, betat)
			} else {
				kernelNum := layer.kernels.Shape()[0]
				biases = weightsData[ptr : ptr+kernelNum]
				ptr += kernelNum
			}
			weightsNumel := layer.kernels.Shape().TotalSize()
			kernelW := weightsData[ptr : ptr+weightsNumel]

			isize := layer.kernels.Shape()[1:4].TotalSize()
			if layer.batchNormalize > 0 && layer.outShape.TotalSize() > 0 {
				for i := 0; i < layer.kernels.Shape()[0]; i++ {
					scale := gamma[i] / float32(math.Sqrt(float64(vars[i]+epsilon)))
					for j := 0; j < isize; j++ {
						kernelW[isize*i+j] = kernelW[isize*i+j] * scale
					}
				}
			}
			if layer.bias {
				isize := layer.convOut.Shape()[2:4].TotalSize()
				biasT, err := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(layer.kernels.Shape()[0]), tensor.WithBacking(biases)).Repeat(1, isize)
				if err != nil {
					panic(err)
				}
				fmt.Println(layer.kernels.Shape()[0], biasT.Shape())
				err = biasT.Reshape(layer.convOut.Shape()...)
				if err != nil {
					panic(err)
				}
				err = gorgonia.Let(layer.biases, biasT)
			}
			err = gorgonia.Let(layer.kernels, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(layer.kernels.Shape()...), tensor.WithBacking(kernelW)))
			if err != nil {
				panic(err)
			}
			ptr += weightsNumel
		}
	}
	return model, nil
}
