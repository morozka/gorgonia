package gorgonia

import (
	"fmt"
	"hash"
	"image"
	"math"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type yoloOp struct {
	anchors     []float32
	masks       []int
	ignoreTresh float32
	dimensions  int
	numClasses  int
	trainMode   bool
}

func newYoloOp(anchors []float32, masks []int, netSize, numClasses int, ignoreTresh float32) *yoloOp {
	yoloOp := &yoloOp{
		anchors:     anchors,
		dimensions:  netSize,
		numClasses:  numClasses,
		ignoreTresh: ignoreTresh,
		masks:       masks,
		trainMode:   false,
	}
	return yoloOp
}

// YOLOv3 https://arxiv.org/abs/1804.02767
func YOLOv3(input *Node, anchors []float32, masks []int, netSize, numClasses int, ignoreTresh float32, targets ...*Node) (*Node, error) {
	op := newYoloOp(anchors, masks, netSize, numClasses, ignoreTresh)
	return ApplyOp(op, input)
}

func (op *yoloOp) Arity() int {
	return 1
}

func (op *yoloOp) ReturnsPtr() bool { return false }

func (op *yoloOp) CallsExtern() bool { return false }

func (op *yoloOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "YOLO{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) Hashcode() uint32 { return simpleHash(op) }

func (op *yoloOp) String() string {
	return fmt.Sprintf("YOLO{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	shp := inputs[0].(tensor.Shape)
	if len(shp) < 4 {
		return nil, fmt.Errorf("InferShape() for YOLO must contain 4 dimensions")
	}
	s := shp.Clone()
	if op.trainMode {
		return []int{s[0], s[2] * s[3] * len(op.masks), (s[1] - 1) / len(op.masks)}, nil
	}
	return []int{s[0], s[2] * s[3] * len(op.masks), s[1] / len(op.masks)}, nil
}

func (op *yoloOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	o := newTensorType(3, a)
	return hm.NewFnType(t, o)
}

func (op *yoloOp) OverwritesInput() int { return -1 }

func (op *yoloOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, errors.Wrap(err, "Can't check arity for YOLO operation")
	}
	var in tensor.Tensor
	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Can't check YOLO input: expected input has to be a tensor")
	}
	if in.Shape().Dims() != 4 {
		return nil, errors.Errorf("Can't check YOLO input: expected input must have 4 dimensions")
	}
	return in, nil
}

func sigmoidSlice(v tensor.View) error {
	switch v.Dtype() {
	case Float32:
		_, err := v.Apply(_sigmoidf32, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't apply _sigmoidf32 as activation function to YOLO operation")
		}
	case Float64:
		_, err := v.Apply(_sigmoidf64, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't apply _sigmoidf64 as activation function to YOLO operation")
		}
	default:
		return fmt.Errorf("Unsupported numeric type for YOLO sigmoid function. Please use float64 or float32")
	}
	return nil
}

func expSlice(v tensor.View) error {
	switch v.Dtype() {
	case Float32:
		_, err := v.Apply(math32.Exp, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't apply exp32 to YOLO operation")
		}
	case Float64:
		_, err := v.Apply(math.Exp, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't apply exp64 to YOLO operation")
		}
	default:
		return fmt.Errorf("Unsupported numeric type for YOLO for exp function. Please use float64 or float32")
	}
	return nil
}

func (op *yoloOp) Do(inputs ...Value) (retVal Value, err error) {
	// Just inference without backpropagation
	if !op.trainMode {
		inputTensor, err := op.checkInput(inputs...)
		if err != nil {
			return nil, errors.Wrap(err, "Can't check YOLO input")
		}
		batchSize := inputTensor.Shape()[0]
		stride := op.dimensions / inputTensor.Shape()[2]
		gridSize := inputTensor.Shape()[2]
		bboxAttributes := 5 + op.numClasses
		numAnchors := len(op.anchors) / 2
		currentAnchors := []float32{}
		for i := range op.masks {
			if op.masks[i] >= numAnchors {
				return nil, fmt.Errorf("Incorrect mask %v for anchors in YOLO layer", op.masks)
			}
			currentAnchors = append(currentAnchors, op.anchors[i*2], op.anchors[i*2+1])
		}
		return op.evaluateYOLO_f32(inputTensor, batchSize, stride, gridSize, bboxAttributes, len(op.masks), currentAnchors)
	}

	// Training mode
	// @todo

	return nil, nil
}

func (op *yoloOp) evaluateYOLO_f32(input tensor.Tensor, batchSize, stride, grid, bboxAttrs, numAnchors int, currentAnchors []float32) (retVal tensor.Tensor, err error) {

	inputNumericType := input.Dtype()
	if inputNumericType != Float32 {
		return nil, fmt.Errorf("evaluateYOLO_f32() called with input tensor of type %v. Float32 is required", inputNumericType)
	}

	err = input.Reshape(batchSize, bboxAttrs*numAnchors, grid*grid)
	if err != nil {
		return nil, errors.Wrap(err, "Can't make reshape grid^2 for YOLO")
	}

	err = input.T(0, 2, 1)
	if err != nil {
		return nil, errors.Wrap(err, "Can't safely transponse input for YOLO")
	}
	err = input.Transpose()
	if err != nil {
		return nil, errors.Wrap(err, "Can't transponse input for YOLO")
	}
	err = input.Reshape(batchSize, grid*grid*numAnchors, bboxAttrs)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape bbox for YOLO")
	}

	// Activation of x, y, and objects via sigmoid function
	slXY, err := input.Slice(nil, nil, S(0, 2))
	err = sigmoidSlice(slXY)
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate XY")
	}
	slClasses, err := input.Slice(nil, nil, S(4, 5+op.numClasses))
	err = sigmoidSlice(slClasses)
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate classes")
	}

	step := grid * numAnchors
	for i := 0; i < grid; i++ {

		vy, err := input.Slice(nil, S(i*step, i*step+step), S(1))
		if err != nil {
			return nil, errors.Wrap(err, "Can't slice while doing steps for grid")
		}

		_, err = tensor.Add(vy, float32(i), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
		}

		for n := 0; n < numAnchors; n++ {
			anchorsSlice, err := input.Slice(nil, S(i*numAnchors+n, input.Shape()[1], step), S(0))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice anchors while doing steps for grid")
			}
			_, err = tensor.Add(anchorsSlice, float32(i), tensor.UseUnsafe())
			if err != nil {
				return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
			}
		}

	}

	anchors := []float32{}
	for i := 0; i < grid*grid; i++ {
		anchors = append(anchors, currentAnchors...)
	}

	anchorsTensor := tensor.New(tensor.Of(inputNumericType), tensor.WithShape(1, grid*grid*numAnchors, 2))
	for i := range anchors {
		anchorsTensor.Set(i, anchors[i])
	}

	_, err = tensor.Div(anchorsTensor, float32(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Div(...) for float32")
	}

	vhw, err := input.Slice(nil, nil, S(2, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(2,4)")
	}

	_, err = vhw.Apply(math32.Exp, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't apply exp32 to YOLO operation")
	}

	_, err = tensor.Mul(vhw, anchorsTensor, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for anchors")
	}

	vv, err := input.Slice(nil, nil, S(0, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(0,4)")
	}

	_, err = tensor.Mul(vv, float32(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for float32")
	}

	return input, nil
}

func iou_f32(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

func getBestIOU_f32(input, target []float32, numClasses, dims int) [][]float32 {
	ious := make([][]float32, 0)
	imgsize := float32(dims)
	for i := 0; i < len(input); i = i + numClasses + 5 {
		ious = append(ious, []float32{0, -1})
		r1 := rectifyBox_f32(input[i], input[i+1], input[i+2], input[i+3], dims)
		for j := 0; j < len(target); j = j + 5 {
			r2 := rectifyBox_f32(target[j+1]*imgsize, target[j+2]*imgsize, target[j+3]*imgsize, target[j+4]*imgsize, dims)
			curiou := iou_f32(r1, r2)
			if curiou > ious[i/(5+numClasses)][0] {
				ious[i/(5+numClasses)][0] = curiou
				ious[i/(5+numClasses)][1] = float32(j / 5)
			}
		}
	}
	return ious
}

func getBestAnchors_f32(target []float32, anchors []float32, masks []int, dims int, gridSize float32) [][]int {
	bestAnchors := make([][]int, len(target)/5)
	imgsize := float32(dims)
	for j := 0; j < len(target); j = j + 5 {
		targetRect := rectifyBox_f32(0, 0, target[j+3]*imgsize, target[j+4]*imgsize, dims) //not absolutely confident in rectangle sizes
		bestIOU := float32(0.0)
		bestAnchors[j/5] = make([]int, 3)
		for i := 0; i < len(anchors); i = i + 2 {
			anchorRect := rectifyBox_f32(0, 0, anchors[i], anchors[i+1], dims)
			currentIOU := iou_f32(anchorRect, targetRect)
			if currentIOU >= bestIOU {
				bestAnchors[j/5][0] = i
				bestIOU = currentIOU
			}
		}
		bestAnchors[j/5][0] = findIntElement(masks, bestAnchors[j/5][0]/2)
		if bestAnchors[j/5][0] != -1 {
			bestAnchors[j/5][1] = int(target[j+1] * gridSize)
			bestAnchors[j/5][2] = int(target[j+2] * gridSize)
		}
	}
	return bestAnchors
}

func prepareOutputYOLO_f32(input, yoloBoxes, target, anchors []float32, masks []int, numClasses, dims, gridSize int, ignoreTresh float32) []float32 {
	yoloBBoxes := make([]float32, len(yoloBoxes))
	gridSizeF32 := float32(gridSize)
	bestAnchors := getBestAnchors_f32(target, anchors, masks, dims, gridSizeF32)
	bestIous := getBestIOU_f32(yoloBoxes, target, numClasses, dims)
	for i := 0; i < len(yoloBoxes); i = i + (5 + numClasses) {
		if bestIous[i/(5+numClasses)][0] <= ignoreTresh {
			yoloBBoxes[i+4] = bceLoss32(0, yoloBoxes[i+4])
		}
	}
	for i := 0; i < len(bestAnchors); i++ {
		if bestAnchors[i][0] != -1 {
			scale := (2 - target[i*5+3]*target[i*5+4])
			giInt := bestAnchors[i][1]
			gjInt := bestAnchors[i][2]
			gx := invsigm32(target[i*5+1]*gridSizeF32 - float32(giInt))
			gy := invsigm32(target[i*5+2]*gridSizeF32 - float32(gjInt))
			gw := math32.Log(target[i*5+3]/anchors[bestAnchors[i][0]] + 1e-16)
			gh := math32.Log(target[i*5+4]/anchors[bestAnchors[i][0]+1] + 1e-16)
			bboxIdx := gjInt*gridSize*len(masks) + giInt*len(masks) + bestAnchors[i][0]
			yoloBBoxes[bboxIdx] = mseLoss32(gx, input[bboxIdx], scale)
			yoloBBoxes[bboxIdx+1] = mseLoss32(gy, input[bboxIdx+1], scale)
			yoloBBoxes[bboxIdx+2] = mseLoss32(gw, input[bboxIdx+2], scale)
			yoloBBoxes[bboxIdx+3] = mseLoss32(gh, input[bboxIdx+3], scale)
			yoloBBoxes[bboxIdx+4] = bceLoss32(1, yoloBoxes[bboxIdx+4])
			for j := 0; j < numClasses; j++ {
				if j == int(target[i]) {
					yoloBBoxes[bboxIdx+5+j] = bceLoss32(1, yoloBoxes[bboxIdx+4])
				} else {
					yoloBBoxes[bboxIdx+5+j] = bceLoss32(0, yoloBoxes[bboxIdx+4])
				}
			}
		}
	}
	return yoloBBoxes
}

func findIntElement(arr []int, ele int) int {
	for i := range arr {
		if arr[i] == ele {
			return i
		}
	}
	return -1
}

func rectifyBox_f32(x, y, h, w float32, imgSize int) image.Rectangle {
	return image.Rect(maxInt(int(x-w/2), 0), maxInt(int(y-h/2), 0), minInt(int(x+w/2+1), imgSize), minInt(int(y+h/2+1), imgSize))
}

func bceLoss32(target, pred float32) float32 {
	if target == 1.0 {
		return -(math32.Log(pred + 1e-16))
	}
	return -(math32.Log((1.0 - pred) + 1e-16))
}

func mseLoss32(target, pred, scale float32) float32 {
	return math32.Pow(scale*(target-pred), 2) / 2.0
}

func invsigm32(target float32) float32 {
	return -math32.Log(1-target+1e-16) + math32.Log(target+1e-16)
}

func (op *yoloOp) evaluateYOLO_f64(input tensor.Tensor, batchSize, stride, grid, bboxAttrs, numAnchors int, currentAnchors []float64) (retVal tensor.Tensor, err error) {
	inputNumericType := input.Dtype()
	if inputNumericType != Float64 {
		return nil, fmt.Errorf("evaluateYOLO_f64() called with input tensor of type %v. Float64 is required", inputNumericType)
	}
	err = input.Reshape(batchSize, bboxAttrs*numAnchors, grid*grid)
	if err != nil {
		return nil, errors.Wrap(err, "Can't make reshape grid^2 for YOLO")
	}
	err = input.T(0, 2, 1)
	if err != nil {
		return nil, errors.Wrap(err, "Can't safely transponse input for YOLO")
	}
	err = input.Transpose()
	if err != nil {
		return nil, errors.Wrap(err, "Can't transponse input for YOLO")
	}
	err = input.Reshape(batchSize, grid*grid*numAnchors, bboxAttrs)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape bbox for YOLO")
	}

	// Activation of x, y, and objects via sigmoid function
	slXY, err := input.Slice(nil, nil, S(0, 2))
	err = sigmoidSlice(slXY)
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate XY")
	}
	slClasses, err := input.Slice(nil, nil, S(4, 5+op.numClasses))
	err = sigmoidSlice(slClasses)
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate classes")
	}

	step := grid * numAnchors
	for i := 0; i < grid; i++ {
		vy, err := input.Slice(nil, S(i*step, i*step+step), S(1))
		if err != nil {
			return nil, errors.Wrap(err, "Can't slice while doing steps for grid")
		}
		_, err = tensor.Add(vy, float64(i), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float64; (1)")
		}
		for n := 0; n < numAnchors; n++ {
			anchorsSlice, err := input.Slice(nil, S(i*numAnchors+n, input.Shape()[1], step), S(0))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice anchors while doing steps for grid")
			}
			_, err = tensor.Add(anchorsSlice, float64(i), tensor.UseUnsafe())
			if err != nil {
				return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float64; (2)")
			}
		}

	}

	anchors := []float64{}
	for i := 0; i < grid*grid; i++ {
		anchors = append(anchors, currentAnchors...)
	}

	anchorsTensor := tensor.New(tensor.Of(inputNumericType), tensor.WithShape(1, grid*grid*numAnchors, 2))
	for i := range anchors {
		anchorsTensor.Set(i, anchors[i])
	}

	_, err = tensor.Div(anchorsTensor, float64(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Div(...) for float64")
	}

	vhw, err := input.Slice(nil, nil, S(2, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(2,4)")
	}

	_, err = vhw.Apply(math.Exp, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't apply exp64 to YOLO operation")
	}

	_, err = tensor.Mul(vhw, anchorsTensor, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for anchors")
	}

	vv, err := input.Slice(nil, nil, S(0, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(0,4)")
	}

	_, err = tensor.Mul(vv, float64(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for float64")
	}

	return input, nil
}

func iou_f64(r1, r2 image.Rectangle) float64 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float64(interArea) / float64(r1Area+r2Area-interArea)
}

func getBestIOU_f64(input, target []float64, numClasses, dims int) [][]float64 {
	ious := make([][]float64, 0)
	imgsize := float64(dims)
	for i := 0; i < len(input); i = i + numClasses + 5 {
		ious = append(ious, []float64{0, -1})
		r1 := rectifyBox_f64(input[i], input[i+1], input[i+2], input[i+3], dims)
		for j := 0; j < len(target); j = j + 5 {
			r2 := rectifyBox_f64(target[j+1]*imgsize, target[j+2]*imgsize, target[j+3]*imgsize, target[j+4]*imgsize, dims)
			curiou := iou_f64(r1, r2)
			if curiou > ious[i/(5+numClasses)][0] {
				ious[i/(5+numClasses)][0] = curiou
				ious[i/(5+numClasses)][1] = float64(j / 5)
			}
		}
	}
	return ious
}

func getBestAnchors_f64(target []float64, anchors []float64, masks []int, dims int, gridSize float64) [][]int {
	bestAnchors := make([][]int, len(target)/5)
	imgsize := float64(dims)
	for j := 0; j < len(target); j = j + 5 {
		targetRect := rectifyBox_f64(0, 0, target[j+3]*imgsize, target[j+4]*imgsize, dims) //not absolutely confident in rectangle sizes
		bestIOU := float64(0.0)
		bestAnchors[j/5] = make([]int, 3)
		for i := 0; i < len(anchors); i = i + 2 {
			anchorRect := rectifyBox_f64(0, 0, anchors[i], anchors[i+1], dims)
			currentIOU := iou_f64(anchorRect, targetRect)
			if currentIOU >= bestIOU {
				bestAnchors[j/5][0] = i
				bestIOU = currentIOU
			}
		}
		bestAnchors[j/5][0] = findIntElement(masks, bestAnchors[j/5][0]/2)
		if bestAnchors[j/5][0] != -1 {
			bestAnchors[j/5][1] = int(target[j+1] * gridSize)
			bestAnchors[j/5][2] = int(target[j+2] * gridSize)
		}
	}
	return bestAnchors
}

func prepareOutputYOLO_f64(input, yoloBoxes, target, anchors []float64, masks []int, numClasses, dims, gridSize int, ignoreTresh float64) []float64 {
	yoloBBoxes := make([]float64, len(yoloBoxes))
	gridSizeF64 := float64(gridSize)
	bestAnchors := getBestAnchors_f64(target, anchors, masks, dims, gridSizeF64)
	bestIous := getBestIOU_f64(yoloBoxes, target, numClasses, dims)
	for i := 0; i < len(yoloBoxes); i = i + (5 + numClasses) {
		if bestIous[i/(5+numClasses)][0] <= ignoreTresh {
			yoloBBoxes[i+4] = bceLoss64(0, yoloBoxes[i+4])
		}
	}
	for i := 0; i < len(bestAnchors); i++ {
		if bestAnchors[i][0] != -1 {
			scale := (2 - target[i*5+3]*target[i*5+4])
			giInt := bestAnchors[i][1]
			gjInt := bestAnchors[i][2]
			gx := invsigm64(target[i*5+1]*gridSizeF64 - float64(giInt))
			gy := invsigm64(target[i*5+2]*gridSizeF64 - float64(gjInt))
			gw := math.Log(target[i*5+3]/anchors[bestAnchors[i][0]] + 1e-16)
			gh := math.Log(target[i*5+4]/anchors[bestAnchors[i][0]+1] + 1e-16)
			bboxIdx := gjInt*gridSize*len(masks) + giInt*len(masks) + bestAnchors[i][0]
			yoloBBoxes[bboxIdx] = mseLoss64(gx, input[bboxIdx], scale)
			yoloBBoxes[bboxIdx+1] = mseLoss64(gy, input[bboxIdx+1], scale)
			yoloBBoxes[bboxIdx+2] = mseLoss64(gw, input[bboxIdx+2], scale)
			yoloBBoxes[bboxIdx+3] = mseLoss64(gh, input[bboxIdx+3], scale)
			yoloBBoxes[bboxIdx+4] = bceLoss64(1, yoloBoxes[bboxIdx+4])
			for j := 0; j < numClasses; j++ {
				if j == int(target[i]) {
					yoloBBoxes[bboxIdx+5+j] = bceLoss64(1, yoloBoxes[bboxIdx+4])
				} else {
					yoloBBoxes[bboxIdx+5+j] = bceLoss64(0, yoloBoxes[bboxIdx+4])
				}
			}
		}
	}
	return yoloBBoxes
}

func rectifyBox_f64(x, y, h, w float64, imgSize int) image.Rectangle {
	return image.Rect(maxInt(int(x-w/2), 0), maxInt(int(y-h/2), 0), minInt(int(x+w/2+1), imgSize), minInt(int(y+h/2+1), imgSize))
}

func bceLoss64(target, pred float64) float64 {
	if target == 1.0 {
		return -(math.Log(pred + 1e-16))
	}
	return -(math.Log((1.0 - pred) + 1e-16))
}

func mseLoss64(target, pred, scale float64) float64 {
	return math.Pow(scale*(target-pred), 2) / 2.0
}

func invsigm64(target float64) float64 {
	return -math.Log(1-target+1e-16) + math.Log(target+1e-16)
}
