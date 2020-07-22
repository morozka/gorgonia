package gorgonia

import (
	"fmt"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
	"hash"
	"image"
	"math"
)

type yoloOp struct {
	anchors     []float64
	mask        []int
	ignoreTresh float64
	inpDim      int
	numClasses  int
	train       bool
}

func newYoloOp(anchors []float64, mask []int, imheight, numclasses int, ignoreTresh float64, train bool) *yoloOp {
	yoloOp := &yoloOp{
		anchors:     anchors,
		inpDim:      imheight,
		numClasses:  numclasses,
		ignoreTresh: ignoreTresh,
		mask:        mask,
		train:       train,
	}
	return yoloOp
}

//YoloDetector yolov3 output layer
func YoloDetector(x *Node, anchors []float64, mask []int, imheight, numclasses int, ignoreTresh float64, target ...*Node) (*Node, error) {
	if len(target) > 0 {
		sx, _ := Slice(x, S(0), nil, nil, nil)
		st, _ := Slice(target[0], S(0), nil, nil, nil)
		rx := Must(Concat(0, sx, st))
		rx = Must(Reshape(rx, []int{1, rx.Shape()[0], rx.Shape()[1], rx.Shape()[2]}))
		op := newYoloOp(anchors, mask, imheight, numclasses, ignoreTresh, true)
		retVal, err := ApplyOp(op, rx)
		return retVal, err
	}
	op := newYoloOp(anchors, mask, imheight, numclasses, ignoreTresh, false)
	retVal, err := ApplyOp(op, x)
	return retVal, err
}

func (op *yoloOp) Arity() int {
	return 1
}

func (op *yoloOp) ReturnsPtr() bool { return false }

func (op *yoloOp) CallsExtern() bool { return false }

func (op *yoloOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Yolo{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) Hashcode() uint32 { return simpleHash(op) }

func (op *yoloOp) String() string {
	return fmt.Sprintf("Yolo{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	if op.train {
		return []int{s[0], s[2] * s[3] * len(op.mask), (s[1] - 1) / len(op.mask)}, nil
	}
	return []int{s[0], s[2] * s[3] * len(op.mask), s[1] / len(op.mask)}, nil
}

func (op *yoloOp) Type() hm.Type {

	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	o := newTensorType(3, a)
	return hm.NewFnType(t, o)

}
func (op *yoloOp) OverwritesInput() int { return -1 }

func (op *yoloOp) checkInput(inputs ...Value) (tensor.Tensor, error) {

	//Delete?
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var in tensor.Tensor
	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	if in.Shape().Dims() != 4 {
		return nil, errors.Errorf("Expected input to have 4 dimensions")
	}
	return in, nil
}

func sigmSlice(v tensor.View, old error) {
	fmt.Println(v.Shape())
	if old != nil {
		panic(old)
	}
	switch v.Dtype() {
	case Float32:
		if _, err := v.Apply(_sigmoidf32, tensor.UseUnsafe()); err != nil {
			panic(err)
		}
	case Float64:
		if _, err := v.Apply(_sigmoidf64, tensor.UseUnsafe()); err != nil {
			panic(err)
		}
	default:
		panic("Unsupportable type for Yolo")
	}
}

func expSlice(v tensor.View, old error) {
	if old != nil {
		panic(old)
	}
	switch v.Dtype() {
	case Float32:
		if _, err := v.Apply(func(x float32) float32 {
			return float32(math.Exp(float64(x)))
		}, tensor.UseUnsafe()); err != nil {
			panic(err)
		}
	case Float64:
		if _, err := v.Apply(math.Exp, tensor.UseUnsafe()); err != nil {
			panic(err)
		}
	default:
		panic("Unsupportable type for Yolo")
	}
}

func convertToFloat32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i := range in {
		out[i] = float32(in[i])
	}
	return out
}

func (op *yoloOp) Do(inputs ...Value) (retVal Value, err error) {
	if !op.train {
		in, _ := op.checkInput(inputs...)
		batch := in.Shape()[0]
		stride := int(op.inpDim / in.Shape()[2])
		grid := in.Shape()[2]
		bboxAttrs := 5 + op.numClasses
		numAnchors := len(op.mask)
		currentAnchors := []float64{}
		for _, i := range op.mask {
			if i >= (len(op.anchors) / 2) {
				return nil, errors.New("Incorrect mask for anchors on yolo layer with name" + fmt.Sprint(op.mask))
			}
			currentAnchors = append(currentAnchors, op.anchors[i*2], op.anchors[i*2+1])
		}
		fmt.Println(currentAnchors, op.anchors, in.Shape()[2], int(op.inpDim/stride))
		return op.yoloDoer(in, batch, stride, grid, bboxAttrs, numAnchors, currentAnchors)
	}
	in, _ := op.checkInput(inputs...)
	inv, _ := in.Slice(nil, S(0, in.Shape()[1]-1), nil, nil)
	numTargets, _ := in.At(0, in.Shape()[1]-1, 0, 0)
	batch := in.Shape()[0]
	stride := int(op.inpDim / in.Shape()[2])
	grid := in.Shape()[2]
	bboxAttrs := 5 + op.numClasses
	numAnchors := len(op.mask)
	currentAnchors := []float64{}
	for _, i := range op.mask {
		if i >= (len(op.anchors) / 2) {
			return nil, errors.New("Incorrect mask for anchors on yolo layer with name" + fmt.Sprint(op.mask))
		}
		currentAnchors = append(currentAnchors, op.anchors[i*2], op.anchors[i*2+1])
	}
	var targets []float32
	switch in.Dtype() {
	case Float32:
		lt := int(numTargets.(float32))
		targets = make([]float32, lt, lt)
		for i := 1; i <= lt; i++ {
			buf, _ := in.At(0, in.Shape()[1]-1, 0+i/grid, 0+i%grid)
			targets[i-1] = buf.(float32)
		}
		break
	case Float64:
		lt := int(numTargets.(float64))
		targets = make([]float32, lt, lt)
		for i := 1; i <= lt; i++ {
			buf, _ := in.At(0, in.Shape()[1]-1, 0+i/grid, 0+i%grid)
			targets[i-1] = float32(buf.(float64))
		}
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	in = inv.Materialize()
	/*
		repeates := numAnchors / (in.Shape()[1] / (5 + op.numClasses))
		inr, err := tensor.Concat(1, in)
		for i := 1; i < repeates; i++ {
			inr, err = tensor.Concat(1, inr, in)
		}*/
	outyolo, _ := op.yoloDoer(in, batch, stride, grid, bboxAttrs, numAnchors, currentAnchors)
	yboxes32 := make([]float32, 0)
	switch outyolo.Dtype() {
	case Float32:
		outyolo.Reshape(outyolo.Shape()[0] * outyolo.Shape()[1] * outyolo.Shape()[2])
		for i := 0; i < outyolo.Shape()[0]; i++ {
			buf, _ := outyolo.At(i)
			yboxes32 = append(yboxes32, buf.(float32))
		}
		break
	case Float64:
		outyolo.Reshape(outyolo.Shape()[0] * outyolo.Shape()[1] * outyolo.Shape()[2])
		for i := 0; i < outyolo.Shape()[0]; i++ {
			buf, _ := outyolo.At(i)
			yboxes32 = append(yboxes32, float32(buf.(float64)))
		}
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	return outyolo, nil

}
func (op *yoloOp) yoloDoer(in tensor.Tensor, batch, stride, grid, bboxAttrs, numAnchors int, currentAnchors []float64) (retVal tensor.Tensor, err error) {
	in.Reshape(batch, bboxAttrs*numAnchors, grid*grid)

	in.T(0, 2, 1)
	in.Transpose()
	in.Reshape(batch, grid*grid*numAnchors, bboxAttrs)

	// Activation of x, y, and objectness params
	sigmSlice(in.Slice(nil, nil, S(0, 2)))
	sigmSlice(in.Slice(nil, nil, S(4, 5+op.numClasses)))

	step := grid * numAnchors

	for ind := 0; ind < grid; ind++ {
		//View with the same Y coordinate (row)
		vy, err := in.Slice(nil, S(ind*step, ind*step+step), S(1))
		if err != nil {
			fmt.Println("1")
			panic(err)
		}
		switch in.Dtype() {
		case Float32:
			_, err = tensor.Add(vy, float32(ind), tensor.UseUnsafe())
			break
		case Float64:
			_, err = tensor.Add(vy, float64(ind), tensor.UseUnsafe())
			break
		default:
			panic("Unsupportable type for Yolo")
		}
		if err != nil {
			fmt.Println("2")
			panic(err)
		}

		//Tricky part
		for n := 0; n < numAnchors; n++ {
			//View with the same X coordinate (column)
			vx, err := in.Slice(nil, S(ind*numAnchors+n, in.Shape()[1], step), S(0))
			if err != nil {
				fmt.Println("3")
				panic(err)
			}
			switch in.Dtype() {
			case Float32:
				_, err = tensor.Add(vx, float32(ind), tensor.UseUnsafe())
				break
			case Float64:
				_, err = tensor.Add(vx, float64(ind), tensor.UseUnsafe())
				break
			default:
				panic("Unsupportable type for Yolo")
			}
			if err != nil {
				panic(err)
			}
		}

	}

	anchs := make([]float64, 0)
	for i := 0; i < grid*grid; i++ {
		anchs = append(anchs, currentAnchors...)
	}

	anch := tensor.New(
		tensor.Of(in.Dtype()),
		tensor.WithShape(1, grid*grid*numAnchors, 2),
	)
	for i := range anchs {
		switch in.Dtype() {
		case Float32:
			anch.Set(i, float32(anchs[i]))
			break
		case Float64:
			anch.Set(i, float64(anchs[i]))
		default:
			break
		}
	}
	fmt.Println(in.Dtype(), anch.Dtype())

	switch in.Dtype() {
	case Float32:
		_, err = tensor.Div(anch, float32(stride), tensor.UseUnsafe())
		if err != nil {
			fmt.Println("4")
			panic(err)
		}
		break
	case Float64:
		_, err = tensor.Div(anch, float64(stride), tensor.UseUnsafe())
		if err != nil {
			fmt.Println("5")
			panic(err)
		}
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	if err != nil {
		fmt.Println("6")
		panic(err)
	}

	fmt.Println(anch.Dtype(), in.Dtype())

	vhw, err := in.Slice(nil, nil, S(2, 4))
	expSlice(vhw, err)
	// one := tensor.Ones(anch.Dtype(), vhw.Shape()...)

	_, err = tensor.Mul(vhw, anch, tensor.UseUnsafe())
	if err != nil {
		fmt.Println(vhw.Dtype(), anch.Dtype(), in.Dtype())
		fmt.Println("7")
		panic(err)
	}
	// fmt.Println(one)

	vv, err := in.Slice(nil, nil, S(0, 4))
	if err != nil {
		fmt.Println("8")
		panic(err)
	}

	switch in.Dtype() {
	case Float32:
		_, err = tensor.Mul(vv, float32(stride), tensor.UseUnsafe())
		break
	case Float64:
		_, err = tensor.Mul(vv, float64(stride), tensor.UseUnsafe())
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	if err != nil {
		fmt.Println("9")
		panic(err)
	}
	return in, nil
}
func iou(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

//returns best iou for all predictions and the number of target
func (op *yoloOp) prepBestIous(input, target []float32) [][]float32 {
	ious := make([][]float32, 0)
	imgsize := float32(op.inpDim)
	for i := 0; i < len(input); i = i + op.numClasses + 5 {
		ious = append(ious, []float32{-1, -1})
		r1 := rectifyBox(input[i], input[i+1], input[i+2], input[i+3], op.inpDim)
		for j := 0; j < len(target); j = j + 5 {
			r2 := rectifyBox(target[j+1]*imgsize, target[j+2]*imgsize, target[j+3]*imgsize, target[j+4]*imgsize, op.inpDim)
			curiou := iou(r1, r2)
			if curiou > ious[i/85][0] {
				ious[i/85][0] = curiou
				ious[i/85][1] = float32(j / 5)
			}
		}
	}
	return ious
}

//returns -1 if best anchor is not in the mask, else returns num of box
func (op *yoloOp) prepBestAnchor(target []float32) []int {
	bestAnchors := make([]int, len(target)/5, len(target)/5)
	imgsize := float32(op.inpDim)
	for j := 0; j < len(target); j = j + 5 {
		r2 := rectifyBox(0, 0, target[j+3]*imgsize, target[j+4]*imgsize, op.inpDim)
		var bestiou float32
		bestiou = 0.0
		for i := 0; i < len(op.anchors); i = i + 2 {
			r1 := rectifyBox(0, 0, float32(op.anchors[i]), float32(op.anchors[i+1]), op.inpDim)
			curiou := iou(r1, r2)
			if curiou >= bestiou {
				bestAnchors[j/5] = indexInt(op.mask, i/2)
				bestiou = curiou
			}
		}
	}
	return bestAnchors
}
func indexInt(arr []int, k int) int {
	for i, j := range arr {
		if j == k {
			return i
		}
	}
	return -1
}
func rectifyBox(x, y, h, w float32, imgsize int) image.Rectangle {
	return image.Rect(max(int(x-w/2), 0), max(int(y-h/2), 0), min(int(x+w/2+1), imgsize), min(int(y+h/2+1), imgsize))
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
