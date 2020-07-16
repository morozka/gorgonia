package gorgonia

import (
	"fmt"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
	"hash"
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
		//x, err := Concat(1, x, target[0])
		//fmt.Println(err)
		fmt.Println("concattttttt")
		op := newYoloOp(anchors, mask, imheight, numclasses, ignoreTresh, true)
		retVal, err := ApplyOp(op, x)
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
	return s, nil
}

func (op *yoloOp) Type() hm.Type {

	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t)

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
	fmt.Println(in.Shape(), "test")
	batch := in.Shape()[0]
	stride := int(op.inpDim / in.Shape()[2])
	grid := in.Shape()[2]
	bboxAttrs := 5 + op.numClasses
	numAnchors := len(op.anchors) / 2
	in, _ = op.yoloDoer(in, batch, stride, grid, bboxAttrs, numAnchors, op.anchors)
	yboxes32 := make([]float32, 0)
	switch in.Dtype() {
	case Float32:
		in.Reshape(in.Shape()[0] * in.Shape()[1] * in.Shape()[2])
		for i := 0; i < in.Shape()[0]; i++ {
			buf, _ := in.At(i)
			yboxes32 = append(yboxes32, buf.(float32))
		}
		break
	case Float64:
		//NOT CHECKED!
		in.Reshape(in.Shape()[0] * in.Shape()[1] * in.Shape()[2])
		for i := 0; i < in.Shape()[0]*in.Shape()[1]*in.Shape()[2]; i++ {
			buf, _ := in.At(i)
			yboxes32 = append(yboxes32, buf.(float32))
		}
		break
	default:
		panic("Unsupportable type for Yolo")
	}

	return in, nil
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
			panic(err)
		}

		//Tricky part
		for n := 0; n < numAnchors; n++ {
			//View with the same X coordinate (column)
			vx, err := in.Slice(nil, S(ind*numAnchors+n, in.Shape()[1], step), S(0))
			if err != nil {
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
			panic(err)
		}
		break
	case Float64:
		_, err = tensor.Div(anch, float64(stride), tensor.UseUnsafe())
		if err != nil {
			panic(err)
		}
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	if err != nil {
		panic(err)
	}

	fmt.Println(anch.Dtype(), in.Dtype())

	vhw, err := in.Slice(nil, nil, S(2, 4))
	expSlice(vhw, err)
	// one := tensor.Ones(anch.Dtype(), vhw.Shape()...)

	_, err = tensor.Mul(vhw, anch, tensor.UseUnsafe())
	if err != nil {
		fmt.Println(vhw.Dtype(), anch.Dtype(), in.Dtype())
		panic(err)
	}
	// fmt.Println(one)

	vv, err := in.Slice(nil, nil, S(0, 4))
	if err != nil {
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
		panic(err)
	}
	return in, nil
}

//getTensorData32 - returns all elements of a tensor as an array
func getTensorData32(in tensor.Tensor) []float32 {
	data := make([]float32, 0)
	switch in.Dtype() {
	case tensor.Float32:
		in.Reshape(in.Shape()[0] * in.Shape()[1] * in.Shape()[2])
		for i := 0; i < in.Shape()[0]; i++ {
			buf, _ := in.At(i)
			data = append(data, buf.(float32))
		}
		break
	case tensor.Float64:
		//NOT CHECKED!
		in.Reshape(in.Shape()[0] * in.Shape()[1] * in.Shape()[2])
		for i := 0; i < in.Shape()[0]*in.Shape()[1]*in.Shape()[2]; i++ {
			buf, _ := in.At(i)
			data = append(data, buf.(float32))
		}
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	return data
}
