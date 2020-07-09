package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type yoloOp struct {
	anchors    []float64
	inpDim     int
	numClasses int
}

func newYoloOp(n *Node, anchors []float64, imheight, numclasses int) *yoloOp {
	upsampleop := &yoloOp{
		anchors:    anchors,
		inpDim:     imheight,
		numClasses: numclasses,
	}
	return upsampleop
}

//YoloDetector yolov3 output layer
func YoloDetector(x *Node, anchors []float64, imheight, numclasses int) (*Node, error) {
	// group := encoding.NewGroup("Yolo")
	// xShape := x.Shape()
	op := newYoloOp(x, anchors, imheight, numclasses)
	// _ = group
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
	if _, err := v.Apply(_sigmoidf64, tensor.UseUnsafe()); err != nil {
		panic(err)
	}
}

func (op *yoloOp) Do(inputs ...Value) (retVal Value, err error) {

	in, _ := op.checkInput(inputs...)
	batch := in.Shape()[0]
	stride := int(op.inpDim / in.Shape()[2])
	grid := int(op.inpDim / stride)
	bboxAttrs := 5 + op.numClasses
	numAnchors := len(op.anchors) / 2

	fmt.Println(grid, numAnchors, op.inpDim)

	in.Reshape(batch, bboxAttrs*numAnchors, grid*grid)

	in.T(1, 2)
	in.Transpose()
	in.Reshape(batch, grid*grid*numAnchors, bboxAttrs)

	fmt.Println(in.Shape())
	anchs := make([]float64, 0)
	for i := 0; i < grid*grid; i++ {
		anchs = append(anchs, op.anchors...)
	}
	anch := tensor.New(
		tensor.Of(in.Dtype()),
		tensor.WithBacking(anchs),
		tensor.WithShape(1, grid*grid*numAnchors, 2),
	)

	switch in.Dtype() {
	case Float32:
		_, err = tensor.Div(anch, float32(stride), tensor.UseUnsafe())
		break
	case Float64:
		_, err = tensor.Div(anch, float64(stride), tensor.UseUnsafe())
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	if err != nil {
		panic(err)
	}
	fmt.Println(stride)
	fmt.Println(anch)

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
			tensor.Add(vy, float32(ind), tensor.UseUnsafe())
			break
		case Float64:
			tensor.Add(vy, float64(ind), tensor.UseUnsafe())
			break
		default:
			panic("Unsupportable type for Yolo")
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
	vhw, err := in.Slice(nil, nil, S(2, 4))
	expSlice(vhw, err)
	tensor.Mul(vhw, anch, tensor.UseUnsafe())

	vv, err := in.Slice(nil, nil, S(0, 4))
	if err != nil {
		panic(err)
	}

	switch in.Dtype() {
	case Float32:
		_, err = tensor.Mul(vv, float64(stride), tensor.UseUnsafe())
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
	fmt.Println(stride, in)
	return in, nil
}
