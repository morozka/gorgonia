package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type yoloOp struct {
	anchors    [][2]int
	inpDim     int
	numClasses int
}

func newYoloOp(n *Node, anchors [][2]int, imheight, numclasses int) *yoloOp {
	upsampleop := &yoloOp{
		anchors:    anchors,
		inpDim:     imheight,
		numClasses: numclasses,
	}
	return upsampleop
}

//YoloDetector yolov3 output layer
func YoloDetector(x *Node, anchors [][2]int, imheight, numclasses int) (*Node, error) {
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
	if old != nil {
		panic(old)
	}
	switch v.Dtype() {
	case Float32:
		if _, err := v.Apply(_sigmoidf32, tensor.WithReuse(v)); err != nil {
			panic(err)
		}
	case Float64:
		if _, err := v.Apply(_sigmoidf64, tensor.WithReuse(v)); err != nil {
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
		if _, err := v.Apply(_sigmoidf32, tensor.WithReuse(v)); err != nil {
			panic(err)
		}
	case Float64:
		if _, err := v.Apply(_sigmoidf64, tensor.WithReuse(v)); err != nil {
			panic(err)
		}
	default:
		panic("Unsupportable type for Yolo")
	}
}

func (op *yoloOp) Do(inputs ...Value) (retVal Value, err error) {

	// in, _ := op.checkInput(inputs...)
	ein := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	in := tensor.New(tensor.Of(tensor.Float64), tensor.WithBacking(ein), tensor.WithShape(1, 1, 3, 3))
	batch := in.Shape()[0]
	stride := int(op.inpDim / in.Shape()[2])
	grid := int(op.inpDim / stride)
	bboxAttrs := 5 + op.numClasses
	numAnchors := len(op.anchors)

	sh := in.Shape()
	fmt.Println( /* batch, stride, grid, bboxAttrs, */ numAnchors, op.inpDim)
	fmt.Println("slice")
	v, rrr := in.Slice(S(0, sh[0]), S(0, sh[1]), S(0, sh[2]), S(0, 2))
	if rrr != nil {
		panic(rrr)
	}
	fmt.Println("act")
	_, errr := v.Apply(_sigmoidf64, tensor.WithReuse(v))
	if errr != nil {
		panic(errr)
	}

	return nil, nil
	in.Reshape(batch, bboxAttrs*numAnchors, grid*grid)

	in.T(1, 2)
	in.Transpose()
	in.Reshape(batch, grid*grid*numAnchors, bboxAttrs)

	for i := range op.anchors {
		op.anchors[i][0] = op.anchors[i][0] / stride
		op.anchors[i][1] = op.anchors[i][1] / stride
	}

	sh = in.Shape()
	sigmSlice(in.Slice(S(0, sh[0]), S(0, sh[1]), S(0, 1)))
	sigmSlice(in.Slice(S(0, sh[0]), S(0, sh[1]), S(1, 2)))
	sigmSlice(in.Slice(S(0, sh[0]), S(0, sh[1]), S(4, 5)))

	fmt.Println(in.Shape())

	// for batch := range sh[0] {
	// for xy := range sh[1] {
	//
	// }
	//
	// }

	return in, nil
}
