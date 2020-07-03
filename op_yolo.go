package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type yoloOp struct {
	anchors []int
}

func newYoloOp(n *Node, anchors []int) *yoloOp {
	upsampleop := &yoloOp{
		anchors: anchors,
	}
	return upsampleop
}

//YoloDetector yolov3 output layer
func YoloDetector(x *Node, anchors []int) (*Node, error) {
	// group := encoding.NewGroup("Yolo")
	// xShape := x.Shape()
	op := newYoloOp(x, anchors)
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

func (op *yoloOp) Do(inputs ...Value) (retVal Value, err error) {

	in, _ := op.checkInput(inputs...)

	sigm := newElemUnaryOpType(sigmoidOpType, in.Dtype())
	sh := in.Shape()
	v, _ := in.Slice(S(0, sh[0]), S(0, sh[1]), S(0), S(0, sh[3]))
	retv, _ := sigm.Do(v)
	// v.Apply(retv)
	fmt.Println(v)
	fmt.Println(retv)
	fmt.Println(in)
	return in, nil

}
