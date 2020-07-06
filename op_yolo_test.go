package gorgonia

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {
	ein := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	inTen := tensor.New(tensor.Of(tensor.Float64), tensor.WithBacking(ein), tensor.WithShape(1, 1, 3, 3))

	tv, er := inTen.Slice(S(0, 1), S(0, 100), S(0, 100), S(0, 100))
	if er != nil {
		panic(er)
	}
	sigmSlice(tv, nil)
	fmt.Println(inTen)

	/* g := NewGraph()
	inp := NewTensor(g, tensor.Float64, 4, WithShape(inTen.Shape()...), WithName("inp"))
	out := Must(YoloDetector(inp, []int{}))

	vm := NewTapeMachine(g)
	if err := Let(inp, inTen); err != nil {
		panic(err)
	}
	vm.RunAll()
	vm.Close()
	t.Log("\n", out.Value())
	*/
}
