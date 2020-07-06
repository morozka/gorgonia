package gorgonia

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {
	/* 	g := NewGraph()
	   	inp := NewTensor(g, tensor.Float64, 4, WithShape(1, 255, 13, 13), WithName("inp"), WithInit(GlorotU(1.0)))
	   	out := Must(YoloDetector(inp, [][2]int{{116, 90}, {156, 198}, {373, 326}}, 416, 80))

	   	vm := NewTapeMachine(g)
	   	// if err := Let(inp, inTen); err != nil {
	   	// panic(err)
	   	// }
	   	vm.RunAll()
	   	vm.Close()
	   	t.Log("\n", out.Value()) */

	ein := []float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
		10, 11, 12}
	in := tensor.New(tensor.Of(tensor.Float64), tensor.WithBacking(ein), tensor.WithShape(1, 12, 1))

	step := 3
	for i := 0; i < step; i++ {
		vx, err := in.Slice(nil, S(i*step, i*step+step), S(0))
		if err != nil {
			panic(err)
		}
		fmt.Println(in.Shape()[1])
		vy, err := in.Slice(nil, S(i, 25, step), S(0))
		if err != nil {
			panic(err)
		}

		fmt.Println(i, vx, vy)
	}
}
