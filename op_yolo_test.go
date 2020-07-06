package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {
	g := NewGraph()
	inp := NewTensor(g, tensor.Float64, 4, WithShape(1, 255, 13, 13), WithName("inp"), WithInit(GlorotU(1.0)))
	out := Must(YoloDetector(inp, [][2]int{{116, 90}, {156, 198}, {373, 326}}, 416, 80))

	vm := NewTapeMachine(g)
	// if err := Let(inp, inTen); err != nil {
	// panic(err)
	// }
	vm.RunAll()
	vm.Close()
	t.Log("\n", out.Value())
}
