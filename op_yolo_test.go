package gorgonia

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {
	/* g := NewGraph()
	inp := NewTensor(g, tensor.Float64, 4,
		WithShape(1, 255, 13, 13),
		WithName("inp"),
		WithInit(Zeroes()))
	out := Must(YoloDetector(inp, []int{116, 90, 156, 198, 373, 326}, 416, 80))

	// t.Log("\n", inp.Value())

	vm := NewTapeMachine(g)
	vm.RunAll()
	vm.Close()
	t.Log("\n", out.Value())
	f, _ := os.Create("./out")
	defer f.Close()
	f.WriteString(fmt.Sprint(out.Shape(), out.Value() */
	e := []float64{0, 0, 0, 1, 1, 1, 2, 2, 2}
	in := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithBacking(e),
		tensor.WithShape(1, 9, 1),
	)
	// step := 3
	vy, _ := in.Slice(nil, S(1, 9, 3), S(0))
	fmt.Println(vy)
	return
	for ind := 0; ind < 3; ind++ {
		/*vx, err := in.Slice(nil, S(ind*step, ind*step+step), S(0))
		if err != nil {
			panic(err)
		}

		 switch in.Dtype() {
		case Float32:
			if _, err = tensor.Add(vx, float32(ind), tensor.UseUnsafe()); err != nil {
				panic(err)
			}
			break
		case Float64:
			if _, err = tensor.Add(vx, float64(ind), tensor.UseUnsafe()); err != nil {
				panic(err)
			}
			break
		default:
			panic("Unsupportable type for Yolo")
			} */
		// fmt.Println(ind, vx)
		for n := 0; n < 1; n++ {
			vy, err := in.Slice(nil, S(1, 9, 3), S(0))
			fmt.Println(vy)
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
		}

		/*
			step := 2
			fmt.Println(in.Shape()[1])
			for i := 0; i < step; i++ {

				vy, err := in.Slice(nil, S(i, 25, step), S(0))
				if err != nil {
					panic(err)
				}
				fmt.Println("Expected: ")

				for s := i; s < in.Shape()[1]; s += step {
					fmt.Print(ein[s], " ")
				}
				fmt.Println("]")
				fmt.Println("Got:")
				fmt.Println(i, vx, vy)
			}
		*/
	}
	in.Reshape(1, 3, 3)
	fmt.Println(in)

}
