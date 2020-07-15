package gorgonia

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {
	input := tensor.New(tensor.Of(tensor.Float32))
	r, _ := os.Open("./1input.[(10, 13), (16, 30), (33, 23)].npy")
	input.ReadNpy(r)
	output := tensor.New(tensor.Of(tensor.Float32))
	r, _ = os.Open("./1output.[(10, 13), (16, 30), (33, 23)].npy")
	output.ReadNpy(r)

	g := NewGraph()
	inp := NewTensor(g, tensor.Float32, 4,
		WithShape(input.Shape()...),
		WithName("inp"),
	)

	//inp2 := NewTensor(g, tensor.Float32, 4,
	//	WithShape(input.Shape()...),
	//	WithName("inp2"),
	//)
	// fmt.Println(input)
	out := Must(YoloDetector(inp, []float64{10, 13, 16, 30, 33, 23}, []int{0, 1, 2}, 416, 80, 0.5))

	// t.Log("\n", inp.Value())
	vm := NewTapeMachine(g)
	if err := Let(inp, input); err != nil {
		panic(err)
	}
	vm.RunAll()
	vm.Close()

	t.Log("Got:\n", out.Value())
	t.Log("Expected:\n", output)

	// ff, _ := os.Create("./myout.npy")

	// out.Value().(*tensor.Dense).WriteNpy(ff)
	if !assert.Equal(t, out.Value().Data(), output.Data(), "Output is not equal to expected value") {
		panic("NOT EQUEAL")
	}

	// e := []float64{
	// 	0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0,
	// 	0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1,
	// 	0, 2, 0, 2, 1, 2, 1, 2, 2, 2, 2, 2,
	// }
	/*in := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithBacking(Zeroes()),
		tensor.WithShape(1, 18, 2),
	)
	// in.Reshape()
	// fmt.Println(in)
	grid := 3
	numAnchors := 2
	step := grid * numAnchors
	 for ind := 0; ind < grid; ind++ {
		vx, err := in.Slice(nil, S(ind*step, ind*step+step), S(1))
		if err != nil {
			panic(err)
		}
		switch in.Dtype() {
		case Float32:
			tensor.Add(vx, float32(ind), tensor.UseUnsafe())
			break
		case Float64:
			tensor.Add(vx, float64(ind), tensor.UseUnsafe())
			break
		default:
			panic("Unsupportable type for Yolo")
		}
		// fmt.Println(ind, vx)
		//Tricky part
		for n := 0; n < numAnchors; n++ {
			vy, err := in.Slice(nil, S(ind*numAnchors+n, in.Shape()[1], step), S(0))
			if err != nil {
				panic(err)
			}
			// fmt.Println("VY:", ind, n, vy)
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
	}
	in.Reshape(1, 9, 4)
	for i := 0; i < 9; i++ {
		if i%3 == 0 {
			fmt.Println()
		}
		x, _ := in.Slice(nil, S(i), S(0))
		fmt.Print(x)
	}
	fmt.Println()
	fmt.Println(in)
	f, _ := os.Create("./out")
	defer f.Close()
	f.WriteString(fmt.Sprint(in.Shape(), "\n", in.Data())) */
}
