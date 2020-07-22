package gorgonia

import (
	"os"
	"testing"

	//"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"

	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
)

func TestYolo(t *testing.T) {
	target, _ := prepareTrain32("/home/smr/go/src/github.com/gorgonia/examples/tiny-yolo-v3-coco/data", 52)
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
	_ = target
	inp2 := NewTensor(g, tensor.Float32, 4,
		WithShape(target.Shape()...),
		WithName("inp2"),
	)
	inp3 := NewTensor(g, tensor.Float32, 3,
		WithShape(output.Shape()...),
		WithName("inp3"),
	)
	out := Must(YoloDetector(inp, []float64{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319}, []int{0, 1, 2}, 416, 80, 0.5, inp2))
	//n0 := Must(Reshape(out, []int{8112 * 85}))
	//n1 := Must(Reshape(inp3, []int{8112 * 85}))
	//n2 := Must(Sub(n1, n0))
	//n3 := Must(Sum(n2, 0))
	vm := NewTapeMachine(g)
	if err := Let(inp, input); err != nil {
		t.Log(err)
	}
	if err := Let(inp2, target); err != nil {
		t.Log(err)
	}
	if err := Let(inp3, output); err != nil {
		t.Log(err)
	}
	err := vm.RunAll()

	t.Log(err)

	vm.Close()
	t.Log("Got:\n", out.Value(), out.Shape())
	t.Log("Expected:\n", output, output.Shape())

	// ff, _ := os.Create("./myout.npy")
	// out.Value().(*tensor.Dense).WriteNpy(ff)

	//if !assert.Equal(t, out.Value().Data(), output.Data(), "Output is not equal to expected value") {
	//panic("NOT EQUEAL")
	//}
}
func prepareTrain32(pathToDir string, gridSize int) (*tensor.Dense, error) {
	files, err := ioutil.ReadDir(pathToDir)
	if err != nil {
		return &tensor.Dense{}, err
	}
	farr := [][]float32{}
	maxLen := gridSize * gridSize
	numTrainFiles := 0
	for _, file := range files {
		cfarr := []float32{}
		if file.IsDir() || filepath.Ext(file.Name()) != ".txt" {
			continue
		}
		numTrainFiles++
		f, err := ioutil.ReadFile(pathToDir + "/" + file.Name())
		if err != nil {
			return &tensor.Dense{}, err
		}
		str := string(f)
		fmt.Println(str)
		str = strings.ReplaceAll(str, "\n", " ")
		arr := strings.Split(str, " ")
		for i := 0; i < len(arr); i++ {
			if s, err := strconv.ParseFloat(arr[i], 32); err == nil {
				cfarr = append(cfarr, float32(s))
			} else {
				return &tensor.Dense{}, err
			}
		}
		farr = append(farr, cfarr)
	}
	backArr := []float32{}
	for i := 0; i < len(farr); i++ {
		backArr = append(backArr, float32(len(farr[i])))
		backArr = append(backArr, farr[i]...)
		if len(farr[i]) < maxLen {
			zeroes := make([]float32, maxLen-len(farr[i])-1)
			backArr = append(backArr, zeroes...)
		}
	}
	return tensor.New(tensor.WithShape(numTrainFiles, 1, gridSize, gridSize), tensor.Of(tensor.Float32), tensor.WithBacking(backArr)), nil
}
