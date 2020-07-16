package gorgonia

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"

	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
)

func TestYolo(t *testing.T) {
	target, _ := prepareTrain32("/home/smr/go/src/github.com/gorgonia/examples/tiny-yolo-v3-coco/data", 52)
	//t.Log(target, err)
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

	inp2 := NewTensor(g, tensor.Float32, 4,
		WithShape(target.Shape()...),
		WithName("inp2"),
	)
	out := Must(YoloDetector(inp, []float64{10, 13, 16, 30, 33, 23}, []int{0, 1, 2}, 416, 80, 0.5, inp2))

	// t.Log("\n", inp.Value())

	vm := NewTapeMachine(g)
	if err := Let(inp, input); err != nil {
		panic(err)
	}
	if err := Let(inp2, target); err != nil {
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
}
func prepareTrain32(pathToDir string, gridSize int) (tensor.Tensor, error) {
	files, err := ioutil.ReadDir(pathToDir)
	if err != nil {
		return nil, err
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
			return nil, err
		}
		str := string(f)
		fmt.Println(str)
		str = strings.ReplaceAll(str, "\n", " ")
		arr := strings.Split(str, " ")
		for i := 0; i < len(arr); i++ {
			if s, err := strconv.ParseFloat(arr[i], 32); err == nil {
				cfarr = append(cfarr, float32(s))
			} else {
				return nil, err
			}
		}
		farr = append(farr, cfarr)
	}
	backArr := []float32{}
	for i := 0; i < len(farr); i++ {
		backArr = append(backArr, float32(len(farr[i])))
		backArr = append(backArr, farr[i]...)
		if len(farr[i]) < maxLen {
			zeroes := make([]float32, maxLen-len(farr[i]))
			backArr = append(backArr, zeroes...)
		}
	}
	fmt.Println(backArr)
	return tensor.New(tensor.WithShape(numTrainFiles, 1, gridSize, gridSize), tensor.Of(tensor.Float32), tensor.WithBacking(backArr)), nil
}
