package main

import (
	"encoding/binary"
	"fmt"
	"gorgonia.org/tensor"
	"io/ioutil"
	"math"
	"path/filepath"
	"strconv"
	"strings"
)

// Float32frombytes Converts []byte to float32
func Float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

//PrepareTrain32 - prepares training tensor
func PrepareTrain32(pathToDir string, gridSize int) (tensor.Tensor, error) {
	files, err := ioutil.ReadDir(pathToDir)
	if err != nil {
		return nil, err
	}
	farr := [][]float32{}
	maxLen := gridSize * gridSize
	numTrainFiles := 0
	for _, file := range files {
		cfarr := []float32{}
		fmt.Println(file.IsDir(), filepath.Ext(file.Name()))
		if file.IsDir() || filepath.Ext(file.Name()) != ".txt" {
			continue
		}
		numTrainFiles++
		f, err := ioutil.ReadFile(pathToDir + "/" + file.Name())
		if err != nil {
			return nil, err
		}
		str := string(f)

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
	return tensor.New(tensor.WithShape(numTrainFiles, 1, gridSize, gridSize), tensor.Of(tensor.Float32), tensor.WithBacking(backArr)), nil
}

//GetTensorData32 - returns all elements of a tensor as an array
func GetTensorData32(in tensor.Tensor) []float32 {
	data := make([]float32, 0)
	switch in.Dtype() {
	case tensor.Float32:
		in.Reshape(in.Shape()[0] * in.Shape()[1] * in.Shape()[2])
		for i := 0; i < in.Shape()[0]; i++ {
			buf, _ := in.At(i)
			data = append(data, buf.(float32))
		}
		break
	case tensor.Float64:
		in.Reshape(in.Shape()[0] * in.Shape()[1] * in.Shape()[2])
		for i := 0; i < in.Shape()[0]; i++ {
			buf, _ := in.At(i)
			data = append(data, float32(buf.(float64)))
		}
		break
	default:
		panic("Unsupportable type for Yolo")
	}
	return data
}
