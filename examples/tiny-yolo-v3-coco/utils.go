package main

import (
	"encoding/binary"
	"math"
)

// Float32frombytes Converts []byte to float32
func Float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

//PrepareTrain32 - prepares training tensor
func PrepareTrain32(pathToDir string) (tensor.Tensor, int,error){
	files, err := ioutil.ReadDir(pathToDir)
    if err != nil {
        return nil,0,err
	}
	farr:=[][]float32{}
	maxLen:=0
	numTrainFiles:=0
    for _, file := range files {
		cfarr:=[]float32{}
		fmt.Println(file.IsDir() , filepath.Ext(file.Name()))
		if file.IsDir() || filepath.Ext(file.Name())!=".txt"{continue}
		numTrainFiles++
		f,err:= ioutil.ReadFile(file.Name())
		if err!=nil{
			return nil,0,err
		}
		str := string(f)
		str = strings.ReplaceAll(str,"\n"," ")
		arr:=strings.Split(str, " ")
		for i:=0;i<len(arr);i++{
			if s, err := strconv.ParseFloat(arr[i], 32); err == nil {
				cfarr=append(cfarr,float32(s))
			}else{
				return nil,0,err
			}
		}
		if len(arr)>maxLen{
			maxLen=len(arr)
		}
		farr=append(farr,cfarr)
	}
	backArr:=[]float32{}
	for i:=0;i<len(farr);i++{
		backArr=append(backArr,farr[i]...)
		if len(farr[i])<maxLen{
			zeroes:=make([]float32,maxLen-len(farr[i]))
			backArr=append(backArr,zeroes...)
		}
	}
	return  tensor.New(tensor.WithShape(numTrainFiles, maxLen/5, 5), tensor.Of(tensor.Float32), tensor.WithBacking(backArr)),maxLen/5,nil
}