package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type yoloLayer struct {
	masks          []int
	anchors        [][2]int
	flattenAhcnors []int
	inputSize      int
	classesNum     int
	ignoreThresh   float32
}

func (l *yoloLayer) String() string {
	str := "YOLO layer: "
	for m := range l.masks {
		str += fmt.Sprintf("Mask->%[1]d Anchors->[%[2]d, %[3]d]", l.masks[m], l.anchors[m][0], l.anchors[m][1])
		if m != len(l.masks)-1 {
			str += "\t|\t"
		}
	}
	return str
}

func (l *yoloLayer) Type() string {
	return "yolo"
}

func (l *yoloLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {
	inputN := input[0]
	if len(inputN.Shape()) == 0 {
		return nil, fmt.Errorf("Input shape for YOLO layer is nil")
	}
	fanchors64 := make([]float64, len(l.flattenAhcnors))
	for i := range fanchors64 {
		fanchors64[i] = float64(l.flattenAhcnors[i])
	}
	//TEST ONLY
	fanchors64 = append(fanchors64, fanchors64...)
	yoloNode, err := gorgonia.YoloDetector(inputN, fanchors64, l.masks, l.inputSize, l.classesNum, float64(l.ignoreThresh))
	if err != nil {
		return nil, errors.Wrap(err, "Can't prepare YOLOv3 operation")
	}
	return yoloNode, nil
}
