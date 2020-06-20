package main

import "fmt"

type yoloLayer struct {
	masks   []int
	anchors [][2]int
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
