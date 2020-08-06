package main

import (

	"fmt"
	"image"
	"sort"

	"gorgonia.org/tensor"

)

var (
	anchors       = []float32{0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828}
	classesArr    = []string{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}

	scoreTreshold = float32(0.8)
	iouTreshold   = float32(0.3)

)

// DetectionRectangle Representation of detection
type DetectionRectangle struct {
	conf  float32
	rect  image.Rectangle
	class string
	score float32
}

// GetClass Get class of object
func (dr DetectionRectangle) GetClass() string {
	return dr.class
}

// Detections Detection rectangles
type Detections []*DetectionRectangle

func (detections Detections) Len() int { return len(detections) }
func (detections Detections) Swap(i, j int) {
	detections[i], detections[j] = detections[j], detections[i]
}
func (detections Detections) Less(i, j int) bool { return detections[i].conf < detections[j].conf }

// DetectionsOrder Ordering for X-axis
type DetectionsOrder []*DetectionRectangle

func (detections DetectionsOrder) Len() int { return len(detections) }
func (detections DetectionsOrder) Swap(i, j int) {
	detections[i], detections[j] = detections[j], detections[i]
}
func (detections DetectionsOrder) Less(i, j int) bool {
	return detections[i].rect.Min.X < detections[j].rect.Min.X
}

// ProcessOutput Detection layer
func (net *YOLOv3) ProcessOutput() (Detections, error) {

	bb := make(Detections, 0)
	t := net.out[0].Value().(tensor.Tensor)
	att := t.Data().([]float32)

	bb = parseOutToDetections(att, bb, scoreTreshold)
	t = net.out[1].Value().(tensor.Tensor)
	att = t.Data().([]float32)
	bb = parseOutToDetections(att, bb, scoreTreshold)
	fmt.Println(bb)

	bb = nonMaxSupr(bb)
	sort.Sort(DetectionsOrder(bb))

	return bb, nil
}

func parseOutToDetections(att []float32, bb Detections, scoreTreshold float32) Detections {
	for i := 0; i < len(att); i += 85 {
		class := 0
		var maxProbability float32
		for j := 5; j < 5+len(classesArr); j++ {
			if att[i+j] > maxProbability {
				maxProbability = att[i+j]
				class = (j - 5) % len(classesArr)
			}
		}
		if maxProbability*att[i+4] > scoreTreshold {
			box := &DetectionRectangle{
				conf:  att[i+4],
				rect:  Rectify(int(att[i]), int(att[i+1]), int(att[i+2]), int(att[i+3]), 416, 416),
				class: classesArr[class],
				score: maxProbability,
			}
			bb = append(bb, box)
		}

	}
	return bb
}

func nonMaxSupr(detections Detections) Detections {
	//sorts boxes by confidence
	sort.Sort(detections)
	nms := make(Detections, 0)
	if len(detections) == 0 {
		return nms
	}
	nms = append(nms, detections[0])

	for i := 1; i < len(detections); i++ {
		tocheck, del := len(nms), false
		for j := 0; j < tocheck; j++ {
			currIOU := IOUFloat32(detections[i].rect, nms[j].rect)
			if currIOU > iouTreshold && detections[i].class == nms[j].class {
				del = true
				break
			}
		}
		if !del {
			nms = append(nms, detections[i])
		}
	}
	return nms
}
