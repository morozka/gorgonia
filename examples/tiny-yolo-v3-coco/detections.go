package main

import (
	"image"
	"sort"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

var (
	anchors       = []float32{0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828}
	classesArr    = []string{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}
	scoreTreshold = float32(0.6)
	iouTreshold   = float32(0.2)
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

// ProcessOutput Detection layer
func (net *YOLOv3) ProcessOutput() (Detections, error) {
	outNodes := net.GetOutput()
	outValue := outNodes.Value()
	outTensor := outValue.(tensor.Tensor)

	bb := make(Detections, 0)
	outDense := outTensor.(*tensor.Dense)

	err := outDense.Reshape((net.classesNum+5)*net.boxesPerCell, net.cellsInRow, net.cellsInRow)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape Dense while postprocessing YOLO network output")
	}
	data, err := native.Tensor3F32(outDense)
	if err != nil {
		return nil, errors.Wrap(err, "Can't prepare tensor3_f32 while postprocessing YOLO network output")
	}

	// hardcoded 1.0. Need to evaluate actual (input image size) / (net size) scale.
	rw := float32(net.netSize) / float32(net.netSize)
	rh := float32(net.netSize) / float32(net.netSize)

	for cx := 0; cx < net.cellsInRow; cx++ {
		for cy := 0; cy < net.cellsInRow; cy++ {
			for b := 0; b < net.boxesPerCell; b++ {
				class := make([]float32, net.classesNum)
				channel := b * (net.classesNum + 5)
				tx := data[channel][cx][cy]
				ty := data[channel+1][cx][cy]
				tw := data[channel+2][cx][cy]
				th := data[channel+3][cx][cy]
				tc := data[channel+4][cx][cy]
				for cl := 0; cl < net.classesNum; cl++ {
					class[cl] = data[channel+5+cl][cx][cy]
				}
				finclass := Softmax(class)
				maxProbability, maxIndex := MaxFloat32(finclass)

				x := (float32(cy) + SigmoidF32(tx)) * 32 * rw
				y := (float32(cx) + SigmoidF32(ty)) * 32 * rh

				w := math32.Exp(tw) * anchors[2*b] * 32 * rw
				h := math32.Exp(th) * anchors[2*b+1] * 32 * rh

				sigmoidCoefficient := SigmoidF32(tc)
				finalCoefficient := sigmoidCoefficient * maxProbability
				if finalCoefficient > scoreTreshold {
					box := &DetectionRectangle{
						conf:  sigmoidCoefficient,
						rect:  Rectify(int(x), int(y), int(h), int(w), 416, 416),
						class: classesArr[maxIndex],
						score: maxProbability,
					}
					bb = append(bb, box)
				}
			}
		}
	}
	return nil, nil
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
