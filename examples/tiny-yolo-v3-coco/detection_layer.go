package main

//FOR TESTS ONLY!
import (
	"gorgonia.org/gorgonia"
	"image"
	"sort"

	"github.com/chewxy/math32"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

var (
	anchors       = []float32{0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828}
	classesStr    = []string{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}
	scoreTreshold = float32(0.6)
	iouTreshold   = float32(0.2)
)

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

// ProcessOutput Detection layer
func ProcessOutput(out *gorgonia.Node) (Detections, error) {
	outValue := out.Value()
	outTensor := outValue.(tensor.Tensor)

	bb := make(Detections, 0)
	outDense := outTensor.(*tensor.Dense)
	// TODO: Check the error returned by Reshape?
	err := outDense.Reshape((80+5)*5, 13, 13)
	if err != nil {
		return nil, err
	}
	data, err := native.Tensor3F32(outDense)
	if err != nil {
		return nil, err
	}

	rw := float32(416) / float32(416)
	rh := float32(416) / float32(416)

	for cx := 0; cx < 13; cx++ {
		for cy := 0; cy < 13; cy++ {
			for b := 0; b < 5; b++ {
				class := make([]float32, 80)
				channel := b * (80 + 5)
				tx := data[channel][cx][cy]
				ty := data[channel+1][cx][cy]
				tw := data[channel+2][cx][cy]
				th := data[channel+3][cx][cy]
				tc := data[channel+4][cx][cy]
				for cl := 0; cl < 80; cl++ {
					class[cl] = data[channel+5+cl][cx][cy]
				}
				finclass := Softmax(class)
				maxProbability, clind := MaxFloat32(finclass)

				x := (float32(cy) + Sigmoid(tx)) * 32 * rw
				y := (float32(cx) + Sigmoid(ty)) * 32 * rh

				w := math32.Exp(tw) * anchors[2*b] * 32 * rw
				h := math32.Exp(th) * anchors[2*b+1] * 32 * rh

				sigmoidCoefficient := Sigmoid(tc)
				finalCoefficient := sigmoidCoefficient * maxProbability
				if finalCoefficient > scoreTreshold {
					box := &DetectionRectangle{
						conf:  sigmoidCoefficient,
						rect:  Rectify(int(x), int(y), int(h), int(w), 416, 416),
						class: classesStr[clind],
						score: maxProbability,
					}
					bb = append(bb, box)
				}
			}
		}
	}

	bb = nonMaxSupr(bb)
	sort.Sort(DetectionsOrder(bb))

	return bb, nil
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

// Rectify Creates rectangle
func Rectify(x, y, h, w, maxwidth, maxheight int) image.Rectangle {
	return image.Rect(MaxInt(x-w/2, 0), MaxInt(y-h/2, 0), MinInt(x+w/2+1, maxwidth), MinInt(y+h/2+1, maxheight))
}

// IOUFloat32 Intersection Over Union
func IOUFloat32(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

// MaxInt Maximum between two integers
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MinInt Minimum between two integers
func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Softmax Implementation of softmax
func Softmax(a []float32) []float32 {
	sum := float32(0.0)
	output := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		output[i] = math32.Exp(a[i])
		sum += output[i]
	}
	for i := 0; i < len(output); i++ {
		output[i] = output[i] / sum
	}
	return output
}

// MaxFloat32 Finds maximum in slice of float32's
func MaxFloat32(cl []float32) (float32, int) {
	max, maxi := float32(-1.0), -1
	for i := range cl {
		if max < cl[i] {
			max = cl[i]
			maxi = i
		}
	}
	return max, maxi
}

// Sigmoid Implementation of sigmoid function
func Sigmoid(sum float32) float32 {
	return 1.0 / (1.0 + math32.Exp(-sum))
}
