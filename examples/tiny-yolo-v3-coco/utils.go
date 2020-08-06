package main

import (
	"encoding/binary"
	"image"
	"image/jpeg"
	"math"
	"os"

	"github.com/chewxy/math32"
)

// Float32frombytes Converts []byte to float32
func Float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

// GetFloat32Image Returns []float32 representation of image file
func GetFloat32Image(fname string) ([]float32, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	img, err := jpeg.Decode(file)
	if err != nil {
		return nil, err
	}

	return Image2Float32(img)
}

// Image2Float32 Returns []float32 representation of image.Image
func Image2Float32(img image.Image) ([]float32, error) {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	imgwh := width * height
	imgSize := imgwh * 3

	ans := make([]float32, imgSize)
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := img.At(y, x).RGBA()
			rpix, gpix, bpix := float32(r>>8)/float32(255.0), float32(g>>8)/float32(255.0), float32(b>>8)/float32(255.0)
			ans[y+x*height] = rpix
			ans[y+x*height+imgwh] = gpix
			ans[y+x*height+imgwh+imgwh] = bpix
		}
	}
	return ans, nil
}

// IOUFloat32 Intersection Over Union
func IOUFloat32(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
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

// Rectify Creates rectangle
func Rectify(x, y, h, w, maxwidth, maxheight int) image.Rectangle {
	return image.Rect(MaxInt(x-w/2, 0), MaxInt(y-h/2, 0), MinInt(x+w/2+1, maxwidth), MinInt(y+h/2+1, maxheight))
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

// SigmoidF32 Implementation of sigmoid function for float32
func SigmoidF32(sum float32) float32 {
	return 1.0 / (1.0 + math32.Exp(-sum))
}
