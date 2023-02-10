package decisiontree

import (
	"fmt"
	"math"
	"sort"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

type Counter map[string]int

// The Decision Tree
type DecisionTree struct {
	root *node
}

// Initialize the Decision Tree
//
// Parameters
// ----------
// Y : []int : The target variable to predict
// X : dataframe.DataFrame <float64 & float64> : The features
// maxTreeDepth : int : The maximum depth of the tree
// minDfSplit : int : The minimum number of samples to split a node
func TreeInit(Y []int, X dataframe.DataFrame, maxTreeDepth, minDfSplit int) *DecisionTree {
	tree := &DecisionTree{
		root: NodeInit(Y, X, 0, maxTreeDepth, minDfSplit, "ROOT"),
	}
	return tree
}

func count(Y []int) Counter {
	counter := make(Counter)
	for _, cont := range Y {
		counter[fmt.Sprint(cont)]++
	}

	return counter
}
func countErr(Y []int, err error) Counter {
	counter := make(Counter)
	for _, cont := range Y {
		counter[fmt.Sprint(cont)]++
	}

	return counter
}

func maxCount(counter Counter) string {

	keys := make([]string, 0, len(counter))
	for key := range counter {
		keys = append(keys, key)
	}

	if len(keys) <= 1 {
		return keys[0]
	}

	sort.SliceStable(keys, func(i, j int) bool {
		return counter[keys[i]] > counter[keys[j]]
	})

	return keys[0]
}

func giniImputiry(s0, s1 int) float64 {

	if s0+s1 == 0 {
		return 0.0
	}

	prob0 := float64(s0) / float64(s0+s1)
	prob1 := float64(s1) / float64(s0+s1)

	return 1 - (math.Pow(prob0, 2) + math.Pow(prob1, 2))

}
func setFromList(list []string) (set []string) {
	ks := make(map[string]bool) // map to keep track of repeats

	for _, e := range list {
		if _, v := ks[e]; !v {
			ks[e] = true
			set = append(set, e)
		}
	}
	return
}

func uniqueGotaSeries(s series.Series) series.Series {
	return series.New(setFromList(s.Records()), s.Type(), s.Name)
}

func meth(col []float64) []float64 {
	var methed []float64
	for i := 0; i < len(col)-1; i++ {
		methed = append(methed, (col[i]+col[i+1])/2)
	}

	return methed
}

// Generate tree
func (tree *DecisionTree) Sprout() {
	tree.root.sprout()
}

// Predict the target variable for a given set of features
//
// Parameters
// ----------
// X : dataframe.DataFrame : The features
func (tree *DecisionTree) Predict(data dataframe.DataFrame) []string {

	features := tree.root.data.features
	var predictions []string

	x, _ := data.Dims()
	for i := 0; i < x; i++ {
		nmap := make(map[string]float64)
		for _, feature := range features {
			nmap[feature] = data.Col(feature).Elem(i).Float()
		}
		predictions = append(predictions, tree.root.predict(nmap))
	}

	return predictions
}

// Print the tree in a human readable format
func (tree *DecisionTree) Print() {
	tree.root.print(1)
}
