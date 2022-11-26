package Dtree

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

type Data struct {
	rule         string
	features     []string
	counts       map[string]int
	giniImpurity float64
	yhat         string
	nb           int
}

type node struct {
	Y           []int
	X           dataframe.DataFrame
	bestFeature string
	bestValue   string
	depth       int
	maxDepth    int
	minDfSplit  int
	data        *Data
	left        *node
	right       *node
}

func NodeInit(Y []int, X dataframe.DataFrame, depth, maxDepth, minDfSplit int, rule string) *node {

	genCount := count(Y)
	var nd *node = &node{
		Y:          Y,
		X:          X,
		depth:      depth,
		maxDepth:   maxDepth,
		minDfSplit: minDfSplit,
		data: &Data{
			rule:         rule,
			features:     X.Names(),
			counts:       genCount,
			giniImpurity: giniImputiry(genCount["0"], genCount["1"]),
			yhat:         maxCount(genCount),
			nb:           len(Y),
		},
		left:        nil,
		right:       nil,
		bestFeature: "",
		bestValue:   "",
	}

	return nd
}

func (nd *node) split() (string, string) {
	df := nd.X.Copy().Mutate(
		series.New(nd.Y, series.Int, "Y"),
	)

	giniBase := nd.data.giniImpurity
	maxGain := 0.0
	bestFeature := ""
	bestValue := ""

	for _, feature := range nd.data.features {
		sorted := df.Arrange(
			dataframe.Sort(feature),
		)
		xmeans := meth(uniqueGotaSeries(sorted.Col(feature)).Float())

		for _, val := range xmeans {
			leftCounts := countErr(
				sorted.Filter(
					dataframe.F{Colname: feature, Comparator: series.Less, Comparando: val},
				).Col("Y").Int(),
			)
			rightCounts := countErr(
				sorted.Filter(
					dataframe.F{Colname: feature, Comparator: series.GreaterEq, Comparando: val},
				).Col("Y").Int(),
			)

			s0Left, s1Left, s0Right, s1Right := leftCounts["0"], leftCounts["1"], rightCounts["0"], rightCounts["1"]

			totalLeft := s0Left + s1Left
			totalRight := s0Right + s1Right

			weightLeft := float64(totalLeft) / float64(totalLeft+totalRight)
			weightRight := float64(totalRight) / float64(totalLeft+totalRight)

			wGINI := weightLeft*giniImputiry(s0Left, s1Left) + weightRight*giniImputiry(s0Right, s1Right)

			GINIgain := giniBase - wGINI

			if GINIgain >= maxGain {
				bestFeature = feature
				bestValue = fmt.Sprint(val)

				maxGain = GINIgain
			}
		}
	}

	return bestFeature, bestValue
}

func (nd *node) sprout() {

	if nd.depth < nd.maxDepth && nd.data.nb >= nd.minDfSplit {
		bestFeature, bestValue := nd.split()
		nd.bestFeature = bestFeature
		nd.bestValue = bestValue
		df := nd.X.Mutate(
			series.New(nd.Y, series.Int, "Y"),
		)

		if bestFeature == "" {
			panic("bestFeature")
		}

		leftDf, rightDf := df.Filter(
			dataframe.F{Colname: bestFeature, Comparator: series.LessEq, Comparando: bestValue},
		).Copy(), df.Filter(
			dataframe.F{Colname: bestFeature, Comparator: series.Greater, Comparando: bestValue},
		).Copy()
		// NodeInit(Y []int, X dataframe.DataFrame, maxDepth, minDfSplit int)
		leftY, lerr := leftDf.Col("Y").Int()
		rightY, rerr := rightDf.Col("Y").Int()
		if lerr != nil || rerr != nil {
			panic(lerr)
		}
		nd.left = NodeInit(
			leftY,
			leftDf,
			nd.depth+1,
			nd.maxDepth,
			nd.minDfSplit,
			fmt.Sprintf("%s <= %s", bestFeature, bestValue),
		)
		nd.left.sprout()

		nd.right = NodeInit(
			rightY,
			rightDf,
			nd.depth+1,
			nd.maxDepth,
			nd.minDfSplit,
			fmt.Sprintf("%s > %s", bestFeature, bestValue),
		)
		nd.right.sprout()

	}
}

func (nd *node) print(padding int) {
	spaces := strings.Repeat("-", padding*nd.depth)
	fmt.Print(spaces)
	fmt.Println(" Split rule : ", nd.data.rule)
	fmt.Print(strings.Repeat(" ", padding*nd.depth))
	fmt.Println("   | GINI impurity of the node: ", nd.data.giniImpurity)
	fmt.Print(strings.Repeat(" ", padding*nd.depth))
	fmt.Printf("   | Class distribution in the node: {0: %d, 1: %d}\n", nd.data.counts["0"], nd.data.counts["1"])
	fmt.Print(strings.Repeat(" ", padding*nd.depth))
	fmt.Println("   | Predicted class: ", nd.data.yhat)

	if nd.left != nil {
		nd.left.print(padding + 2)
	}

	if nd.right != nil {
		nd.right.print(padding + 2)
	}
}

func (nd *node) predict(to map[string]float64) string {

	if nd.left == nil && nd.right == nil {
		return nd.data.yhat
	}

	parsed, err := strconv.ParseFloat(nd.bestValue, 64)
	if err != nil {
		panic(err)
	}

	if to[nd.bestFeature] <= parsed {
		return nd.left.predict(to)
	} else {
		return nd.right.predict(to)
	}

}
