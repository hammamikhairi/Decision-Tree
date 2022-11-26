package main

import (
	"fmt"
	"os"

	Dtree "DecisionTree/DTree"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func main() {

	csvfile, err := os.Open("Data/train.csv")
	if err != nil {
		panic(err)
	}

	f := dataframe.ReadCSV(csvfile)

	f = f.Select([]string{"Survived", "Age", "Fare"}).Filter(
		dataframe.F{Colname: "Age", Comparator: series.GreaterEq, Comparando: 0},
	).Filter(
		dataframe.F{Colname: "Fare", Comparator: series.GreaterEq, Comparando: 0},
	).Filter(
		dataframe.F{Colname: "Survived", Comparator: series.GreaterEq, Comparando: 0},
	)

	Y, err := f.Col("Survived").Int()
	if err != nil {
		panic(err)
	}

	tree := Dtree.TreeInit(Y, f.Select([]string{"Age", "Fare"}), 2, 20)

	tree.Sprout()

	tree.Print()

	df := dataframe.New(
		series.New([]string{"20", "15"}, series.String, "Age"),
		series.New([]float64{53.025, 14.0}, series.Float, "Fare"),
	)

	fmt.Println(tree.Predict(df))

}
