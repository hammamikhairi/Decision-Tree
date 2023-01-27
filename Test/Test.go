package main

import (
	"fmt"
	"os"

	DecisionTree "github.com/hammamikhairi/Decision-Tree"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func ReadCSVToDF(path string) dataframe.DataFrame {
	csvfile, err := os.Open(path)
	if err != nil {
		panic(err)
	}

	return dataframe.ReadCSV(csvfile)
}

func test() {

	// Load Data
	// tested on the Titanic dataset from kaggle
	// https://www.kaggle.com/competitions/titanic/data
	f := ReadCSVToDF("Data/train.csv")

	f = f.Select([]string{"Survived", "Age", "Fare", "Pclass", "SibSp"}).Filter(
		dataframe.F{Colname: "Age", Comparator: series.GreaterEq, Comparando: 0},
		dataframe.F{Colname: "Fare", Comparator: series.GreaterEq, Comparando: 0},
		dataframe.F{Colname: "Survived", Comparator: series.GreaterEq, Comparando: 0},
		dataframe.F{Colname: "Pclass", Comparator: series.GreaterEq, Comparando: 0},
		dataframe.F{Colname: "SibSp", Comparator: series.GreaterEq, Comparando: 0},
	)

	Y, err := f.Col("Survived").Int()
	if err != nil {
		panic(err)
	}

	tree := DecisionTree.TreeInit(Y, f.Select([]string{"Age", "Fare", "Pclass", "SibSp"}), 100, 4)

	tree.Sprout()

	df := ReadCSVToDF("Data/test.csv").Select([]string{"Age", "Fare", "Pclass", "SibSp"})

	response := tree.Predict(df)

	res, err := ReadCSVToDF("Data/verif.csv").Col("Survived").Int()

	if err != nil {
		panic(err)
	}

	var total int = len(response)
	var passed int = 0

	for index, pred := range response {
		if pred == fmt.Sprint(res[index]) {
			passed++
		} else {
			fmt.Println(pred, res[index])
		}
	}

	fmt.Println(total, passed)

}
