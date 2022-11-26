# Decision Tree

## What is a Decision Tree

Decision Tree is the most powerful and popular tool for classification and prediction. A Decision tree is a flowchart-like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. ( sited from [here](https://www.geeksforgeeks.org/decision-tree/) )

<p align="center">
  <img src="https://forum.huawei.com/enterprise/en/data/attachment/forum/202103/24/190400o09x7rhnnhy2yon7.png?1.png" alt="Decision Tree"/>
</p>

## Usage

```go

package main

import (
  "fmt"
  "os"

  Dtree "DecisionTree/DTree"

  "github.com/go-gota/gota/dataframe"
  "github.com/go-gota/gota/series"
)

func main() {

  // Load data
  csvfile, err := os.Open("Data/train.csv")
  if err != nil {
    panic(err)
  }

  // convert data into a dataframe
  f := dataframe.ReadCSV(csvfile)

  // chose the column to train on
  Y, err := f.Col("Survived").Int()
  if err != nil {
    panic(err)
  }

  // initialize Decision Tree
  tree := Dtree.TreeInit(Y, f.Select([]string{"Age", "Fare"}), 2, 20)

  // generate Tree
  tree.Sprout()

  // print Tree
  tree.Print()

  // Root
  //    | GINI impurity :  0.48238903404499056
  //    | Class distribution : {0: 424, 1: 290}
  //    | Predicted class :  0
  // --- Split rule :  Fare <= 52.277100000000004
  //       | GINI impurity :  0.4448243103771814
  //       | Class distribution : {0: 389, 1: 195}
  //       | Predicted class :  0
  // ---------- Split rule :  Fare <= 10.48125
  //              | GINI impurity :  0.31596085502704785
  //              | Class distribution : {0: 192, 1: 47}
  //              | Predicted class :  0
  // ---------- Split rule :  Fare > 10.48125
  //              | GINI impurity :  0.48991388363789123
  //              | Class distribution : {0: 197, 1: 148}
  //              | Predicted class :  0
  // --- Split rule :  Fare > 52.277100000000004
  //       | GINI impurity :  0.39349112426035515
  //       | Class distribution : {0: 35, 1: 95}
  //       | Predicted class :  1
  // ---------- Split rule :  Age <= 63.5
  //              | GINI impurity :  0.37696075392150785
  //              | Class distribution : {0: 32, 1: 95}
  //              | Predicted class :  1
  // ---------- Split rule :  Age > 63.5
  //              | GINI impurity :  0
  //              | Class distribution : {0: 3, 1: 0}
  //              | Predicted class :  0


  // prepare a dataframe for prediction
  df := dataframe.New(
    series.New([]string{"20", "15"}, series.String, "Age"),
    series.New([]float64{53.025, 14.0}, series.Float, "Fare"),
  )

  // predict
  fmt.Println(tree.Predict(df)) // -> [1, 0]

}

```