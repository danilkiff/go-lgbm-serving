// Команда dump прогоняет holdout через Go/cgo-предиктор LightGBM и пишет raw
// margin и SHAP contributions в CSV, по строке на каждый вход.
//
// Запустите на двух платформах с ОДНИМИ model.txt и holdout.csv, затем сравните
// выводы: любое различие - кросс-платформенное численное расхождение одной и той
// же модели (см. README, "Численный паритет").
//
//	dump <model.txt> <holdout.csv> <out.csv>
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: dump <model.txt> <holdout.csv> <out.csv>")
		os.Exit(2)
	}
	modelPath, holdoutPath, outPath := os.Args[1], os.Args[2], os.Args[3]

	b, err := lgbm.LoadBooster(modelPath)
	if err != nil {
		fatal(err)
	}
	defer b.Close()

	rows, err := loadCSV(holdoutPath)
	if err != nil {
		fatal(err)
	}

	out, err := os.Create(outPath)
	if err != nil {
		fatal(err)
	}
	defer out.Close()

	if err := writeDump(b, rows, out); err != nil {
		fatal(err)
	}
	fmt.Printf("wrote %s: %d rows, %d features\n", outPath, len(rows), b.NumFeature())
}

// writeDump скорит каждую строку (raw margin + SHAP contributions) и пишет CSV:
// заголовок "raw,c0..cN", затем по строке на вход. Вынесено из main, чтобы было
// тестируемо без запуска процесса.
func writeDump(b *lgbm.Booster, rows [][]float64, out io.Writer) error {
	w := csv.NewWriter(out)
	defer w.Flush()

	header := []string{"raw"}
	for i := 0; i < b.NumFeature()+1; i++ {
		header = append(header, fmt.Sprintf("c%d", i))
	}
	if err := w.Write(header); err != nil {
		return err
	}
	for i, row := range rows {
		raw, err := b.PredictRaw(row)
		if err != nil {
			return fmt.Errorf("row %d: %w", i, err)
		}
		contrib, err := b.PredictContrib(row)
		if err != nil {
			return fmt.Errorf("row %d: %w", i, err)
		}
		rec := make([]string, 0, 1+len(contrib))
		rec = append(rec, strconv.FormatFloat(raw, 'g', 17, 64))
		for _, c := range contrib {
			rec = append(rec, strconv.FormatFloat(c, 'g', 17, 64))
		}
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	return w.Error()
}

func loadCSV(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	out := make([][]float64, 0, len(recs)-1)
	for _, rec := range recs[1:] { // пропустить заголовок
		row := make([]float64, len(rec))
		for j, s := range rec {
			if row[j], err = strconv.ParseFloat(s, 64); err != nil {
				return nil, err
			}
		}
		out = append(out, row)
	}
	return out, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "dump:", err)
	os.Exit(1)
}
