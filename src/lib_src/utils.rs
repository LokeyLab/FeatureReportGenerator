use ndarray::{Array2, Axis};
use polars::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;

pub(crate) fn df_to_ndarray(df: &DataFrame) -> Result<Array2<f64>, PolarsError> {
    let nrows = df.height() as usize;
    let ncols = df.width();
    let mut array = Array2::<f64>::zeros((nrows, ncols));

    // Using Rayon to iterate over columns in parallel
    let results: Result<Vec<(usize, ChunkedArray<Float64Type>)>, PolarsError> = df
        .get_columns()
        .into_par_iter()
        .enumerate()
        .map(|(col_idx, series)| {
            let col = series.f64().unwrap().clone();
            return Ok((col_idx, col));
        })
        .collect();
    let results = results?;

    for (col_idx, col) in results.into_iter() {
        for (row_idx, val) in col.into_iter().enumerate() {
            array[[row_idx, col_idx]] = val.unwrap(); // Unwrap is safe if you know there are no NaNs, otherwise handle errors
        }
    }

    Ok(array)
}

pub(crate) fn ndarray_to_dataframe(array: &Array2<f64>) -> Result<DataFrame, PolarsError> {
    let cols: Vec<Vec<f64>> = array.axis_iter(Axis(1)).map(|col| col.to_vec()).collect();
    let ser: Vec<Series> = cols
        .into_par_iter()
        .enumerate()
        .map(|(i, col)| Series::new(&format!("{}", i), &col))
        .collect();

    DataFrame::new(ser).map_err(PolarsError::from)
}

#[cfg(test)]
mod util_test {
    use super::*;
    use polars::{frame::DataFrame, prelude::*, series::Series};

    #[test]
    fn df_to_ndarray_test() {
        let df = DataFrame::new(vec![
            Series::new("A", vec![1.0, 2.0, 3.0]),
            Series::new("B", vec![4.0, 5.0, 6.0]),
            Series::new("C", vec![7.0, 8.0, 9.0]),
            Series::new("D", vec![7.0, 8.0, 9.0]),
            Series::new("E", vec![7.0, 8.0, 9.0]),
            Series::new("F", vec![7.0, 8.0, 9.0]),
            Series::new("G", vec![7.0, 8.0, 9.0]),
        ])
        .unwrap();
        println!("DF: {:?}", df);
        // Convert DataFrame to ndarray
        let array = df_to_ndarray(&df).unwrap();
        println!("Array:\n{:?}", array);
    }

    #[test]
    fn ndarray_to_df_test() {
        let df = DataFrame::new(vec![
            Series::new("A", vec![1.0, 2.0, 3.0]),
            Series::new("B", vec![4.0, 5.0, 6.0]),
            Series::new("C", vec![7.0, 8.0, 9.0]),
            Series::new("D", vec![7.0, 8.0, 9.0]),
            Series::new("E", vec![7.0, 8.0, 9.0]),
            Series::new("F", vec![7.0, 8.0, 9.0]),
            Series::new("G", vec![7.0, 8.0, 9.0]),
        ])
        .unwrap();
        println!("DF: {:?}", df);
        // Convert DataFrame to ndarray
        let array = df_to_ndarray(&df).unwrap();

        let new_df = ndarray_to_dataframe(&array).unwrap();
        println!("{:?}", new_df);
    }
}
