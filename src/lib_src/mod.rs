pub(crate) mod calculate;
pub(crate) mod utils;

use ndarray::{Array2, Axis};
use polars::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArrayError {
    #[error("error computing the mean")]
    MeanError,
    #[error("index out of bounds")]
    IndexError,
    #[error("unknown data error")]
    Unknown,
}

use calculate::{corr_dist, pearsonr};
use utils::*;

fn pairwise_corr_process_data(
    exp_df: &DataFrame,
    ref_df: &DataFrame,
    distance: bool,
) -> Result<Array2<f64>, ArrayError> {
    let ref_array = df_to_ndarray(ref_df).unwrap();
    let comp_sig_array = df_to_ndarray(exp_df).unwrap();

    let data: Vec<Vec<f64>> = comp_sig_array
        .axis_iter(Axis(0))
        .map(|comp_row| {
            let comp_sig = comp_row.to_owned();
            if distance {
                corr_dist(&comp_sig, &ref_array).unwrap()
            } else {
                pearsonr(&comp_sig, &ref_array).unwrap()
            }
        })
        .collect();

    let nrows = data.len();
    let ncols = if nrows > 0 { data[0].len() } else { 0 };

    let mut res_arr = Array2::<f64>::zeros((nrows, ncols));
    for (i, row) in data.into_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            res_arr[[i, j]] = val;
        }
    }

    return Ok(res_arr);
}

#[cfg(test)]
mod pairwise_corr_test {
    use super::pairwise_corr_process_data;
    use ndarray::Array2;
    use rand::Rng;
    // use ndarray_rand::rand_distr::Uniform;
    // use ndarray_rand::RandomExt;
    use polars::prelude::*;
    fn create_random_dataframe(nrows: i32, ncols: usize) -> Result<DataFrame, PolarsError> {
        // let nrows = 384;
        // let ncols = 234;
        let mut rng = rand::thread_rng();

        // Create a Vec of Series, each with random f64 values
        let mut series_vec = Vec::with_capacity(ncols);
        for i in 0..ncols {
            // Generate random f64 values for each column
            let values: Vec<f64> = (0..nrows).map(|_| rng.gen()).collect();
            let series = Series::new(format!("col{}", i).as_str(), &values);
            series_vec.push(series);
        }

        // Combine all Series into a DataFrame
        DataFrame::new(series_vec)
    }

    #[test]
    fn test_helper() -> Result<(), PolarsError> {
        let ncols = 5880;
        let exp_df = create_random_dataframe(384, ncols).unwrap().clone();
        let ref_df = create_random_dataframe(20000, ncols).unwrap().clone();

        println!("beginning test");
        let data = pairwise_corr_process_data(&exp_df, &ref_df, false).unwrap();
        println!("{:?}", data.shape()); //returns array of 384 x 15000
        Ok(())
    }
}
