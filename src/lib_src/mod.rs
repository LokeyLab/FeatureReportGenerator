mod calculate;
mod io;
mod utils;

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
pub use io::*;
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

pub fn pairwise_corr_process(
    exp_df: &DataFrame,
    ref_df: &DataFrame,
    distance: bool,
    idx: usize,
) -> Result<DataFrame, PolarsError> {
    let exp_index = exp_df.select_at_idx(idx).unwrap().clone();
    let ref_index: Vec<String> = ref_df
        .select_at_idx(idx)
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .filter_map(|name| name.map(|n| n.to_string()))
        .collect();

    let mut exp_df_mut = exp_df.clone();
    let mut ref_df_mut = ref_df.clone();
    let _ = exp_df_mut.drop_in_place(exp_df_mut.clone().get_column_names()[0])?;
    let _ = ref_df_mut.drop_in_place(ref_df_mut.clone().get_column_names()[0])?;

    let res_arr = pairwise_corr_process_data(&exp_df_mut, &ref_df_mut, distance).unwrap();
    let mut reporting_df = ndarray_to_dataframe(&res_arr)?;
    reporting_df.set_column_names(&ref_index)?;
    reporting_df.insert_column(idx, exp_index)?;

    Ok(reporting_df)
}

#[cfg(test)]
mod pairwise_corr_test {
    use crate::lib_src::pairwise_corr_process;

    use super::pairwise_corr_process_data;
    use rand::Rng;
    // use ndarray_rand::rand_distr::Uniform;
    // use ndarray_rand::RandomExt;
    use polars::prelude::*;
    fn create_random_dataframe(
        nrows: i32,
        ncols: usize,
        dummy_index: bool,
    ) -> Result<DataFrame, PolarsError> {
        let mut rng = rand::thread_rng();

        // Create a Vec of Series, each with random f64 values
        let mut series_vec = Vec::with_capacity(ncols + if dummy_index { 1 } else { 0 });

        // If dummy_index is true, create an "index" column with sequential integers
        if dummy_index {
            let index_values: Vec<String> = (0..nrows).map(|i| i.to_string()).collect(); // Generating indices
            let index_series = Series::new("index", &index_values);
            series_vec.push(index_series); // Adding the index series as the first column
        }

        // Generate random f64 columns
        for i in 0..ncols {
            let values: Vec<f64> = (0..nrows).map(|_| rng.gen()).collect();
            let series = Series::new(&format!("col{}", i), &values);
            series_vec.push(series);
        }

        // Combine all Series into a DataFrame
        let df = DataFrame::new(series_vec)?;

        // Get the list of column names, sort them, and then select columns in sorted order
        // let mut column_names = df.get_column_names();
        // column_names.sort(); // Sorts column names alphabetically
        // df = df.select(&column_names)?; // Reorder DataFrame based on sorted column names

        Ok(df)
    }

    #[test]
    fn test_helper() -> Result<(), PolarsError> {
        let ncols = 5880;
        let exp_df = create_random_dataframe(384, ncols, false).unwrap().clone();
        let ref_df = create_random_dataframe(20000, ncols, false)
            .unwrap()
            .clone();

        println!("beginning test");
        let data = pairwise_corr_process_data(&exp_df, &ref_df, false).unwrap();
        println!("{:?}", data.shape()); //returns array of 384 x 15000
        Ok(())
    }

    #[test]
    fn test_main_corr() -> Result<(), PolarsError> {
        let ncols = 5880;
        let exp_df = create_random_dataframe(384, ncols, true).unwrap().clone();
        let ref_df = create_random_dataframe(20000, ncols, true).unwrap().clone();

        println!("{exp_df}\n{ref_df}\n------");
        println!("beginning test");
        let data = pairwise_corr_process(&exp_df, &ref_df, false, 0).unwrap();
        println!("{:?} {:?}", data.shape(), data); //returns array of 384 x 15000
        Ok(())
    }

    fn read_csv(filename: &str) -> Result<DataFrame, PolarsError> {
        CsvReader::from_path(filename)?.has_header(true).finish()
    }
    #[test]
    fn test_real_df() -> Result<(), PolarsError> {
        let ref_df = read_csv("/Users/dterciano/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uM_concats_complete/TargetMol_10uM_PMA+NoPMA_longConcat_HD.csv")?;
        let exp_df = read_csv("/Users/dterciano/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uMData/TargetMol_10uM2_1_HD.csv")?;
        println!("{}\n{}\n------", ref_df, exp_df);
        println!("beginning test");
        let data = pairwise_corr_process(&exp_df, &ref_df, true, 0)?;
        println!("{data}");
        Ok(())
    }
}
