pub(crate) mod calculate;
pub(crate) mod utils;

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

pub fn pairwise_corr_process(exp_df: &DataFrame, ref_df: &DataFrame, distance: bool) {
    let ref_array = df_to_ndarray(ref_df).unwrap();
    todo!("implement rest");
}
