use super::ArrayError;

// use polars::{frame::row::Row, prelude::*};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

fn centered_correlation(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    let umu = u.mean().unwrap();
    let vmu = v.mean().unwrap();
    let u_centered = u - umu;
    let v_centered = v - vmu;
    let uv = (&u_centered * &v_centered).mean().unwrap();
    let uu = (&u_centered * &u_centered).mean().unwrap();
    let vv = (&v_centered * &v_centered).mean().unwrap();
    return (1.0 - uv / (uu * vv).sqrt()).abs(); // dist
}

pub(crate) fn corr_dist(
    comp_sig: &Array1<f64>,
    ref_array: &Array2<f64>,
) -> Result<Vec<f64>, ArrayError> {
    let indices: Vec<usize> = (0..ref_array.nrows()).collect();
    let return_dist: Vec<f64> = indices
        .into_par_iter()
        .map(|idx| {
            let x = ref_array.row(idx).to_owned();
            return centered_correlation(comp_sig, &x);
        })
        .collect();

    Ok(return_dist)
}

fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let sum_x = x.sum();
    let sum_y = y.sum();
    let sum_xy = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi * yi)
        .sum::<f64>();
    let sum_x_sq = x.mapv(|xi| xi * xi).sum();
    let sum_y_sq = y.mapv(|yi| yi * yi).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x_sq - sum_x.powi(2)) * (n * sum_y_sq - sum_y.powi(2))).sqrt();

    numerator / denominator
}

pub(crate) fn pearsonr(
    comp_sig: &Array1<f64>,
    ref_array: &Array2<f64>,
) -> Result<Vec<f64>, ArrayError> {
    let indices: Vec<usize> = (0..ref_array.nrows()).collect();
    let return_corrs: Vec<f64> = indices
        .into_par_iter()
        .map(|i| {
            let x = ref_array.row(i).to_owned();
            pearson_correlation(comp_sig, &x)
        })
        .collect();

    return Ok(return_corrs);
}
